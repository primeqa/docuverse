Troubleshooting
===============

This page catalogs common errors with their verbatim messages, causes, and fixes,
followed by performance and recall debugging checklists.

.. contents:: On this page
   :local:
   :depth: 1


Errors
------

.. _err-unsupported-D:

Unsupported encoder dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    simdq build: D must be one of (384, 768, 1024, 1536); got <D>

:Cause: The embedding dimension produced by your encoder is not in the set of
    dimensions for which native SIMD kernels have been compiled.  This is checked
    before any computation takes place.

:Fix: Switch to a supported model (e.g. a 768-d or 1024-d granite-embedding
    checkpoint) or use ``simdq_d`` to project down to a supported size.  See
    :doc:`parameters` for the full ``simdq_d`` / ``simdq_family`` reference.


.. _err-K-range:

K out of range
~~~~~~~~~~~~~~

.. code-block:: text

    simdq search: K must be in [1, 256]; got <K>

:Cause: The caller requested more than 256 results, or zero/negative results.
    The packed scan kernels use 8-bit result counters, so K is bounded at 256.

:Fix: Reduce ``top_k`` (or ``simdq_k_prime``) to 256 or fewer.  If your
    pipeline genuinely needs more than 256 candidates, run two separate queries
    and merge the result lists downstream.


.. _err-K-prime:

K_prime constraint violated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    simdq search: need K <= K_prime <= 256; got K=<K>, K'=<K_prime>

:Cause: ``K_prime`` (the over-fetch size passed to the quantized scan before
    float rescoring) must satisfy ``K <= K_prime <= 256``.  A value below ``K``
    would silently drop candidates before rescoring; a value above 256 exceeds
    the kernel limit.

:Fix: Ensure ``simdq_k_prime >= top_k`` and ``simdq_k_prime <= 256``.  See
    :doc:`parameters` for ``simdq_rescore_k_prime``.


.. _err-identity-projection:

projection='identity' requires d == D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    simdq build: projection='identity' requires d == D; got D=<D>, d=<d>. Use projection='random_orthogonal' for d=D/2 reductions.

:Cause: ``projection='identity'`` means "no projection matrix — use the
    encoder output as-is", which only makes sense when the projected dimension
    ``d`` equals the full encoder dimension ``D``.  Requesting a smaller ``d``
    with ``identity`` is contradictory.

:Fix: Either set ``simdq_d`` equal to ``D`` (keep full resolution) or set
    ``simdq_projection='random_orthogonal'`` to enable dimensionality reduction
    to ``d = D/2``.  See :doc:`parameters` for the projection options.


.. _err-format-version:

Format version mismatch
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    simdq load: format_version mismatch (file=<file_version>, code=<code_version>). Rebuild the index.

:Cause: The ``meta.json`` in the saved index directory was written by a newer
    (or older) build of the simdq library that uses a different on-disk layout.
    Loading across format versions is not supported.

:Fix: Delete the old index directory and rebuild with the current library
    version.  If you need to keep the existing index, pin the library version
    that wrote it or migrate manually by loading the raw arrays and calling
    ``SimdqIndex.build`` again.


.. _err-encoder-dim:

Encoder/config dimension mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    simdq ingest: encoder produced dim <X> but hidden_dim=<Y>

:Cause: The encoder model returned vectors of dimension ``X`` but the engine
    was configured with ``hidden_dim=Y``.  This typically happens when the
    checkpoint path in the config points to a different model than the one
    originally used to set ``hidden_dim``.

:Fix: Verify that ``encoder_model`` and ``hidden_dim`` in your YAML config
    refer to the same checkpoint.  If you changed the encoder, update
    ``hidden_dim`` to match.  See :doc:`parameters` for ``simdq_hidden_dim``.


.. _err-b-value:

Invalid quantization bit-width
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    simdq build: asymmetric requires b in (1, 2, 4); got <b>

:Cause: The ``b`` parameter controls how many bits per dimension the asymmetric
    quantizer uses.  Only 1, 2, and 4 are supported by the native kernels.

:Fix: Set ``simdq_b`` to 1, 2, or 4 in your config.  Start with ``b=2`` for
    a good recall/speed trade-off; use ``b=4`` when you need recall closest to
    the float baseline.  See :doc:`tuning` for guidance on choosing ``b``.


.. _err-hamming-divisible:

Hamming d not divisible by 64
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    simdq build: hamming requires d divisible by 64; got d=<d>

:Cause: The Hamming kernel packs bits into 64-bit words.  If ``d`` is not a
    multiple of 64 the packing arithmetic is undefined, so the check fires
    before any native code runs.

:Fix: Choose ``simdq_d`` from the set {192, 384, 512, 768, 1024, 1536} — all
    are divisible by 64.  If your current ``simdq_d`` is not in that set it
    is also not in ``SUPPORTED_d`` so the earlier dimension-validation error
    will fire first.


.. _err-no-docs:

No documents survived text filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    simdq ingest: no documents survived text filtering

:Cause: Every document in the corpus was removed by the pre-processing filter
    (empty string, below minimum length, etc.) before encoding began.

:Fix: Check your ``min_doc_length`` setting and verify that the input corpus
    file is not empty or already pre-filtered to zero rows.  Run with
    ``log_level=DEBUG`` to see per-document filter decisions.


Performance debugging
---------------------

**OMP_NUM_THREADS ignored?**
    The SIMD scan kernels read ``simdq_num_threads`` from the engine config, not
    the ``OMP_NUM_THREADS`` environment variable.

    Diagnostic::

        python -c "from docuverse.engines.retrieval.simdq import SimdqEngine; \
                   e = SimdqEngine.from_config('your_config.yaml'); \
                   print(e.config.simdq_num_threads)"

    Fix: Set ``simdq_num_threads: 8`` (or your desired count) explicitly in
    your YAML config.  See :doc:`parameters` for ``simdq_num_threads``.

**mmap thrash?**
    When ``simdq_store_floats=True`` (the default) the engine memory-maps
    ``floats.bin`` for float rescoring.  If this file is larger than available
    RAM the OS will thrash pages on every rescore step.

    Diagnostic:

    .. code-block:: bash

        ls -lh <project_dir>/simdq_data/<index>/floats.bin
        free -h

    Fix: If ``floats.bin`` is close to or larger than free RAM, either set
    ``simdq_store_floats: false`` (disables float rescoring; recall may drop
    slightly) or reduce corpus size.  See :doc:`tuning` for the recall impact.

**AVX-512 not used?**
    The build system selects the widest available SIMD ISA at compile time via
    ``-march=native``.  If the index was built on a machine without AVX-512 the
    resulting shared library will use AVX2 paths instead.

    Diagnostic:

    .. code-block:: bash

        cat /proc/cpuinfo | grep avx512f

    Fix: Rebuild the C extension on a host that has AVX-512 (``avx512f``
    visible in ``/proc/cpuinfo``).  Run ``pip install -e .`` (or
    ``python setup.py build_ext --inplace``) and confirm that the CMake output
    shows ``-march=native`` picking up ``-mavx512f``.


Recall debugging decision tree
-------------------------------

When recall on your corpus is below the float baseline, work down this tree
before assuming the kernel is wrong.

.. code-block:: text

    low NDCG@10 → first try rescore_alpha=10
                  ↓
                  still low → switch from b=1 to b=2
                              ↓
                              still low → drop random_orthogonal, use identity at d=D

See :doc:`parameters` for ``simdq_rescore_alpha`` and :doc:`tuning` for a
worked example of tuning ``b`` and ``rescore_alpha`` together.
