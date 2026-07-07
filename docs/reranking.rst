Reranking
=========

A reranker re-orders the top results from a retriever using a stronger (and
slower) scoring model. In DocUVerse reranking is an optional pipeline stage
(``R``) that runs after retrieval, taking each query's retrieved passages and
sorting them by a fresh score.

Reranker types
--------------

Rerankers live in ``docuverse/engines/reranking/`` and share the base
:py:class:`~docuverse.engines.reranking.reranker.Reranker` class. The
``reranker_engine`` field selects one:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``reranker_engine``
     - Reranker
   * - ``dense``
     - ``DenseReranker`` — bi-encoder cosine similarity.
   * - ``cross-encoder``
     - ``CrossEncoderReranker`` — scores each ``[query, passage]`` pair
       directly (most accurate, slowest).
   * - ``splade``
     - ``SpladeReranker`` — learned-sparse dot product.
   * - ``none``
     - No reranking.

The ``dense`` and ``splade`` rerankers derive from ``BiEncoderReranker``,
which also implements the score-combination strategies below.

Attaching a reranker (Python)
-----------------------------

The fluent way is :py:meth:`~docuverse.SearchEngine.with_reranker`, which
mutates the engine in place and returns ``self`` for chaining:

.. code-block:: python

    from docuverse import SearchEngine

    engine = SearchEngine.from_preset("milvus-dense").with_reranker(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        reranker_engine="cross-encoder",
    )

The signature is ``with_reranker(reranker_model, reranker_engine="dense",
**kwargs)`` — ``reranker_model`` is a HuggingFace name or local path, and any
extra keyword arguments (e.g. ``reranker_batch_size=64``) are applied to the
reranker config. This composition avoids a preset cross-product of every
retriever × every reranker.

Configuration
-------------

Reranking is configured by ``RerankerArguments``
(``docuverse/engines/search_engine_config_params.py``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Purpose
   * - ``reranker_model``
     - ``None``
     - Model name or path.
   * - ``reranker_engine``
     - ``dense``
     - ``dense`` / ``cross-encoder`` / ``splade`` / ``none``.
   * - ``reranker_top_k``
     - ``-1``
     - How many retrieved docs to rerank (``-1`` = all).
   * - ``reranker_batch_size``
     - ``32``
     - CPU batch size.
   * - ``reranker_gpu_batch_size``
     - ``128``
     - GPU batch size.
   * - ``reranker_combination_type``
     - ``none``
     - How to fuse scores: ``none`` / ``weight`` / ``rrf``.
   * - ``reranker_combine_weight``
     - ``1.0``
     - Weight for ``weight`` fusion (1.0 = reranker only, 0.0 = keep original).
   * - ``reranker_attn_implementation``
     - ``sdpa``
     - Attention backend for the model.
   * - ``reranker_backend``
     - ``None``
     - SentenceTransformer backend (``openvino`` / ``onnx``).

Score combination
-----------------

For the bi-encoder rerankers, ``reranker_combination_type`` decides how the
reranker score interacts with the original retrieval score:

* ``none`` (default) — sort by the reranker score alone.
* ``weight`` — linear blend:
  ``reranker_score * w + retrieval_score * (1 - w)`` where ``w`` is
  ``reranker_combine_weight``.
* ``rrf`` — Reciprocal Rank Fusion of the two rankings.

When ``reranker_top_k`` is positive, only that many passages are reranked; any
tail beyond it is appended unchanged.

Running reranking (CLI)
-----------------------

Add ``R`` to the pipeline actions, after retrieval:

.. code-block:: bash

    python -m docuverse.utils.ingest_and_test --config recipe.yaml --actions "irRe"

Reranking is applied inside ``engine.search()`` whenever a reranker is
configured; results are cached (``.rerank.pkl.bz2``) so a re-run reuses them
unless retrieval changed.

See also
--------

* :doc:`evaluation` — measure the reranker's effect with ``--actions irRe``.
* :doc:`config` — the ``RerankerArguments`` fields.
