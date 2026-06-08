Presets
=======

Presets are named recipes — small YAML files under
``docuverse/presets/recipes/`` that lock in a sensible default
configuration for one backend. They're the input to
:py:meth:`docuverse.SearchEngine.from_preset` and to ``docuverse run
--preset``.

Listing and inspecting presets
------------------------------

From Python:

.. code-block:: python

    from docuverse.presets import list_presets, load_preset

    print(list_presets())
    print(load_preset("milvus-dense"))

From the CLI:

.. code-block:: bash

    docuverse presets list
    docuverse presets show milvus-dense
    docuverse presets dump milvus-dense > my-recipe.yaml

Available presets
-----------------

================== ==========================================================
Preset name        Backend / variant
================== ==========================================================
``milvus-dense``    Milvus, dense embeddings (default for the quickstart).
``milvus-sparse``   Milvus, sparse embeddings.
``milvus-bm25``     Milvus, BM25 keyword search.
``milvus-splade``   Milvus, SPLADE-style sparse expansion.
``milvus-hybrid``   Milvus, dense + sparse fusion.
``elastic-dense``   Elasticsearch with a client-side dense embedder.
``elastic-bm25``    Elasticsearch BM25 (no embedder needed).
``elastic-elser``   Elasticsearch with the server-deployed ELSER model.
``chromadb``        ChromaDB, dense embeddings.
``faiss``           FAISS, dense embeddings, in-memory index.
``lancedb``         LanceDB, dense embeddings.
================== ==========================================================

Override merge order
--------------------

When you call ``from_preset(name, **overrides)`` or
``run --preset NAME --config FILE --override key=value``, fields are
deep-merged in this order (last wins):

#. The preset itself (``docuverse/presets/recipes/<name>.yaml``).
#. The ``--config`` file, if any.
#. ``--override key=value`` pairs (parsed as YAML, so ``top_k=10`` is an
   ``int`` and ``secure=true`` is a ``bool``). Dotted keys walk into
   nested dicts.

The resulting dict is what would have been written into a single
``recipe.yaml`` — see ``examples/quickstart/recipe.yaml`` for the
fully-spelled-out form.

Embedding models
----------------

Presets that need a client-side embedder default to one of two
permissively-licensed IBM Granite r2 models:

* ``ibm-granite/granite-embedding-small-english-r2`` — English, fast.
* ``ibm-granite/granite-embedding-97m-multilingual-r2`` — multilingual.

These are the only model identifiers shipped in
``docuverse/presets/recipes/``; the test suite enforces this so that
copy-pasting a preset never silently pulls in a non-default model.
Override ``model_name`` in your own config or recipe if you want
something else.

Writing your own preset
-----------------------

Drop a YAML file into ``docuverse/presets/recipes/<your-name>.yaml``
and it will be discovered by :py:func:`docuverse.presets.list_presets`
on next import. The format is the same as any DocUVerse config file —
see :doc:`cli` for the field reference and the quickstart's
``recipe.yaml`` for a worked example.

See also
--------

* :doc:`quickstart` — the end-to-end Milvus-Lite walkthrough.
* :doc:`cli` — engine-config flag reference.
