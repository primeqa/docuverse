Quickstart
==========

A 10-passage / 5-query toy corpus you can ingest, search, and evaluate
end-to-end in under a minute, with **no external service**: Milvus-Lite
runs embedded inside the Python process and stores everything in a
single file under ``./.docuverse/milvus.db``.

Install
-------

.. code-block:: bash

    pip install -e ".[quickstart]"

The ``quickstart`` extra pulls in ``pymilvus[milvus_lite]`` so the
example runs without standing up an external service.

What's in ``examples/quickstart/``
----------------------------------

================================ ==================================================
File                              Purpose
================================ ==================================================
``passages.jsonl``                10 documents (science + history snippets).
``queries.jsonl``                 5 queries, each with a ``relevant`` field naming
                                  the gold passage id.
``qrels.tsv``                     Same gold judgments in TREC qrels format.
``recipe.yaml``                   A spelled-out config equivalent to
                                  ``from_preset("milvus-dense", ...)``.
================================ ==================================================

Run it three ways
-----------------

All three produce the same ranked results.

Python API — ``from_preset``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from docuverse import SearchEngine

    engine = SearchEngine.from_preset(
        "milvus-dense",
        model_name="ibm-granite/granite-embedding-small-english-r2",
        index_name="docuverse_quickstart",
        input_passages="examples/quickstart/passages.jsonl",
        input_queries="examples/quickstart/queries.jsonl",
        output_file="examples/quickstart/output.json",
    )
    engine.ingest(engine.read_data())

    queries = engine.read_questions()
    results = engine.search(queries)
    print(engine.compute_score(queries, results))

Python API — explicit YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from docuverse import SearchEngine

    engine = SearchEngine(config_or_path="examples/quickstart/recipe.yaml")
    engine.ingest(engine.read_data())
    queries = engine.read_questions()
    print(engine.compute_score(queries, engine.search(queries)))

CLI
~~~

.. code-block:: bash

    docuverse run --config examples/quickstart/recipe.yaml

…or compose preset + overrides directly:

.. code-block:: bash

    docuverse run --preset milvus-dense \
      --override model_name=ibm-granite/granite-embedding-small-english-r2 \
      --override index_name=docuverse_quickstart \
      --override input_passages=examples/quickstart/passages.jsonl \
      --override input_queries=examples/quickstart/queries.jsonl \
      --override output_file=examples/quickstart/output.json

Multilingual corpora
--------------------

Swap the model for the multilingual r2 if your corpus is not English-only:

.. code-block:: yaml

    model_name: ibm-granite/granite-embedding-97m-multilingual-r2

Cleaning up
-----------

The embedded Milvus-Lite database lives at ``./.docuverse/milvus.db``.
Delete that file (or the whole ``.docuverse/`` directory) to start fresh
— ``.docuverse/`` is in ``.gitignore`` so it won't end up in commits.

See also
--------

* :doc:`presets` — the full list of named recipes you can pass to
  ``from_preset`` or ``--preset``.
* :doc:`cli` — the unified ``docuverse`` CLI reference.
