Command-line interface
======================

The unified ``docuverse`` CLI wraps the same Python API used by
:py:meth:`docuverse.SearchEngine.from_preset`. It is installed by
``pip install -e .`` (entry point: ``docuverse = "docuverse.cli:main"``).

Discovery and help::

    docuverse                       # list subcommands
    docuverse presets list          # list named recipes
    docuverse <subcommand> --help

Subcommands
-----------

================ ===============================================================
Subcommand       Purpose
================ ===============================================================
``run``          Full ingest + search + evaluate (legacy ``ingest-and-test``).
``search``       Run a query (or queries file) against an existing index.
``ingest``       Populate an index from a passages file.
``evaluate``     Score a results file against gold qrels.
``presets``      ``list`` / ``show`` / ``dump`` named recipes.
``embed``        Compute embeddings (wraps ``compute_embeddings_from_jsonl.py``).
``convert``      Format conversions (``json-to-jsonl``, ``to-trec``).
``db``           Milvus admin: ``stats`` / ``list`` / ``copy`` / ``query``.
``prepare``      Dataset preparation: ``hf`` (HuggingFace download).
================ ===============================================================

Engine config flags
-------------------

``run``, ``search``, ``ingest``, and ``evaluate`` all accept the same engine
source flags. Precedence (matching the Python API):

#. ``--preset NAME``        — base recipe.
#. ``--config FILE``        — deep-merged on top.
#. ``--override KEY=VALUE`` — applied last, repeatable.
#. Subcommand-specific flags (``--input``, ``--query``, …) win last.

``--override`` values are parsed as YAML, so ``top_k=10`` becomes ``int``
and ``secure=true`` becomes ``bool``. Dotted keys walk into nested dicts.

Examples
--------

.. code-block:: shell

    # Run the full quickstart pipeline:
    docuverse run --config examples/quickstart/recipe.yaml

    # Ingest only:
    docuverse ingest --preset milvus-dense \
      --input examples/quickstart/passages.jsonl \
      --index docuverse_quickstart

    # One-off query against an already-ingested index:
    docuverse search --preset milvus-dense \
      --override index_name=docuverse_quickstart \
      --query "How do plants make energy from sunlight?" --top-k 3

Performance contract
--------------------

The CLI defers heavy imports (``torch``, ``pymilvus``, ``transformers``)
until a subcommand actually needs them. ``docuverse --help`` and
``docuverse presets list`` start in well under 100 ms even when those
optional dependencies are installed; this is enforced by
``tests/test_cli_dispatch.py::test_help_does_not_import_heavy_deps``.
