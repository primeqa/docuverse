simdq Quickstart
================

simdq is an in-process binary-quantized retrieval engine for ≤10M-vector
BEIR-scale evaluation. No external services required — the index lives on
disk under a single directory.

Install
-------

.. code-block:: bash

    pip install -e ".[simdq]"

Verify the import works:

.. code-block:: bash

    python -c "from docuverse.engines.retrieval.simdq import SimdqIndex"

No output means success. Proceed to the config below.

Config file
-----------

Save the following as ``beir_scifact_simdq.yaml``:

.. code-block:: yaml

    search_engine:
      db_engine: simdq
      model_name: ibm-granite/granite-embedding-278m-multilingual
      index_name: scifact_simdq_b2
      project_dir: /tmp/simdq_quickstart
      top_k: 100
      ingestion_batch_size: 256
      max_doc_length: 512
      simdq_family: asymmetric
      simdq_b: 2
      simdq_d: null
      simdq_projection: identity
      simdq_projection_seed: 42
      simdq_store_floats: true
      simdq_rescore_alpha: 10
      simdq_num_threads: 0

    retrieval:
      input_passages: ds:BeIR/scifact:corpus
      input_queries: ds:BeIR/scifact:queries
      ignore_empty_questions: true

    evaluation:
      eval_measure: "ndcg"
      ranks: "10"

See :doc:`parameters` for full ``simdq_*`` field reference.

Run it
------

.. code-block:: bash

    python -m docuverse.utils.ingest_and_test \
        --config beir_scifact_simdq.yaml \
        --actions "ire"

``i`` = ingest, ``r`` = retrieve, ``e`` = evaluate.

Expected output
---------------

After embedding and indexing (a few minutes on first run) you should see::

    ndcg@10: 0.693

The on-disk index lives at::

    /tmp/simdq_quickstart/simdq_data/scifact_simdq_b2/

Delete that directory to force a full re-index on the next run.
See also: :doc:`parameters` (field reference), :doc:`tuning` (recipe selection).
