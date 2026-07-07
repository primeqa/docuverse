Evaluation
==========

DocUVerse scores a retrieval run against gold relevance judgments with a
self-contained evaluator — no ``pytrec_eval`` or ``ranx`` dependency. Scoring
is the ``e`` stage of the pipeline and is implemented by
:py:class:`~docuverse.utils.evaluator.EvaluationEngine`
(``docuverse/utils/evaluator.py``). It compares the passage IDs returned by
the retriever (and optional reranker) against the gold-relevant IDs stored in
each query's ``relevant`` field.

Metrics
-------

The evaluator computes two families of metrics
(``docuverse/utils/evaluation_output.py``). Rank-based metrics are reported
at each configured rank *k* (default **1, 3, 5**); global metrics are single
values.

``match`` (Match@K, rank-based)
    Binary hit metric — 1 if **any** relevant document appears in the
    top-K results, 0 otherwise. Averaged over all queries.

``mrr`` (Mean Reciprocal Rank, rank-based)
    ``1 / rank`` of the first relevant document, averaged over queries.
    Queries with no relevant document in the ranked list contribute 0.

``ndcg`` (NDCG@K, rank-based)
    Normalized Discounted Cumulative Gain. Graded by
    ``1 / log2(rank + 1)`` and normalized against the ideal DCG
    (capped at the number of gold-relevant documents per query).

``map`` (Mean Average Precision, rank-based)
    Average precision across all relevant documents, averaged over
    queries.

``ece`` (global)
    Expected Calibration Error — measures how well retrieval scores are
    calibrated as probabilities.

``brier`` (global)
    Brier score — mean squared error between retrieval scores and binary
    relevance labels.

.. note::

   ``ece`` and ``brier`` operate on raw retrieval scores, so they are
   most meaningful when the backend returns well-scaled similarity scores
   (e.g., cosine or dot-product in ``[0, 1]``).

Configuration
-------------

Evaluation is controlled by ``EvaluationArguments``
(``docuverse/engines/search_engine_config_params.py``):

================= =========== ==================================================
Field             Default     Purpose
================= =========== ==================================================
``eval_measure``  ``match``   Metric(s) to compute — any comma-separated
                              combination of ``match``, ``mrr``, ``ndcg``,
                              ``map``, ``ece``, ``brier``
                              (e.g. ``"ndcg,match,mrr"``).
``ranks``         ``"1,3,5"`` Comma-separated cut-off ranks at which the
                              rank-based metrics are evaluated; parsed into
                              ``iranks``.
``match_by``      ``id``      How a retrieved passage is matched to gold:
                              ``id`` or ``url``.
``compute_rouge`` ``False``   Also compute ROUGE against answer text.
================= =========== ==================================================

In a YAML config these live under the evaluation section:

.. code-block:: yaml

    evaluate:
      eval_measure: "ndcg,match,mrr"
      ranks: "1,5,10"
      match_by: "id"

Matching against gold
---------------------

Gold judgments come from each query's ``relevant`` field (or a separate qrels
file — see :doc:`data-formats`) and are turned into a ``{query_id: {doc_id:
1}}`` map. A retrieved passage counts as a hit depending on ``match_by``:

* ``id`` (default) — the passage's original document id (``orig_docid``,
  i.e. the id with chunk offsets stripped) is compared against the gold
  document ids. Chunks from the same source document share the same
  ``orig_docid``.
* ``url`` — the passage's normalized ``metadata.url`` is compared against
  the query's ``metadata.norm-gold-urls``; if a chunk has no URL it falls
  back to id matching.

Ground truth format
-------------------

Each query record must contain a ``relevant`` field — a list of corpus
document IDs that are considered gold-relevant for that query:

.. code-block:: json

    {
      "id": "q1",
      "text": "What causes the northern lights?",
      "relevant": ["doc_042", "doc_107"]
    }

TSV query files use a ``relevant`` column that holds a comma-separated
list of the same IDs. When loading from HuggingFace with
``ds:BeIR/…:queries``, qrels are merged automatically — see
:doc:`data-formats` for details.

Running evaluation
------------------

**As part of the pipeline** — include ``e`` in the actions to score
immediately after retrieval:

.. code-block:: bash

    python -m docuverse.utils.ingest_and_test --config recipe.yaml --actions "ire"

Under the hood ``ingest_and_test`` constructs an ``EvaluationEngine`` and
calls ``compute_score(queries, results, model_name=...)``, then writes the
metrics next to the results as a ``.metrics`` file (e.g. ``output.json`` →
``output.metrics``).

**Python API**:

.. code-block:: python

    from docuverse import SearchEngine

    engine = SearchEngine.from_preset(
        "milvus-dense",
        input_queries="queries.jsonl",
        eval_measure="ndcg,match,mrr",
        ranks="1,5,10",
    )
    queries = engine.read_questions()
    results = engine.search(queries)
    print(engine.compute_score(queries, results))

**CLI — standalone**, to score an existing results file without re-running
retrieval:

.. code-block:: bash

    docuverse evaluate \
      --results output.json \
      --queries queries.jsonl \
      --config recipe.yaml

``--results`` and ``--queries`` are required; the config (or a preset)
supplies the data template and the evaluation settings.

Output
------

``compute_score`` returns an ``EvaluationOutput`` that renders as a
human-readable table — per-rank columns (``METRIC@K`` convention, ``M@K``
being Match@K) plus the query counts it scored:

.. code-block:: text

    Model          NDCG@1    NDCG@5    NDCG@10   M@1       M@5       M@10
    my-model       0.412     0.538     0.571     0.402     0.612     0.648

Each value is averaged over the scored queries. Queries whose ``relevant``
field is empty are skipped when ``ignore_empty_questions: true`` is set.

The object also exposes the raw numbers programmatically (``.match[k]``,
``.ndcg[k]``, ``.mrr[k]``, ``.map[k]``, ``.ece``, ``.brier``) along with
``.num_ranked_queries`` and ``.num_judged_queries``.

See also
--------

* :doc:`data-formats` — the qrels / gold-relevance format.
* :doc:`reranking` — evaluate the effect of a reranker with ``--actions irRe``.
* :doc:`quickstart` — end-to-end example including evaluation.
* :doc:`config` — full configuration reference.
* :doc:`cli` — the ``evaluate`` subcommand.
