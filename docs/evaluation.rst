Evaluation
==========

DocUVerse scores a retrieval run against gold relevance judgments with a
self-contained evaluator — no ``pytrec_eval`` or ``ranx`` dependency. Scoring
is the ``e`` stage of the pipeline and is implemented by
:py:class:`~docuverse.utils.evaluator.EvaluationEngine`
(``docuverse/utils/evaluator.py``).

Metrics
-------

The evaluator computes two families of metrics
(``docuverse/utils/evaluation_output.py``):

* **Rank-based** (reported at each configured rank *k*): ``match``, ``mrr``,
  ``ndcg``, ``map``.
* **Global** (single value): ``ece`` (expected calibration error),
  ``brier`` (Brier score).

By default metrics are reported at ranks **1, 3, 5**. ``match@k`` is the
fraction of queries whose gold document appears in the top *k* — a simple hit
rate.

Configuration
-------------

Evaluation is controlled by ``EvaluationArguments``
(``docuverse/engines/search_engine_config_params.py``):

================= =========== ==================================================
Field             Default     Purpose
================= =========== ==================================================
``eval_measure``  ``match``   Metric(s) to compute — one of ``match``, ``mrr``,
                              ``ndcg``, ``ece``, ``brier`` (comma-separated for
                              several).
``ranks``         ``"1,3,5"`` Comma-separated ranks; parsed into ``iranks``.
``match_by``      ``id``      How a retrieved passage is matched to gold:
                              ``id`` or ``url``.
``compute_rouge`` ``False``   Also compute ROUGE against answer text.
================= =========== ==================================================

Matching against gold
---------------------

Gold judgments come from each query's ``relevant`` field (or a separate qrels
file — see :doc:`data-formats`) and are turned into a ``{query_id: {doc_id:
1}}`` map. A retrieved passage counts as a hit depending on ``match_by``:

* ``id`` (default) — the passage's original document id (chunk offsets
  stripped) is compared against the gold document ids.
* ``url`` — the passage's ``metadata.url`` is compared against the query's
  normalized gold URLs; if a chunk has no URL it falls back to id matching.

Running evaluation
------------------

Include ``e`` in the pipeline actions to score immediately after retrieval:

.. code-block:: bash

    python -m docuverse.utils.ingest_and_test --config recipe.yaml --actions "ire"

or score an existing results file on its own:

.. code-block:: bash

    docuverse evaluate --config recipe.yaml --input results.jsonl

Under the hood ``ingest_and_test`` constructs an ``EvaluationEngine`` and
calls ``compute_score(queries, results, model_name=...)``, then writes the
metrics next to the results as a ``.metrics`` file.

Output
------

``compute_score`` returns an ``EvaluationOutput`` that renders as a
human-readable table — per-rank columns plus the query counts it scored:

.. code-block:: text

    Model    M@1     M@3     M@5
    model    0.95    0.87    0.82

The object also exposes the raw numbers programmatically (``.match[k]``,
``.ndcg[k]``, ``.mrr[k]``, ``.map[k]``, ``.ece``, ``.brier``) along with
``.num_ranked_queries`` and ``.num_judged_queries``.

See also
--------

* :doc:`data-formats` — the qrels / gold-relevance format.
* :doc:`reranking` — evaluate the effect of a reranker with ``--actions irRe``.
* :doc:`cli` — the ``evaluate`` subcommand.
