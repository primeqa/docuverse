simdq Tuning Guide
==================

This page gives you two things: a decision tree to pick a recipe without
reading any math, and a short "why" for each parameter so you can deviate
from the defaults with confidence.  For the full field reference see
:doc:`parameters`.

Recipe decision tree
====================

.. code-block:: text

    Variable-norm corpus (e.g. heterogeneous text length)?
        yes → use store_floats=True + rescore_alpha=10 (R0 or R2)
        no  → codes-only is fine

    Disk constrained?
        yes → b=1 + rescore_alpha=10  (R2: small codes, fp16 floats)
        no  → b=2 no rescore           (R3: larger codes, no floats)

    Latency-tight, RAM-cheap?
        → b=2, store_floats=False, num_threads=#physical cores  (R3)

    Want to halve the index?
        → b=2 d=D/2 random_orthogonal  (R4) — same recall as R3 in many cases
        → b=4 d=D/2 random_orthogonal  (R5) — same byte budget as R3, more bits

.. list-table:: Recipe reference
   :header-rows: 1
   :widths: 8 42 28 22

   * - Recipe
     - Configuration
     - When to pick this
     - Example corpus
   * - R0
     - ``family=hamming, b=1, projection=identity, d=None,``
       ``store_floats=True, rescore_alpha=10``
     - 1-bit Hamming + rescore; variable-norm corpora, disk-constrained
     - BEIR-scale heterogeneous text (e.g. NQ, HotpotQA)
   * - R1
     - ``family=asymmetric, b=1, projection=identity, d=None,``
       ``store_floats=False, rescore_alpha=1``
     - 1-bit asymmetric codes-only; uniform-norm corpora, latency-bound
     - Short uniform passages (FEVER, FiQA)
   * - R2
     - ``family=asymmetric, b=1, projection=identity, d=None,``
       ``store_floats=True, rescore_alpha=10``
     - 1-bit asym + rescore; disk-cheap, recall-priority
     - BEIR variable-norm with disk available (SciFact, Trec-COVID)
   * - R3
     - ``family=asymmetric, b=2, projection=identity, d=None,``
       ``store_floats=False, rescore_alpha=1``
     - 2-bit asym codes-only; latency-tight, RAM-cheap, uniform corpus
     - Standard balanced default for medium-sized corpora
   * - R4
     - ``family=asymmetric, b=2, projection=random_orthogonal, d=D/2,``
       ``store_floats=False, rescore_alpha=1``
     - 2-bit asym + d=D/2; index-size-halved at minor recall cost
     - Large corpora where index footprint dominates
   * - R5
     - ``family=asymmetric, b=4, projection=random_orthogonal, d=D/2,``
       ``store_floats=False, rescore_alpha=1``
     - 4-bit asym + d=D/2; same bytes as R3 with finer quantization levels
     - Recall-sensitive large corpora

Per-parameter math intuition
=============================

``b`` — bits per dimension
---------------------------

1-bit gives binary Hamming; 2-bit gives ±1, ±3 levels; 4-bit gives 16 levels.
The asymmetric estimator's variance shrinks like 1/2^(2b) — so b=2 typically
halves the rescoring need vs b=1.
See ``docs/whitepaper_simdq_bit_hashing.md`` §4.

``d`` — reduced dimension
--------------------------

The Johnson–Lindenstrauss lemma says a random orthogonal projection preserves
cosine within ε with high probability for ``d ≳ log(N)/ε²``.  At N=10M,
d=384 gives ε≈0.05, meaning recall degrades only mildly relative to using the
full dimension D.
See ``docs/whitepaper_simdq_bit_hashing.md`` §3.

``projection``
--------------

``identity`` is free but requires ``d=D`` (no reduction).
``random_orthogonal`` costs an N×d×D matmul at ingest time when d=D/2, paid
once at index build and not at query time.

``store_floats`` and ``rescore_alpha``
--------------------------------------

Two-stage rescoring costs +2·N·d bytes on disk (fp16 floats) and +α·d
floating-point multiply-adds per query.  Rule of thumb: α=10 closes roughly
80% of the recall gap to the float baseline, so it is almost always worth the
latency overhead when disk is available.

``num_threads``
---------------

Controls the OMP thread count for the inner scan loop.  The sweet spot is
``#physical cores`` for memory-bound asymmetric scans; hyperthreads rarely
help because the bottleneck is memory bandwidth, not compute.

What to measure when tuning
============================

When iterating on a recipe, collect the following metrics on a held-out query
set before committing to a configuration:

- **NDCG@10** — primary ranking-quality signal; compare against the float
  baseline to quantify quantization loss.
- **Recall@100** — measures whether the right documents reach the rescore
  window; a low value here cannot be recovered by the reranker.
- **p50 / p99 query latency** — p99 is often dominated by GC or thread
  contention; tune ``num_threads`` if p99 >> p50.
- **On-disk bytes** — ``codes.bin`` + ``floats.bin``; verify the footprint
  meets your storage budget before full-corpus ingest.
- **Peak RSS** — codes are memory-mapped but floats are loaded for rescore;
  measure under realistic concurrent-query load.
