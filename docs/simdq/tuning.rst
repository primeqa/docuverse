simdq Tuning Guide
==================

This page gives you two things: a decision tree to pick a recipe without
reading any math, and a short "why" for each parameter so you can deviate
from the defaults with confidence.  For the full field reference see
:doc:`parameters`.

Recipe decision tree
====================

.. code-block:: text

    Want the fastest AND smallest lossless mode? (the usual answer)
        → family=hamming + store_floats=True + rescore_alpha=10  (R0)
          1-bit Hamming + fp32 rescore: on NQ it is faster than b=2 at half the
          code bytes and matches fp32 NDCG. See "Performance results" below.

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

    Embeddings anisotropic? (rogue/dead dims, skewed sign bits — check with
    tests/test_embedding_isotropy.py)
        → add standardize=True  (R6 from R3, R7 from R5)
        almost always helps, and is *required* for random_orthogonal to perform

    Reducing dimension (d=D/2)?
        → prefer learned_orthogonal over random_orthogonal (R9 > R7): ITQ packs
          more signal into the reduced bits than a random rotation

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
   * - R6
     - ``family=asymmetric, b=2, projection=identity, d=None,``
       ``store_floats=False, rescore_alpha=1, standardize=True``
     - R3 + standardization; anisotropic encoders, codes-only, no rotation
     - Granite / most transformer encoders (anisotropic by default)
   * - R7
     - ``family=asymmetric, b=4, projection=random_orthogonal, d=D/2,``
       ``store_floats=False, rescore_alpha=1, standardize=True``
     - R5 + standardization; *random*-rotation projected recipe — superseded by
       R9 (learned rotation) at the same byte budget
     - Large anisotropic corpora where index footprint dominates
   * - R8
     - ``family=asymmetric, b=2, projection=learned_orthogonal, d=None,``
       ``store_floats=False, rescore_alpha=1, standardize=True``
     - R6 with a learned (ITQ) rotation; full-dim, best Recall@100 measured,
       NDCG tied with R6 — near-free upgrade when recall matters
     - Anisotropic encoders, codes-only, recall-priority
   * - R9
     - ``family=asymmetric, b=4, projection=learned_orthogonal, d=D/2,``
       ``store_floats=False, rescore_alpha=1, standardize=True``
     - R7 with a learned (ITQ) rotation; the recommended *projected* (d=D/2)
       recipe — beats the random rotation on both NDCG and recall
     - Large anisotropic corpora where index footprint dominates

Performance results
===================

Measured on **Natural Questions dev** (~276k passages, 768-dim
``granite-embedding-311m``; 384-dim is ``granite-97m``, ~267k), search-only with
queries pre-encoded, one identical harness, 16-way concurrent, on a Ryzen 9
9950X3D vs. a local Milvus FLAT (exact fp32) baseline.

.. list-table:: Search speed (ms/query, 16-way concurrent) and quality
   :header-rows: 1
   :widths: 18 16 16 16 16 18

   * - Mode
     - 384d ms/q
     - 768d ms/q
     - NDCG@10 (768d)
     - Match@100 (768d)
     - Code bytes/doc (384/768)
   * - Milvus FLAT fp32
     - 5.0
     - 11.7
     - 0.607 (exact)
     - 0.983
     - 1536 / 3072
   * - simdq ``b=2`` + rescore
     - 0.43
     - 0.81
     - 0.607
     - 0.984
     - 96 / 192
   * - **simdq Hamming (1-bit) + rescore**
     - **0.23**
     - **0.40**
     - **0.607**
     - **0.984**
     - **48 / 96**

Takeaways:

- **1-bit Hamming + fp32 rescore is the fastest and smallest lossless mode** —
  faster than ``b=2`` at half the code bytes, ~29× faster than Milvus FLAT at
  768d, with NDCG@10 identical to fp32 (768d; 384d is within 0.001). The fp32
  rescore (``rescore_alpha=10`` → top-256 candidates re-ranked by exact dot
  product) is what makes the 1-bit ranking lossless.
- The byte counts above are the **scanned codes** (what sets scan bandwidth and
  speed). Rescore additionally reads a handful of fp16 vectors from ``floats.bin``
  per query (memory-mapped, only the top-K′ rows touched), so it is not on the
  hot scan path — but it does dominate *total disk* (``+2·N·d`` bytes).

How these numbers were reached (all internal; no config changes needed):

#. **No per-query BLAS oversubscription.** The identity projection ``W@q`` is
   skipped (it is a no-op) so OpenBLAS cannot spawn a competing thread pool under
   the engine's query-level parallelism.
#. **In-register code unpack.** The ``b=2``/``b=4`` AVX-512 scan unpacks codes
   with ``vpmultishiftqb`` + ``vpshufb`` instead of a scalar stack roundtrip
   (~4× the inner loop).
#. **SoA Hamming storage.** Hamming codes are stored struct-of-arrays so the
   scan never transposes per query; the static codes stay L3-resident and shared
   across concurrent queries (flat Hamming 3.4 → 0.4 ms at 768d).
#. **Batched, threaded search.** ``SimdqEngine.search_all`` encodes all queries
   in one GPU pass, then runs the (GIL-releasing) CPU scans across a thread pool
   — query-level parallelism that works on a single GPU.

IVF outer index (very large corpora only)
------------------------------------------

For Hamming, ``simdq_ivf_nlist > 0`` builds an IVF (inverted-file) outer index:
the corpus is k-means-clustered, codes are reordered so each cluster is
contiguous, and a query scans only the ``simdq_ivf_nprobe`` nearest clusters.
This trades recall for scanning a fraction of the codes.

It only helps when the SoA codes **exceed L3 cache** (roughly tens of millions of
docs). Below that the full flat SoA scan reads each code once and stays
L3-resident, so it beats IVF — e.g. on the 276k NQ corpus, flat Hamming (0.40 ms)
is faster than IVF at any useful ``nprobe`` (nprobe=32 ≈ 2.3 ms, and lossy). Leave
``simdq_ivf_nlist=0`` unless your corpus is large enough that a single pass is
DRAM-bound.

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
``learned_orthogonal`` is the same shape and cost at query time but the matrix
is *fit on the corpus* (see below).

``learned_orthogonal`` — a data-aware (ITQ) rotation
-----------------------------------------------------

A random rotation makes ``sign(Wx)`` an unbiased SimHash code, but it ignores
the data.  ``learned_orthogonal`` instead fits ``W`` with **ITQ** (Iterative
Quantization, Gong & Lazebnik 2011): PCA to ``d`` dims, then an orthogonal
rotation ``R`` that minimises the binary quantization error
``||sign(ZR) − ZR||²`` by alternating sign-binarization with an orthogonal
Procrustes update.  ``W = (P R)ᵀ`` still has orthonormal rows, so the float
rescore tier stays faithful — it is a drop-in for ``random_orthogonal``.  Fit it
on standardized embeddings (``standardize=True``); the fit is a one-time
``itq_iters`` (default 50) passes at build, sampled to ``max_fit_samples`` rows,
and adds nothing at query time.

What it buys, measured on SciFact / ``granite-311m`` (T5 sweep):

.. list-table::
   :header-rows: 1
   :widths: 46 12 12 30

   * - Recipe
     - NDCG@10
     - Recall@100
     - Comparison
   * - R6 (identity, full dim)
     - 0.622
     - 0.937
     - baseline (no rotation)
   * - **R8** (learned, full dim)
     - 0.621
     - **0.947**
     - vs R6: NDCG tied, +0.010 recall
   * - R7 (random, d=D/2)
     - 0.581
     - 0.923
     - baseline (random rotation)
   * - **R9** (learned, d=D/2)
     - **0.599**
     - **0.930**
     - vs R7: **+0.018 NDCG**, +0.007 recall

Two honest takeaways.  At **full dimension** (R8 vs R6) a learned rotation does
*not* beat no rotation on NDCG — identity already keeps all the information, so
ITQ only nudges recall (though R8 has the best Recall@100 of any recipe, so it is
a near-free upgrade when recall is the priority).  ITQ's real win is the
**dimension-reduction path**: when you must go to ``d=D/2``, the learned rotation
packs more signal into the reduced bits than a random one (R9 > R7 on both
metrics), so prefer ``learned_orthogonal`` over ``random_orthogonal`` whenever
``d ≠ D``.  As always, confirm on your own corpus with the recall sweep — the
isotropy diagnostic cannot predict this (rotations are invisible to it).

``standardize`` — center + per-dim standardize before projecting
-----------------------------------------------------------------

``sign(x)`` is only a good SimHash code when the embeddings are roughly
isotropic — centred at the origin, with variance spread evenly across
dimensions.  Real transformer encoders are not: they are anisotropic and carry
*rogue* dimensions (one sign for every vector) and occasional *dead* (constant)
dimensions, so the raw sign bits are unbalanced and waste capacity.  See
``docs/whitepaper_simdq_bit_hashing.md`` §5 and the diagnostic in
``tests/test_embedding_isotropy.py``.

``standardize=True`` fits an affine transform ``x → (x - μ) / σ`` on the corpus
(dead dims are zeroed), persists ``(μ, σ)`` with the index, and applies it
identically to corpus and query *before* the projection ``W``.  It is the cheap
fix for the two problems standardization owns — sign balance and mean offset.
It does **not** decorrelate dimensions; a low ``effective_rank_ratio`` that
survives it is the signal to *also* use ``random_orthogonal`` (the two are
complementary, not alternatives).

Empirically, on SciFact with ``granite-embedding-311m-multilingual-r2``
(the T5 baseline sweep, ``tests/test_simdq_recall.py``):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Recipe
     - NDCG@10
     - Recall@100
   * - R3 (b=2, identity)
     - 0.594
     - 0.910
   * - **R6** (R3 + standardize)
     - **0.622**
     - **0.937**
   * - R5 (b=4, random_orthogonal d=D/2)
     - 0.372
     - 0.813
   * - **R7** (R5 + standardize)
     - **0.581**
     - **0.923**

Two things to read off this table.  First, **standardization is the win**: it
lifts the identity path (R3→R6) and is *decisive* on the projected path
(R5→R7: +0.21 NDCG), because a random rotation applied to un-centred, anisotropic
data scrambles the sign bits and standardization is what lets the rotation help
instead of hurt.  Treat ``standardize=True`` as the default for any transformer
encoder, and as a prerequisite — not an option — whenever
``projection=random_orthogonal``.

Second, **a rotation is mostly a size lever, not a full-dim quality win**.  On
this encoder/corpus the best full-dim recipes — **R6 (identity + standardize)**
at 0.622 / 0.937 and **R8 (learned/ITQ + standardize)** at 0.621 / 0.947 — beat
every ``d=D/2`` recipe on NDCG.  The projected recipes halve the dimension, so
they buy a half-size index at a small quality cost.  When you do reduce, the
*learned* rotation (R9: 0.599 / 0.930) beats the random one (R7: 0.581 / 0.923)
— see ``learned_orthogonal`` above.  Reach for ``d=D/2`` when index footprint
dominates, not for ranking quality.

.. warning::

   The isotropy diagnostic (below) recommends ``random_orthogonal`` whenever a
   low ``effective_rank_ratio`` survives standardization — but that is a
   **geometry heuristic, not an NDCG result**.  ``effective_rank_ratio`` is
   rotation-invariant, so it cannot tell you whether a rotation improves ranking;
   only the recall sweep can.  For ``granite-311m`` it flags "use
   ``random_orthogonal``", yet R6 (no rotation) measured higher.  Always confirm
   the projection choice against ``tests/test_simdq_recall.py``.

To reproduce these numbers or re-measure on a new encoder, populate the
baseline and then gate against it:

.. code-block:: bash

    # writes measured R0..R7 into tests/fixtures/simdq_scifact_baseline.json
    pytest tests/test_simdq_recall.py -m slow --update-baseline

    # re-run without the flag to confirm they reproduce within tolerance
    pytest tests/test_simdq_recall.py -m slow

To check whether an encoder is anisotropic enough to need standardization in
the first place, run the isotropy diagnostic on a sample of its embeddings
(a 2-D ``(N, D)`` ``.npy`` array):

.. code-block:: bash

    pytest tests/test_embedding_isotropy.py -m diagnostic \
        --embeddings-npy embeddings.npy -s

It prints the raw vs. standardized isotropy metrics (it does **not** compute
NDCG/recall — that is the recall sweep's job) and emits a geometry *hint* about
``random_orthogonal`` when a low ``effective_rank_ratio`` survives
standardization.  Use it to decide whether standardization is needed; treat the
rotation hint as a candidate to validate with the recall sweep, per the warning
above.

The fit cost is one pass over the corpus at ingest (mean/variance in float64);
query-time cost is a single ``(q - μ) * σ⁻¹`` vector op.  It is off by default so
existing indexes are unaffected; whether it helps a given encoder is a per-model
decision, gated by the recall sweep.

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
