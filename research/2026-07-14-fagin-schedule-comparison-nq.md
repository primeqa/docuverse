# Fagin schedules on real NQ — does TASD, GTA, or GTASD stop earlier?

**Date:** 2026-07-14
**Status:** Measured on real NQ-dev queries; all four schedules parity-verified
(identical exact top-100 at ε=0)
**Files:**
`scripts/fagin_schedule_cost_compare.py` (schedule cost benchmark),
`scripts/fagin_topk_union_depth.py` (union-depth probe),
`docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h` (kernel),
`docuverse/engines/retrieval/fagin/fagin_index.py`

Related: [[2026-07-09-fagin-weighted-sorted-access]] (steepest schedule),
`2026-07-13-fagin-norm-aware-threshold` (the water-filling bound behind GTA/GTASD).

## Question

Among **TA**, **TASD**, and **GTA** (and **GTASD**), is there any reason to
believe one stops earlier? Does any of the "improved" variants pick *better
columns* to explore?

The four schedules vary along **two independent axes** — how they pick the next
column, and what halting threshold they test:

| schedule | column choice | halting threshold |
|---|---|---|
| **TA** (`lockstep`) | round-robin, weight-blind | box sum `Σⱼ qⱼtⱼ` (loose) |
| **TASD** (`steepest`) | advance dim with largest marginal threshold drop | box sum |
| **GTA** (`lockstep_norm`) | round-robin — **identical reads to TA** | norm-aware water-filling (tight) |
| **GTASD** (`steepest_norm`) | steepest + KKT-scaled key | norm-aware |

So the two sub-questions separate cleanly:

- **"Pick better columns?"** → that is the **steepest** axis (TASD / GTASD).
  GTA reads the *exact same lists in the same order* as TA — it only changes the
  stopping test.
- **"Stop earlier?"** → that is the **norm-aware** axis (GTA / GTASD). Its
  threshold is provably `≤` TA's box sum, so at a given cursor state it can only
  halt at the same depth or shallower, never later.

## Setup

Real NQ-dev, first 50 queries, exact top-100 (ε=0, unlimited depth). Vectors are
the fp32 `floats.bin` from each simdq asym index (the same source the FLAT /
Fagin baselines scan); queries from the per-model `.npy`. `batch=2048` (steepest
advances one dim per round, so a large batch keeps round counts sane; matches
`experiments/simdq/nq_fagin.yaml`). Cost metric is **sorted accesses** — the
honest apples-to-apples number, since `rounds` differ by construction (lockstep
advances every active dim per round, steepest one).

All four schedules were cross-checked to return byte-identical exact top-100 on
every query.

```bash
conda activate ndocu
python scripts/fagin_schedule_cost_compare.py --tag "granite-97m-r2 (384d)" \
  --index experiments/nq_new/simq_data/nq_dev-simdq-simdq-granite97m-512-100-20260702 \
  --queries /home/raduf/sandbox2/docuverse/scratch/nq_hw/q97m.npy \
  --nq 50 --topk 100 --batch 2048
```

## Results — mean sorted accesses per query

**granite-97m (384d), N=225,669:**

| sched | sorted accesses | depth | rounds | vs TA |
|---|---:|---:|---:|---:|
| TA | 28,751,954 | 74,875 | 37 | 100.0% |
| GTA | 28,751,954 | 74,875 | 37 | **100.0%** (identical) |
| TASD | 10,020,004 | 225,669 | 4,904 | 34.8% |
| GTASD | 3,490,316 | 225,669 | 1,715 | **12.1%** |

**granite-311m (768d), N=225,669:**

| sched | sorted accesses | depth | rounds | vs TA |
|---|---:|---:|---:|---:|
| TA | 62,946,017 | 81,961 | 40 | 100.0% |
| GTA | 62,946,017 | 81,961 | 40 | **100.0%** (identical) |
| TASD | 23,523,260 | 225,669 | 11,512 | 37.4% |
| GTASD | 7,839,588 | 225,669 | 3,852 | **12.5%** |

## Findings

1. **Better columns (steepest) is the big lever.** TASD cuts sorted access to
   ~35% of TA on column choice alone. This confirms the earlier weighted-sorted-
   access result on the exact top-100 point.

2. **GTA gives *exactly zero* benefit over TA — same accesses, same rounds,
   byte-identical.** The tighter norm-aware threshold is always `≤` the box sum
   in theory, but under lockstep it saves nothing in practice.

3. **GTASD is the clear winner: ~8× fewer sorted accesses than TA** (12% on both
   encoders), a further ~3× on top of TASD.

## Why GTA ≡ TA but GTASD ≫ TASD (the mechanism)

The norm-aware bound only bites when the frontier vector `t` has **large
components** — that is when the ball constraint `‖x‖ ≤ 1` rules out the loose box
corner `x = t` (which for a normalized corpus has `‖t‖ ≫ 1` and is unreachable).
The two column-choice policies produce opposite frontier shapes:

- **Lockstep (TA / GTA)** descends *uniformly*. By the depth where halting is
  possible (~33% into every list), the frontier is deep in **every** column, so
  all `tⱼ` are small, `‖t‖ ≪ 1`, the ball constraint is **slack**, and the
  water-filling bound collapses back to the box sum. → GTA gains nothing.

- **Steepest (TASD / GTASD)** descends *unevenly* — it drives a few high-weight
  dims deep while leaving others shallow, leaving **large `tⱼ` in the shallow
  dims**. The ball constraint is then **active** and the norm-aware bound is
  genuinely tighter. → GTASD halts at 12% vs TASD's 35%.

**The two improvements are synergistic, not additive.** The norm bound is
worthless under uniform descent; it pays off *because* steepest manufactures the
skewed frontier that activates the `‖x‖ ≤ 1` constraint. Plain GTA is not worth
using; the value is in stacking the norm-aware halt on top of steepest.

## Where the time actually goes (measured, single thread)

Per-phase wall-clock, added via `ns_*` timers in the kernel
(`scripts/fagin_phase_timing.py`), first 50 NQ queries, top-100, ε=0,
batch=2048, 1 thread. Mean ms/query:

**granite-97m (384d):**

| sched | total | sorted | random | heap | thresh | random_acc/N | random % of total |
|---|---:|---:|---:|---:|---:|---:|---:|
| TA | 60.97 | 18.68 | 41.84 | 0.22 | 0.23 | 100% | 68.6% |
| GTA | 60.96 | 18.43 | 40.21 | 0.21 | 2.10 | 100% | 66.0% |
| TASD | **53.45** | 7.51 | 44.06 | 0.29 | 0.52 | 100% | 82.4% |
| GTASD | 168.25 | 2.64 | 42.32 | 0.22 | **122.70** | 100% | 25.2% |

**granite-311m (768d):**

| sched | total | sorted | random | heap | thresh | random_acc/N | random % of total |
|---|---:|---:|---:|---:|---:|---:|---:|
| TA | 108.01 | 39.97 | 67.33 | 0.22 | 0.47 | 100% | 62.3% |
| GTA | 113.18 | 39.84 | 68.22 | 0.21 | 4.89 | 100% | 60.3% |
| TASD | **91.15** | 15.37 | 71.79 | 0.42 | 1.11 | 100% | 78.8% |
| GTASD | 621.19 | 5.29 | 67.74 | 0.26 | **547.05** | 100% | 10.9% |

Three findings, two of them surprising:

1. **`random_accesses == N` for every schedule — Fagin scores the entire
   corpus.** Random access (full dot products) is a flat ~42 ms (97m) / ~68 ms
   (311m) regardless of schedule, and it is the largest single phase for TA/GTA/
   TASD. Fagin here is *brute-force scoring with sorted-access bookkeeping on
   top*. Confirms the cost-model hypothesis: **pruning full dot products (Phase 3
   of the plan) is the real lever, not schedule choice.**

2. **GTASD's 8× sorted-access win is a wall-clock *disaster* — it is the
   SLOWEST schedule** (168 ms vs TA's 61 ms; 621 vs 108 at 768d). The
   norm-aware threshold recompute (`ns_threshold`) explodes to 123 ms / 547 ms
   because it runs an O(ndims·90) water-filling solve on *every one* of the
   ~1,700–3,900 rounds. The sorted-access column (2.6 ms) is exactly where the
   access-count promised, but it was never the bottleneck. **This directly
   motivates Phase 2** (warm-start the ternary search; test the halt less often;
   top-M hybrid to cut round count).

3. **The actual wall-clock winner is plain TASD** (53 / 91 ms) — steepest column
   choice cuts sorted access ~2.5× (18.7→7.5 ms) with negligible threshold cost,
   and no water-filling tax. TA and GTA are indistinguishable (~61 ms), as the
   access counts predicted.

So the "cost ≠ latency" caveat resolves emphatically: **sorted-access count is a
poor proxy for latency on dense vectors** — it measures the cheap phase while
random-access scoring (fixed at N) and, for GTASD, threshold recompute dominate.

## Phase 2 — rescuing GTASD's latency (warm-start + batch)

The GTASD threshold tax is `O(rounds × ndims × iters)`. Two exact-preserving
knobs collapse it (any `lam ≥ 0` is a valid dual bound, so a looser solve costs
at most a few extra rounds, never correctness):

1. **Warm-start the water-filling ternary search** from the previous round's
   `lam` (only one dim changed per round, so the optimum barely moves): narrow
   `[lam/4, 4·lam]` bracket, 40 iters vs 60 cold. Threshold phase 123 → 55 ms
   on 97m (2.2×), total 168 → 100 ms.
2. **Larger `fagin_batch_rows`** — threshold cost is *exactly linear in round
   count*, and sorted accesses stay flat (the halt is already precise), so
   bigger batches cut rounds almost for free:

| batch | GTASD total ms (97m) | thresh ms | rounds | | GTASD total ms (311m) | thresh ms | rounds |
|---|---:|---:|---:|---|---:|---:|---:|
| 2048 | 100 | 55 | 1715 | | 341 | 245 | 3852 |
| 8192 | 56 | 14 | 438 | | 152 | 62 | 964 |
| 16384 | 50 | 7 | 217 | | 128 | 31 | 476 |
| 32768 | — | — | — | | 109 | 15 | 233 |

With warm-start + batch≥16384, **GTASD (50 ms on 97m) beats TA (62 ms)** and
matches TASD — while keeping its ~8× sorted-access advantage and staying exact
(verified against brute force at every batch). So GTASD is usable after all; it
was never an algorithmic problem, only an unnecessarily fine round granularity
plus a cold-started inner solve.

## Caveats / next steps

- **Batch granularity.** The halt is only tested at batch boundaries; a finer
  batch could let GTA shave a fraction of one round, but cannot change the
  qualitative picture (uniform frontier ⇒ slack ball constraint).
- **Exact point only.** This is ε=0. Off the exact point the norm-aware bound may
  behave differently under lockstep — an ε>0 sweep would show whether GTA ever
  separates from TA.

## Appendix — union depth to first-surface the top-100

A companion probe (`scripts/fagin_topk_union_depth.py`): for each query, the
sorted-access depth at which every true top-100 doc has appeared in **at least
one** column (`max_{d∈topK} minⱼ rankⱼ(d)`, query-aware direction). This is the
first-surfacing/union bound — a **lower bound** on where plain TA can halt.

| encoder | min | median | mean | p90 | p99 | max | max / N |
|---|---:|---:|---:|---:|---:|---:|---:|
| 97m (384d) | 916 | 1,527 | 1,642 | 2,245 | 2,726 | 2,974 | 1.32% |
| 311m (768d) | 411 | 874 | 989 | 1,597 | 1,976 | 2,138 | 0.95% |

The 768d encoder first-surfaces its top-100 at roughly half the depth of the
384d one — more dimensions spread each doc's mass across more columns, so each
top-100 doc shows up early in *some* list sooner. Even worst case, no query
descends past ~1.3% of the corpus to have seen all its top-100 in at least one
list.
