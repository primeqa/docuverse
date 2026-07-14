# Fagin Candidate Pruning & Schedule-Cost Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Fagin TA on dense embeddings faster by (a) confirming where time actually goes, (b) reducing per-round overhead of the steepest schedules, and (c) pruning full-dot-product scoring so Fagin stops behaving like brute force.

**Architecture:** The native kernel `fagin_ta.h` currently computes a full D-dim dot product for *every* document it surfaces via sorted access (`random_accesses == N` on NQ — literally brute force in the scoring phase). The k-th-best heap threshold gates only *halting*, never *whether to score a candidate*. This plan first instruments the kernel to measure the real time split (Phase 1), then attacks per-round overhead in the steepest schedules (Phase 2), then adds score-gated candidate pruning and block-max skipping so the k-th threshold prevents full dot products and whole-block sorted access (Phase 3 — the potential order-of-magnitude win). Exactness at epsilon=0 is preserved throughout: any per-list advancement schedule and any upper-bound-based skip is exact (see `research/2026-07-09-fagin-weighted-sorted-access.md`).

**Tech Stack:** C (single-header kernel `fagin_ta.h`, CPython C-API binding `module.c`), Python (`FaginIndex`), numpy, pytest, conda env `ndocu`.

**Baseline measurements (first 50 NQ-dev queries, top-100, epsilon=0, batch=2048):**

| sched | sorted_acc | random_acc (== full dots) | rand/N |
|---|---:|---:|---:|
| TA (lockstep) | 28,751,954 | 225,669 | 100% |
| GTA (lockstep_norm) | 28,751,954 | 225,669 | 100% |
| TASD (steepest) | 10,020,004 | 225,669 | 100% |
| GTASD (steepest_norm) | 3,490,316 | 225,669 | 100% |

(97m, D=384, N=225,669. See `research/2026-07-14-fagin-schedule-comparison-nq.md`.)

**Key insight driving the plan:** the 8× sorted-access win of GTASD is on the *cheap* phase. Random access (full dot products) is identical (== N) across all schedules and dominates cost. Phase 3 is where the real win is, but Phase 1 must confirm the cost model before we touch the architecture.

---

## Reference facts (for any engineer picking this up cold)

- **Env:** `conda activate ndocu`. Build the extension after any C change:
  `python setup.py build_ext --inplace` (compiles `module.c`, which `#include`s
  `fagin_ta.h`). No separate compile step for the header.
- **Kernel entry point:** `fagin_ta_search(...)` in
  `docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h` (signature at
  line ~224). Two branches: `if (!is_steepest)` = lockstep (TA/GTA);
  `else` = steepest (TASD/GTASD).
- **Stats struct:** `fagin_stats_t` (`fagin_ta.h:57-64`): `depth`,
  `sorted_accesses`, `random_accesses`, `rounds`, `exhausted`. It is `memset` to
  0 at entry (line 232) and returned to Python as a dict by `module.c`
  (`py_fagin_search`, builds the stats dict near line 540-560).
- **Full dot product:** `fagin_dot(q, Y + id*D, D)` (`fagin_ta.h:87-93`),
  auto-vectorized. This is the "random access" whose count we must cut.
- **Halting threshold:** box sum `Σ qⱼtⱼ` or, for norm-aware, `fagin_threshold_norm(...)`
  (`fagin_ta.h:208`). The k-th best score is `fagin_key_score(simdq_topk_threshold(&heap))`.
- **Python API:** `FaginIndex.build(vecs)` then
  `idx.search(q, K, batch, epsilon, max_depth, num_threads, schedule)` returns
  `(idxs, scores, stats_dict)`. `fagin_index.py:133-139`.
- **Schedule name → int:** `{"lockstep":0, "steepest":1, "lockstep_norm":2, "steepest_norm":3}`
  (`fagin_index.py` `SCHEDULES`).
- **Test data:** simdq index dir
  `experiments/nq_new/simq_data/nq_dev-simdq-simdq-granite97m-512-100-20260702`
  (fp32 vectors in `floats.bin`, loaded via `SimdqIndex.load(dir).floats_mmap`);
  queries `/home/raduf/sandbox2/docuverse/scratch/nq_hw/q97m.npy`.
- **Existing measurement scripts:** `scripts/fagin_schedule_cost_compare.py`,
  `scripts/fagin_topk_union_depth.py`.
- **Parity invariant:** every schedule + any pruning MUST return identical exact
  top-K at epsilon=0. `tests/test_fagin_native.py` asserts brute-force parity;
  never weaken it.

---

## Phase 1 — Confirm the cost model (measure before changing anything)

**Rationale:** I *claimed* random-access scoring dominates wall-clock. Systematic-debugging discipline: prove it before optimizing. If sorted access or heap ops actually dominate, Phase 3 priorities change. Phase 1 changes no algorithm behavior — it only adds timers, so it cannot alter results.

### Task 1.1: Add phase timers to the kernel stats struct

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h`
  (struct at 57-64; both loop branches)
- Modify: `docuverse/engines/retrieval/simdq/_native/bindings/module.c`
  (stats dict builder in `py_fagin_search`)
- Test: `tests/test_fagin_native.py`

- [ ] **Step 1: Extend `fagin_stats_t` with nanosecond phase accumulators**

In `fagin_ta.h`, extend the struct (keep existing fields; add at the end so
field order / existing code is undisturbed):

```c
typedef struct {
    int64_t depth;
    int64_t sorted_accesses;
    int64_t random_accesses;
    int64_t rounds;
    int     exhausted;
    // Phase 1 instrumentation: wall-clock nanoseconds per phase. Populated only
    // when compiled; zero-overhead monotonic clock reads. See
    // docs/superpowers/plans/2026-07-14-fagin-candidate-pruning.md
    int64_t ns_sorted;    // Phase A: sorted access (gather unseen candidates)
    int64_t ns_random;    // Phase B: full dot products
    int64_t ns_heap;      // Phase C: heap maintenance
    int64_t ns_threshold; // halting-threshold computation (incl. water-filling)
    int64_t ns_total;     // whole fagin_ta_search call
} fagin_stats_t;
```

- [ ] **Step 2: Add a monotonic-nanosecond helper near the top of `fagin_ta.h`**

Add after the `#include`s (needs `<time.h>`):

```c
#include <time.h>
static inline int64_t fagin_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}
```

- [ ] **Step 3: Wrap each phase with timers in BOTH loop branches**

In the lockstep branch (`if (!is_steepest)`): capture `int64_t _t0 = fagin_now_ns();`
before Phase A (line ~292), and after each phase add the delta to the matching
accumulator. Example for Phase B (dot products, lines ~309-313):

```c
int64_t _tb = fagin_now_ns();
#pragma omp parallel for schedule(static)
for (int64_t c = 0; c < n_cand; c++)
    cand_scores[c] = fagin_dot(q, Y + (size_t)cand[c] * (size_t)D, D);
stats->random_accesses += n_cand;
stats->ns_random += fagin_now_ns() - _tb;
```

Apply the same pattern to Phase A (`ns_sorted`), Phase C (`ns_heap`), and the
threshold block (`ns_threshold`) in both branches. Record `ns_total` from the
very start of `fagin_ta_search` (right after `memset`) to just before `return n;`.

- [ ] **Step 4: Export the new fields in `module.c`**

In `py_fagin_search`'s stats-dict construction (near line 540-560), add each new
field. Follow the exact existing pattern used for `sorted_accesses`, e.g.:

```c
PyDict_SetItemString(stats_dict, "ns_random", PyLong_FromLongLong(st.ns_random));
PyDict_SetItemString(stats_dict, "ns_sorted", PyLong_FromLongLong(st.ns_sorted));
PyDict_SetItemString(stats_dict, "ns_heap", PyLong_FromLongLong(st.ns_heap));
PyDict_SetItemString(stats_dict, "ns_threshold", PyLong_FromLongLong(st.ns_threshold));
PyDict_SetItemString(stats_dict, "ns_total", PyLong_FromLongLong(st.ns_total));
```

- [ ] **Step 5: Rebuild the extension**

Run: `python setup.py build_ext --inplace`
Expected: compiles cleanly, copies the `.so` into
`docuverse/engines/retrieval/simdq/`.

- [ ] **Step 6: Write a test asserting the new keys exist and are consistent**

Add to `tests/test_fagin_native.py`:

```python
def test_phase_timers_present_and_consistent():
    rng = np.random.default_rng(0)
    N, D, K = 2000, 32, 10
    Y = rng.standard_normal((N, D)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    order = np.ascontiguousarray(np.argsort(-Y, axis=0).T.astype(np.int32))
    vals = np.ascontiguousarray(np.take_along_axis(Y, np.argsort(-Y, axis=0), axis=0).T)
    q = rng.standard_normal(D).astype(np.float32)
    _, _, stats = _native.fagin_search(Y, order, vals, q, K, 64, 0.0, 0, 1, 1)
    for key in ("ns_sorted", "ns_random", "ns_heap", "ns_threshold", "ns_total"):
        assert key in stats, f"missing {key}"
        assert stats[key] >= 0
    # phases must not exceed the total (allow slack for unattributed glue code)
    assert stats["ns_sorted"] + stats["ns_random"] + stats["ns_heap"] \
        + stats["ns_threshold"] <= stats["ns_total"] + 1
```

(Confirm the `order`/`vals` construction matches how `test_fagin_native.py`'s
existing `_search` helper builds them — reuse that helper if present rather than
duplicating.)

- [ ] **Step 7: Run the test**

Run: `python -m pytest tests/test_fagin_native.py::test_phase_timers_present_and_consistent -v`
Expected: PASS

- [ ] **Step 8: Run the full fagin suite to confirm no regression**

Run: `python -m pytest tests/test_fagin_native.py -v`
Expected: all PASS (parity unchanged — we only added timers).

- [ ] **Step 9: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h \
        docuverse/engines/retrieval/simdq/_native/bindings/module.c \
        tests/test_fagin_native.py
git commit -m "Add per-phase wall-clock timers to Fagin kernel stats"
```

### Task 1.2: Measurement script + write the cost-model finding into the report

**Files:**
- Create: `scripts/fagin_phase_timing.py`
- Modify: `research/2026-07-14-fagin-schedule-comparison-nq.md` (add a
  "Where the time actually goes" section)

- [ ] **Step 1: Write `scripts/fagin_phase_timing.py`**

Mirror the structure/CLI of `scripts/fagin_schedule_cost_compare.py` (argparse:
`--tag --index --queries --nq --topk --batch`). For each schedule
(TA/TASD/GTA/GTASD), run the first `--nq` queries, accumulate the `ns_*` fields,
and print a per-schedule table of mean ms/query split by phase plus the phase's
share of `ns_total`. Warm up with `--nq` throwaway queries first (fill caches)
and pin `num_threads=1` so the split is interpretable. Cross-check all
schedules still return identical top-K (reuse the assert from
`fagin_schedule_cost_compare.py`).

- [ ] **Step 2: Run it on 97m**

Run:
```bash
python scripts/fagin_phase_timing.py --tag "granite-97m-r2 (384d)" \
  --index experiments/nq_new/simq_data/nq_dev-simdq-simdq-granite97m-512-100-20260702 \
  --queries /home/raduf/sandbox2/docuverse/scratch/nq_hw/q97m.npy \
  --nq 50 --topk 100 --batch 2048
```
Expected: a per-phase ms/query breakdown. Hypothesis to confirm/refute:
`ns_random` is the dominant share for all four schedules.

- [ ] **Step 3: Record the finding in the research report**

Add a section to `research/2026-07-14-fagin-schedule-comparison-nq.md` with the
measured phase split and an explicit statement: does random access dominate?
(Yes ⇒ Phase 3 pruning is the priority. No ⇒ note which phase dominates and
re-prioritize.) State the wall-clock TA-vs-GTASD ratio next to the 8×
sorted-access ratio, closing the "cost ≠ latency" caveat.

- [ ] **Step 4: Commit**

```bash
git add scripts/fagin_phase_timing.py research/2026-07-14-fagin-schedule-comparison-nq.md
git commit -m "Measure Fagin per-phase timing on NQ; record cost model"
```

**Phase 1 gate:** Do not start Phase 2/3 until the measured cost split is
recorded. If random access dominates as hypothesized, proceed to Phase 3 first
(bigger win); Phase 2 becomes secondary. Revisit this plan's ordering with the
user after Phase 1.

---

## Phase 2 — Reduce steepest per-round overhead

**Rationale:** GTASD runs ~1,700-3,900 rounds/query, recomputing the O(ndims·90)
water-filling threshold every round when only one dim changed. Two independent,
exact-preserving reductions.

### Task 2.1: `fagin_batch_rows` sweep (no code change — establishes the tuning baseline)

**Files:**
- Create: `scripts/fagin_batch_sweep.py`

- [ ] **Step 1: Write the sweep script**

Argparse CLI mirroring `fagin_schedule_cost_compare.py`, plus
`--batches 512,1024,2048,4096,8192`. For `steepest_norm` (GTASD), run the first
`--nq` queries at each batch and report mean ms/query (needs Task 1.1 timers) and
mean sorted_accesses. Cross-check exact top-K parity at every batch.

- [ ] **Step 2: Run and record the wall-clock-vs-batch curve**

Run:
```bash
python scripts/fagin_batch_sweep.py --tag "granite-97m-r2 (384d)" \
  --index experiments/nq_new/simq_data/nq_dev-simdq-simdq-granite97m-512-100-20260702 \
  --queries /home/raduf/sandbox2/docuverse/scratch/nq_hw/q97m.npy \
  --nq 50 --topk 100 --batches 512,1024,2048,4096,8192
```
Expected: ms/query has a minimum; sorted_accesses grows monotonically with
batch (coarser halting). Identify the wall-clock sweet spot.

- [ ] **Step 3: Commit**

```bash
git add scripts/fagin_batch_sweep.py
git commit -m "Add Fagin batch-size sweep script for steepest schedules"
```

### Task 2.2: Warm-start the water-filling ternary search

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h`
  (`fagin_threshold_norm` at ~208; steepest branch caller at ~426)
- Test: `tests/test_fagin_native.py`

- [ ] **Step 1: Write a failing parity test for warm-started threshold**

Add a test that runs `steepest_norm` on a fixed seed and asserts the returned
top-K equals brute force (the warm-start must not change results — it only
changes how `lam` is initialized). This locks exactness before the change.

```python
def test_gtasd_exact_after_warmstart():
    rng = np.random.default_rng(7)
    N, D, K = 3000, 48, 20
    Y = rng.standard_normal((N, D)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    order, vals = _sorted_lists(Y)   # reuse the suite's helper
    q = rng.standard_normal(D).astype(np.float32)
    _, idx_b, _ = _native.fagin_search(Y, order, vals, q, K, 256, 0.0, 0, 1, 3)
    got = set(np.frombuffer(idx_b, dtype=np.int64).tolist())
    gold = set(np.argsort(-(Y @ q))[:K].tolist())
    assert got == gold
```

- [ ] **Step 2: Run it to confirm it passes on the CURRENT kernel (baseline)**

Run: `python -m pytest tests/test_fagin_native.py::test_gtasd_exact_after_warmstart -v`
Expected: PASS (this is the invariant we must keep — capture it before editing).

- [ ] **Step 3: Widen the ternary-search bracket around the previous `lam`**

In the steepest branch, the previous round's optimal `lam` is already captured
(`&lam`, line ~426). Pass it into `fagin_threshold_norm` as a hint and start the
`[lo, hi]` bracket as a narrow window around it (clamped to the valid range),
falling back to the full `[1e-12, qnorm/R]` bracket on the first round
(`lam == 0`). Because `g(lam)` is convex, a correct minimum is still found; a
too-narrow bracket only risks a *looser* bound (still a valid upper bound ⇒ still
exact), never a wrong result. Add a signature param `double lam_hint` (0 = no
hint) rather than overloading `lam_out`.

- [ ] **Step 4: Rebuild and re-run the parity test**

Run: `python setup.py build_ext --inplace && python -m pytest tests/test_fagin_native.py::test_gtasd_exact_after_warmstart -v`
Expected: PASS (results unchanged).

- [ ] **Step 5: Run full suite + measure the speedup**

Run: `python -m pytest tests/test_fagin_native.py -v` (all PASS), then re-run
`scripts/fagin_phase_timing.py` and confirm `ns_threshold` dropped for GTASD
with identical results.

- [ ] **Step 6: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h tests/test_fagin_native.py
git commit -m "Warm-start Fagin water-filling ternary search from previous lambda"
```

### Task 2.3: Top-M steepest hybrid (advance M steepest dims per round) — SKIPPED (not needed)

> **STATUS: SKIPPED.** Tasks 2.1 (batch sweep) + 2.2 (warm-start) already made
> GTASD the fastest exact schedule. The threshold cost is exactly linear in
> round count, and raising `fagin_batch_rows` (an existing zero-code config
> knob) collapses rounds 1715→217 with *flat* sorted accesses — so the top-M
> hybrid attacks a variable already solved, and its only theoretical edge
> (preserving column-selection sharpness) buys nothing when sorted accesses are
> already flat across batch sizes. Building a new kernel param path here is
> speculative complexity the data does not justify. Revisit only if a future
> workload shows sorted accesses growing materially with batch.

**Original task (retained for context):**

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h`
  (steepest branch, and a new `top_m` param threaded through
  `fagin_ta_search` → `module.c` → `fagin_index.py`)
- Modify: `docuverse/engines/retrieval/simdq/_native/bindings/module.c`
- Modify: `docuverse/engines/retrieval/fagin/fagin_index.py`
- Test: `tests/test_fagin_native.py`

- [ ] **Step 1: Write a failing test: M>1 still returns exact top-K**

```python
@pytest.mark.parametrize("top_m", [1, 2, 4, 8])
def test_steepest_topm_exact(top_m):
    rng = np.random.default_rng(top_m)
    N, D, K = 4000, 64, 25
    Y = rng.standard_normal((N, D)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    order, vals = _sorted_lists(Y)
    q = rng.standard_normal(D).astype(np.float32)
    # new trailing arg: top_m (default 1 == current behavior)
    _, idx_b, _ = _native.fagin_search(Y, order, vals, q, K, 512, 0.0, 0, 1, 3, 1.0, top_m)
    got = set(np.frombuffer(idx_b, dtype=np.int64).tolist())
    gold = set(np.argsort(-(Y @ q))[:K].tolist())
    assert got == gold
```

- [ ] **Step 2: Run it to confirm it FAILS (arg not yet accepted)**

Run: `python -m pytest tests/test_fagin_native.py::test_steepest_topm_exact -v`
Expected: FAIL (`fagin_search` rejects the extra arg / arg ignored).

- [ ] **Step 3: Thread a `top_m` parameter end-to-end**

Add `int64_t top_m` (default 1) to `fagin_ta_search`. In the steepest branch,
pop up to `top_m` dims from the heap per round, advance each by `batch` and score
its candidates (accumulate into the same `cand`/`cand_scores` buffers — they are
already sized `ndims*batch`), then do ONE threshold recompute + ONE halting test
for the whole round, then push the advanced dims back. Parse the optional arg in
`module.c` (`PyArg_ParseTuple` format string gains a trailing `|...L`, default 1)
and pass it from `fagin_index.py` `search(..., top_m=1)`.

- [ ] **Step 4: Rebuild and run the parametrized test**

Run: `python setup.py build_ext --inplace && python -m pytest tests/test_fagin_native.py::test_steepest_topm_exact -v`
Expected: all 4 params PASS.

- [ ] **Step 5: Full suite + measure**

Run full suite (PASS), then extend `fagin_batch_sweep.py` (or a new
`--top-m` sweep) to compare M ∈ {1,2,4,8} wall-clock and sorted_accesses.
Record the M vs latency/accesses trade-off (M=1 sharpest column choice, larger M
fewer rounds).

- [ ] **Step 6: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h \
        docuverse/engines/retrieval/simdq/_native/bindings/module.c \
        docuverse/engines/retrieval/fagin/fagin_index.py tests/test_fagin_native.py
git commit -m "Add top-M steepest hybrid schedule for Fagin (M dims per round)"
```

---

## Phase 3 — Score-gated candidate pruning — CANCELLED (proven void)

> **STATUS: CANCELLED.** After Phase 1 measurement, exact frontier pruning
> (Task 3.1) and block-max WAND (Task 3.2) were both proven void on dense
> L2-normalized embeddings: every surfaced doc has upper bound `UB ≥ T` (the
> halt threshold), so nothing is prunable before global halt (Fagin
> instance-optimality); and dense vectors lack the sparsity WAND needs.
> Incremental accumulation conserves total mul-adds (`random_accesses == N`
> regardless) and regresses due to scatter vs contiguous SIMD. Full proof +
> measurement: `research/2026-07-14-fagin-exact-pruning-negative-result.md`.
> The remaining exact lever is cheaper per-doc scoring (simdq quantized codes,
> already done) or approximate ε>0 pruning. **Do not implement the tasks below.**

**Original rationale (retained for context):** `random_accesses == N` means Fagin fully scores every doc. The
k-th-best heap threshold currently gates only halting. If we instead use it to
*skip* full dot products for candidates whose upper bound is already below the
k-th best, and to *skip whole blocks* of sorted access that cannot lift any
survivor above threshold, Fagin becomes genuinely sublinear in scored docs. This
is the TA→WAND/MaxScore/block-max transition from IR. Exactness holds because we
only skip when a valid UPPER BOUND is provably below a value already in the top-K.

**Caveat to validate empirically:** dense embeddings have all D dims active with
similar magnitudes, so per-doc upper bounds stay high and pruning may be weak.
The norm-aware bound (already implemented) tightens the per-doc bound and should
compound. Phase 3 is a *hypothesis test*, not a guaranteed win — measure pruning
rate before committing to the complexity.

### Task 3.1: Partial-score upper-bound pruning of candidates

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h`
- Test: `tests/test_fagin_native.py`

- [ ] **Step 1: Write a failing test — pruning returns exact top-K AND scores fewer than N docs**

```python
def test_pruning_exact_and_sublinear():
    rng = np.random.default_rng(3)
    N, D, K = 20000, 64, 50
    Y = rng.standard_normal((N, D)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    order, vals = _sorted_lists(Y)
    q = rng.standard_normal(D).astype(np.float32)
    _, idx_b, stats = _native.fagin_search(Y, order, vals, q, K, 512, 0.0, 0, 1, 3)
    got = set(np.frombuffer(idx_b, dtype=np.int64).tolist())
    gold = set(np.argsort(-(Y @ q))[:K].tolist())
    assert got == gold                       # still exact
    assert stats["random_accesses"] < N      # scored fewer than all docs
```

- [ ] **Step 2: Run it — expect the parity half to PASS but `random_accesses < N` to FAIL**

Run: `python -m pytest tests/test_fagin_native.py::test_pruning_exact_and_sublinear -v`
Expected: FAIL on `random_accesses < N` (current kernel scores everything).

- [ ] **Step 3: Track per-candidate partial scores + accumulated seen-mass, defer/skip full scoring**

Design (implement in the steepest branch first, then lockstep): maintain per
active dim the running frontier contribution. For a surfaced-but-not-fully-scored
candidate, its upper bound = (sum of q_j·y_ij over dims where it has already been
passed in sorted order) + (sum of q_j·t_j over dims not yet passed). When
`heap.size == K` and a candidate's upper bound < k-th best, skip its full dot
product entirely. This requires tracking which dims have "passed" each candidate;
start with the simpler variant: once a candidate is seen in *any* lane, compute
its upper bound from the current global frontier `T_remaining` and only do the
full `fagin_dot` if `partial_seen_contribution + T_remaining >= kth`. Increment a
new `stats->pruned` counter for skips.

- [ ] **Step 4: Rebuild, run the test**

Run: `python setup.py build_ext --inplace && python -m pytest tests/test_fagin_native.py::test_pruning_exact_and_sublinear -v`
Expected: PASS (exact, and `random_accesses < N`).

- [ ] **Step 5: Full suite + measure pruning rate on real NQ**

Run full suite (PASS). Extend `fagin_phase_timing.py` to report
`random_accesses / N` (pruning rate) per schedule on the 50 NQ queries.
Record whether dense-vector pruning is meaningful (e.g. <50% scored) or weak.

- [ ] **Step 6: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h tests/test_fagin_native.py
git commit -m "Add upper-bound candidate pruning to Fagin (skip full dot products)"
```

### Task 3.2 (conditional): Block-max sorted-access skipping

**Gate:** Only implement if Task 3.1 shows meaningful pruning (worth the added
complexity). If dense pruning is weak (<~20% skipped), STOP and record the
negative result — do not build block-max on a foundation that does not pay off.

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h`
- Possibly: precompute per-block max contributions at
  `FaginIndex.build` time (`fagin_index.py`), persisted alongside `order`/`vals`.
- Test: `tests/test_fagin_native.py`

- [ ] **Step 1: Write a failing test — block-skipping exact + fewer sorted accesses**

Analogous to 3.1: assert exact top-K and `sorted_accesses` strictly lower than
the non-block-max GTASD baseline on a fixed seed.

- [ ] **Step 2: Precompute block-max metadata (WAND-style)**

At build time, for each lane and each block of `block_size` rows in the sorted
list, store the max `|q-independent value|` so at query time `q_j·blockmax_j`
bounds the block's contribution. Persist as an optional array; fall back to
recomputing if absent (backward compatible with existing index dirs).

- [ ] **Step 3: Skip blocks whose contribution cannot lift any survivor**

In the schedule loop, before advancing a lane by a block, test whether that
block's max possible contribution could raise any current candidate's upper
bound above the k-th best; if not, skip the block (advance the cursor without
scoring). Increment `stats->blocks_skipped`.

- [ ] **Step 4: Rebuild, run test, full suite**

Run: `python setup.py build_ext --inplace && python -m pytest tests/test_fagin_native.py -v`
Expected: all PASS, new block-max test included.

- [ ] **Step 5: Measure end-to-end on NQ + write the Phase 3 research report**

Create `research/2026-07-XX-fagin-block-max-pruning.md` (dated) with the
pruning-rate and wall-clock results across TA/TASD/GTA/GTASD × pruning on/off,
first 50 NQ queries, both encoders. Include the honest negative result if dense
pruning underperforms.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Add block-max sorted-access skipping to Fagin; document results"
```

---

## Self-review notes

- **Spec coverage:** Q1 (measure time split) → Phase 1. Q2 (reduce steepest
  overhead / "batch computing") → Phase 2 (batch sweep, warm-start, top-M
  hybrid — the honest answer to "why not batch GTASD"). Q3 (skip work in other
  lanes using the current top-K) → Phase 3 (candidate pruning + block-max). All
  three user questions map to tasks.
- **Exactness invariant** is asserted by a test in every task that touches the
  algorithm; never weakened.
- **Ordering is provisional:** Phase 1 is a hard gate. If it confirms random
  access dominates, Phase 3 likely outranks Phase 2 for impact — re-confirm
  priority with the user after Phase 1.
- **Backward compatibility:** new kernel args (`top_m`, block-max metadata) are
  optional with current-behavior defaults, so existing callers and index dirs
  keep working.
```
