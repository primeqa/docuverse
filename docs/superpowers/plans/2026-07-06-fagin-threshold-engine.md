# FaginThresholdEngine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new in-process retrieval engine (`db_engine: fagin`) that answers exact top-k inner-product queries with Fagin's Threshold Algorithm over per-dimension sorted lists, with a native C/OpenMP kernel compiled into the existing `_simdq_native` extension.

**Architecture:** Per approved spec `docs/superpowers/specs/2026-07-06-fagin-threshold-engine-design.md`. Index = fp32 matrix `Y[N,D]` + per-dimension descending argsort `order[D,N]` (int32) + sorted values `vals[D,N]` (fp32), built with NumPy at ingest, mmap-loaded at search. Native kernel does lock-step rounds of `B` rows of sorted access per active dimension (head for `q_j>0`, tail for `q_j<0` — the double scan), full-dot-product random access on first sighting (dedupe via bitmap), threshold `T = Σ q_j·t_j`, halting at `kth ≥ T − ε` (ε=0 default = exact). Engine class mirrors `SimdqEngine` verbatim.

**Tech Stack:** CPython C-API binding in `_simdq_native` (reuses `simdq_topk.h` via an order-preserving float→int64 key transform), OpenMP, NumPy, pytest. Conda env `ndocu` for all commands.

**Three deliberate deviations from the spec (small, convention-driven):**
1. `fagin_num_threads` default is `0` = "OpenMP default" (matching `simdq_num_threads`), not `-1`.
2. The dot product is compiler-vectorized (`#pragma omp simd` + `-O3 -march=native`, which emits AVX2/AVX-512 on x86) rather than hand-written intrinsics — one generic-D implementation, per the spec's "no fixed-D whitelist" requirement.
3. Per-query access stats are aggregated on the engine (`ta_stats`, thread-safe) and exposed via `info()`, rather than woven into the evaluation report output. Benchmark scripts can read `engine.info()["ta_stats"]` after a run; wiring the counters into the report formatter is deferred until a comparison table actually needs it.

**Environment notes for the implementer:**
- Run everything through `conda run -n ndocu ...` (per CLAUDE.md).
- Rebuild the native extension after any C change:
  `conda run -n ndocu python setup.py build_ext --inplace`
  (rebuilds `docuverse/engines/retrieval/simdq/_simdq_native.cpython-*.so`; no setup.py changes needed — the extension already compiles `bindings/module.c` with `include/` on the include path).
- `docuverse/utils/retrievers.py` uses **3-space indentation** inside `create_retrieval_engine`. Match it exactly.
- Commit messages are descriptive prose (no Conventional Commits prefix), per CLAUDE.md.

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Create | `docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h` | TA kernel: key transforms, dot product, `fagin_ta_search` |
| Modify | `docuverse/engines/retrieval/simdq/_native/bindings/module.c` | `py_fagin_search` binding + method-table entry + include |
| Create | `tests/test_fagin_native.py` | kernel-level correctness/knob/stats tests |
| Create | `docuverse/engines/retrieval/fagin/__init__.py` | package export |
| Create | `docuverse/engines/retrieval/fagin/fagin_index.py` | `FaginIndex` build/save/load/search |
| Create | `tests/test_fagin_index.py` | index build/persist/search tests |
| Modify | `docuverse/engines/search_engine_config_params.py` (after line 597, `simdq_ivf_nprobe`) | `fagin_*` config fields |
| Create | `docuverse/engines/retrieval/fagin/fagin_engine.py` | `FaginThresholdEngine` |
| Modify | `docuverse/utils/retrievers.py:79-96` | dispatch branch + supported-engines error string |
| Modify | `tests/test_engine_dispatch.py:27-57` | add fagin names to `KNOWN_NAMES` |
| Create | `tests/test_fagin_engine.py` | end-to-end engine tests (tiny encoder) |

---

### Task 1: TA kernel + binding + core exactness test

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/bindings/module.c`
- Test: `tests/test_fagin_native.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fagin_native.py`:

```python
"""Kernel-level tests for _simdq_native.fagin_search (Fagin's Threshold
Algorithm, exact top-K inner product over per-dimension sorted lists)."""
import numpy as np
import pytest

_native = pytest.importorskip("docuverse.engines.retrieval.simdq._simdq_native")


def _search(Y, q, K, batch=8, epsilon=0.0, max_depth=0, num_threads=1):
    """Build sorted lists in NumPy (reference layout) and call the kernel."""
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    perm = np.argsort(-Y, axis=0, kind="stable")           # (N, D), descending
    order = np.ascontiguousarray(perm.T, dtype=np.int32)   # (D, N)
    vals = np.ascontiguousarray(
        np.take_along_axis(Y, perm, axis=0).T, dtype=np.float32)  # (D, N)
    scores_b, idx_b, stats = _native.fagin_search(
        Y, order, vals, q, int(K), int(batch), float(epsilon),
        int(max_depth), int(num_threads))
    scores = np.frombuffer(scores_b, dtype=np.float32).copy()
    idxs = np.frombuffer(idx_b, dtype=np.int64).copy()
    return idxs, scores, stats


def _check_exact_topk(Y, q, idxs, scores, K):
    """Exactness contract, robust to ties and fp32 accumulation order:
    every returned doc scores >= the true kth score, returned score values
    are the true scores of the returned ids, and the returned score multiset
    matches the brute-force top-K score multiset."""
    s = Y.astype(np.float64) @ q.astype(np.float64)
    n_expect = min(K, len(s))
    desc = np.sort(s)[::-1]
    kth = desc[n_expect - 1]
    valid = idxs[:n_expect]
    assert np.all(valid >= 0)
    assert np.all(idxs[n_expect:] == -1)
    assert len(np.unique(valid)) == n_expect
    assert np.all(s[valid] >= kth - 1e-4)
    np.testing.assert_allclose(scores[:n_expect], s[valid], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.sort(scores[:n_expect])[::-1], desc[:n_expect],
                               rtol=1e-4, atol=1e-5)
    assert np.all(np.diff(scores[:n_expect]) <= 1e-6)  # descending


def test_exact_topk_matches_brute_force():
    rng = np.random.default_rng(42)
    Y = rng.standard_normal((500, 32))
    q = rng.standard_normal(32)
    idxs, scores, stats = _search(Y, q, K=10)
    _check_exact_topk(Y, q, idxs, scores, 10)
    assert stats["depth"] >= 1
    assert stats["random_accesses"] <= 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ndocu python -m pytest tests/test_fagin_native.py -v`
Expected: FAIL (or ERROR) with `AttributeError: module ... has no attribute 'fagin_search'`

- [ ] **Step 3: Write the kernel header**

Create `docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h`:

```c
// fagin_ta.h — Fagin's Threshold Algorithm (TA) over per-dimension sorted
// lists (Fagin, Lotem, Naor, PODS'01). Exact top-K inner-product retrieval:
// lock-step sorted access over the active per-dimension lists, one full
// dot-product random access per first-seen document, halting when the kth
// best score reaches the threshold T = sum_j q_j * t_j.
//
// Negative query weights: lists are sorted descending by raw value; for
// q_j < 0 the scan walks the same list bottom-up (ascending raw value ==
// descending contribution q_j * x) — the "double scan". Dimensions with
// q_j == 0 are skipped entirely (they contribute 0 to every score and to
// the threshold, so skipping stays exact).
//
// The top-K heap is simdq_topk.h (int64 keys, keeps K smallest). Float
// scores are mapped through an order-preserving float32 -> uint32 -> int64
// transform, negated so "K smallest keys" == "K largest scores".

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "simdq_topk.h"

typedef struct {
    int64_t depth;            // rows of sorted access per active dim
    int64_t sorted_accesses;  // total sorted accesses (active dims x depth)
    int64_t random_accesses;  // candidates scored (full dot products)
    int64_t rounds;           // TA rounds executed
    int     exhausted;        // 1 if every row was scanned (depth == N)
} fagin_stats_t;

// Order-preserving float32 -> int64 key, negated for simdq_topk (which
// keeps the K *smallest* keys; we want the K *largest* scores). The
// standard monotone IEEE-754 transform: flip all bits of negatives, flip
// only the sign bit of non-negatives.
static inline int64_t fagin_score_key(float s) {
    uint32_t u;
    memcpy(&u, &s, sizeof u);
    u ^= (u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return -(int64_t)u;
}

static inline float fagin_key_score(int64_t k) {
    uint32_t u = (uint32_t)(-k);
    u ^= (u & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    float s;
    memcpy(&s, &u, sizeof u);
    return s;
}

// Generic-D dot product; -O3 -march=native vectorizes this to AVX2/AVX-512
// on x86 (NEON on arm64). No fixed-D whitelist.
static inline float fagin_dot(const float *restrict a, const float *restrict b,
                              int64_t D) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int64_t j = 0; j < D; j++) s += a[j] * b[j];
    return s;
}

// Runs TA for one query. Writes min(K, N) results (best first) into
// out_scores/out_idxs; remaining slots get score 0 / idx -1. Returns the
// number of valid results. `batch` = rows of sorted access per active dim
// per round; `epsilon` = additive halting slack (0 = exact); `max_depth`
// caps sorted-access depth (0 = unlimited).
static inline int fagin_ta_search(
        const float *Y,          // [N, D] row-major fp32
        const int32_t *order,    // [D, N] per-dim argsort of Y[:,j], descending
        const float *vals,       // [D, N] Y[order[j], j] (sorted values)
        int64_t N, int64_t D,
        const float *q,          // [D]
        int K, int64_t batch, float epsilon, int64_t max_depth,
        float *out_scores, int64_t *out_idxs, fagin_stats_t *stats) {

    memset(stats, 0, sizeof *stats);
    for (int i = 0; i < K; i++) { out_scores[i] = 0.0f; out_idxs[i] = -1; }

    int heap_cap = (int)((int64_t)K < N ? (int64_t)K : N);

    // Active dimensions (q_j != 0).
    int32_t *dims = (int32_t *)malloc((size_t)D * sizeof *dims);
    int64_t ndims = 0;
    for (int64_t j = 0; j < D; j++)
        if (q[j] != 0.0f) dims[ndims++] = (int32_t)j;

    if (ndims == 0) {
        // Every document scores 0; any min(K, N) docs are a correct top-K.
        for (int i = 0; i < heap_cap; i++) out_idxs[i] = i;
        stats->exhausted = 1;
        free(dims);
        return heap_cap;
    }

    uint64_t *seen = (uint64_t *)calloc(((size_t)N + 63) / 64, sizeof *seen);
    int64_t *cand = (int64_t *)malloc((size_t)ndims * (size_t)batch * sizeof *cand);
    float *cand_scores = (float *)malloc((size_t)ndims * (size_t)batch
                                         * sizeof *cand_scores);
    int64_t *hkeys = (int64_t *)malloc((size_t)heap_cap * sizeof *hkeys);
    int64_t *hidxs = (int64_t *)malloc((size_t)heap_cap * sizeof *hidxs);

    simdq_topk_t heap;
    simdq_topk_init(&heap, heap_cap, hkeys, hidxs);

    int64_t depth = 0;
    while (depth < N) {
        int64_t take = batch;
        if (take > N - depth) take = N - depth;
        if (max_depth > 0 && take > max_depth - depth) take = max_depth - depth;

        // Phase A (serial, cheap): sorted access — gather unseen candidates.
        int64_t n_cand = 0;
        for (int64_t a = 0; a < ndims; a++) {
            int64_t j = dims[a];
            const int32_t *lst = order + (size_t)j * (size_t)N;
            for (int64_t r = 0; r < take; r++) {
                int64_t pos = (q[j] > 0.0f) ? depth + r : N - 1 - depth - r;
                int64_t id = lst[pos];
                uint64_t bit = 1ull << (id & 63);
                if (!(seen[id >> 6] & bit)) {
                    seen[id >> 6] |= bit;
                    cand[n_cand++] = id;
                }
            }
        }
        stats->sorted_accesses += ndims * take;

        // Phase B (parallel): random access — one full dot per new candidate.
        #pragma omp parallel for schedule(static)
        for (int64_t c = 0; c < n_cand; c++)
            cand_scores[c] = fagin_dot(q, Y + (size_t)cand[c] * (size_t)D, D);
        stats->random_accesses += n_cand;

        // Phase C (serial): heap maintenance (K is small).
        for (int64_t c = 0; c < n_cand; c++) {
            int64_t key = fagin_score_key(cand_scores[c]);
            if (key < simdq_topk_threshold(&heap))
                simdq_topk_offer(&heap, key, cand[c]);
        }

        depth += take;
        stats->rounds += 1;

        // Threshold T = sum_j q_j * t_j at the current cursors.
        float T = 0.0f;
        for (int64_t a = 0; a < ndims; a++) {
            int64_t j = dims[a];
            int64_t pos = (q[j] > 0.0f) ? depth - 1 : N - depth;
            T += q[j] * vals[(size_t)j * (size_t)N + pos];
        }

        if (heap.size >= heap_cap) {
            float kth = fagin_key_score(simdq_topk_threshold(&heap));
            if (kth >= T - epsilon) break;   // slide-19 halting rule (+ eps slack)
        }
        if (max_depth > 0 && depth >= max_depth) break;
    }
    if (depth >= N) stats->exhausted = 1;
    stats->depth = depth;

    int64_t *okeys = (int64_t *)malloc((size_t)heap_cap * sizeof *okeys);
    int64_t *oidxs = (int64_t *)malloc((size_t)heap_cap * sizeof *oidxs);
    int n = simdq_topk_extract_sorted(&heap, okeys, oidxs);
    // Ascending keys == descending scores, so results come out best-first.
    for (int i = 0; i < n; i++) {
        out_scores[i] = fagin_key_score(okeys[i]);
        out_idxs[i] = oidxs[i];
    }
    free(okeys); free(oidxs);
    free(hkeys); free(hidxs);
    free(cand); free(cand_scores);
    free(seen); free(dims);
    return n;
}
```

- [ ] **Step 4: Add the binding to module.c**

In `docuverse/engines/retrieval/simdq/_native/bindings/module.c`:

(a) After the existing kernel includes (the block ending with
`#include "simdq_kernels_asym_b4.h"`), add:

```c
#include "fagin_ta.h"
```

(b) Before the `// ---------- module table ----------` comment, add:

```c
// ---------- fagin threshold algorithm ----------

static PyObject *py_fagin_search(PyObject *self, PyObject *args) {
    PyObject *y_obj, *order_obj, *vals_obj, *q_obj;
    Py_ssize_t K, batch, max_depth, num_threads;
    float epsilon;
    if (!PyArg_ParseTuple(args, "OOOOnnfnn",
                          &y_obj, &order_obj, &vals_obj, &q_obj,
                          &K, &batch, &epsilon, &max_depth, &num_threads))
        return NULL;
    if (K < 1)     { PyErr_SetString(PyExc_ValueError, "K must be >= 1");     return NULL; }
    if (batch < 1) { PyErr_SetString(PyExc_ValueError, "batch must be >= 1"); return NULL; }

    Py_buffer y_view, order_view, vals_view, q_view;
    if (get_buffer(y_obj, &y_view, 'f', 0) != 0) return NULL;
    if (get_buffer(order_obj, &order_view, 'i', 0) != 0) {
        PyBuffer_Release(&y_view); return NULL;
    }
    if (get_buffer(vals_obj, &vals_view, 'f', 0) != 0) {
        PyBuffer_Release(&y_view); PyBuffer_Release(&order_view); return NULL;
    }
    if (get_buffer(q_obj, &q_view, 'f', 0) != 0) {
        PyBuffer_Release(&y_view); PyBuffer_Release(&order_view);
        PyBuffer_Release(&vals_view); return NULL;
    }

    PyObject *scores_bytes = NULL, *idx_bytes = NULL;

    if (y_view.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "Y must be 2-D (N, D)");
        goto failta;
    }
    {
        Py_ssize_t N = y_view.shape[0], D = y_view.shape[1];
        if (order_view.ndim != 2 || order_view.shape[0] != D
                                 || order_view.shape[1] != N) {
            PyErr_Format(PyExc_ValueError, "order must have shape (%zd, %zd)", D, N);
            goto failta;
        }
        if (vals_view.ndim != 2 || vals_view.shape[0] != D
                                || vals_view.shape[1] != N) {
            PyErr_Format(PyExc_ValueError, "vals must have shape (%zd, %zd)", D, N);
            goto failta;
        }
        if (q_view.ndim != 1 || q_view.shape[0] != D) {
            PyErr_Format(PyExc_ValueError, "q must have shape (%zd,)", D);
            goto failta;
        }

        char *scores_data, *idx_data;
        scores_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(float), &scores_data);
        if (!scores_bytes) goto failta;
        idx_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(int64_t), &idx_data);
        if (!idx_bytes) goto failta;

        fagin_stats_t st;
        int saved_threads = omp_get_max_threads();
        if (num_threads > 0) omp_set_num_threads((int)num_threads);

        Py_BEGIN_ALLOW_THREADS
        fagin_ta_search((const float *)y_view.buf,
                        (const int32_t *)order_view.buf,
                        (const float *)vals_view.buf,
                        (int64_t)N, (int64_t)D,
                        (const float *)q_view.buf,
                        (int)K, (int64_t)batch, epsilon, (int64_t)max_depth,
                        (float *)scores_data, (int64_t *)idx_data, &st);
        Py_END_ALLOW_THREADS

        if (num_threads > 0) omp_set_num_threads(saved_threads);

        PyBuffer_Release(&y_view); PyBuffer_Release(&order_view);
        PyBuffer_Release(&vals_view); PyBuffer_Release(&q_view);

        PyObject *stats = Py_BuildValue(
            "{s:L,s:L,s:L,s:L,s:i}",
            "depth", (long long)st.depth,
            "sorted_accesses", (long long)st.sorted_accesses,
            "random_accesses", (long long)st.random_accesses,
            "rounds", (long long)st.rounds,
            "exhausted", st.exhausted);
        if (!stats) { Py_DECREF(scores_bytes); Py_DECREF(idx_bytes); return NULL; }
        return Py_BuildValue("(NNN)", scores_bytes, idx_bytes, stats);
    }

failta:
    Py_XDECREF(scores_bytes);
    Py_XDECREF(idx_bytes);
    PyBuffer_Release(&y_view); PyBuffer_Release(&order_view);
    PyBuffer_Release(&vals_view); PyBuffer_Release(&q_view);
    return NULL;
}
```

(c) In `SimdqMethods[]`, before the `{NULL, NULL, 0, NULL}` sentinel, add:

```c
    {"fagin_search", py_fagin_search, METH_VARARGS,
     "fagin_search(Y, order, vals, q, K, batch, epsilon, max_depth, num_threads)"
     " -> (scores, indices, stats); exact top-K inner product via Fagin's"
     " Threshold Algorithm. Y fp32 (N,D); order int32 (D,N) descending argsort"
     " per dim; vals fp32 (D,N) sorted values; epsilon = additive halting"
     " slack (0 = exact); max_depth 0 = unlimited."},
```

- [ ] **Step 5: Rebuild the extension**

Run: `conda run -n ndocu python setup.py build_ext --inplace`
Expected: compiles `bindings/module.c` without errors and refreshes `docuverse/engines/retrieval/simdq/_simdq_native.cpython-*.so`.

- [ ] **Step 6: Run test to verify it passes**

Run: `conda run -n ndocu python -m pytest tests/test_fagin_native.py -v`
Expected: `test_exact_topk_matches_brute_force PASSED`

- [ ] **Step 7: Verify no simdq regression**

Run: `conda run -n ndocu python -m pytest tests/test_simdq_engine.py -x -q`
Expected: all pass (or skip if `sentence_transformers` unavailable) — proves the module edit didn't break existing bindings.

- [ ] **Step 8: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h \
        docuverse/engines/retrieval/simdq/_native/bindings/module.c \
        tests/test_fagin_native.py
git commit -m "Add Fagin Threshold Algorithm kernel and fagin_search binding to the simdq native extension"
```

---

### Task 2: Kernel edge cases and knobs

**Files:**
- Modify: `tests/test_fagin_native.py` (append)
- Possibly modify: `docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h` (only if a test exposes a bug)

- [ ] **Step 1: Append the edge-case and knob tests**

Append to `tests/test_fagin_native.py`:

```python
def test_negative_and_zero_query_weights():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((400, 24))
    q = rng.standard_normal(24)
    q[::3] = -np.abs(q[::3])       # force negatives (bottom-up scans)
    q[1::5] = 0.0                  # force skipped dims
    idxs, scores, _ = _search(Y, q, K=10)
    _check_exact_topk(Y, q, idxs, scores, 10)


def test_all_negative_query_and_negative_documents():
    rng = np.random.default_rng(1)
    Y = -np.abs(rng.standard_normal((300, 16)))
    q = -np.abs(rng.standard_normal(16))
    idxs, scores, _ = _search(Y, q, K=7)
    _check_exact_topk(Y, q, idxs, scores, 7)


def test_d_not_multiple_of_8_and_odd_batch():
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((50, 7))
    q = rng.standard_normal(7)
    idxs, scores, _ = _search(Y, q, K=5, batch=3)
    _check_exact_topk(Y, q, idxs, scores, 5)


def test_k_equals_n_and_k_greater_than_n():
    rng = np.random.default_rng(3)
    Y = rng.standard_normal((20, 8))
    q = rng.standard_normal(8)
    idxs, scores, _ = _search(Y, q, K=20)
    _check_exact_topk(Y, q, idxs, scores, 20)
    idxs, scores, _ = _search(Y, q, K=32)     # K > N: 20 valid + 12 padding
    _check_exact_topk(Y, q, idxs, scores, 32)


def test_single_document():
    Y = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    q = np.array([0.5, 1.0, -1.0], dtype=np.float32)
    idxs, scores, _ = _search(Y, q, K=1)
    assert idxs[0] == 0
    np.testing.assert_allclose(scores[0], -4.5, rtol=1e-6)


def test_duplicate_rows_ties():
    rng = np.random.default_rng(4)
    row = rng.standard_normal(12)
    Y = np.vstack([row] * 10 + [rng.standard_normal((30, 12))])
    q = rng.standard_normal(12)
    idxs, scores, _ = _search(Y, q, K=15)
    _check_exact_topk(Y, q, idxs, scores, 15)


def test_all_zero_query():
    rng = np.random.default_rng(5)
    Y = rng.standard_normal((40, 6))
    q = np.zeros(6)
    idxs, scores, stats = _search(Y, q, K=5)
    assert np.all(scores[:5] == 0.0)
    assert sorted(idxs[:5].tolist()) == [0, 1, 2, 3, 4]
    assert stats["sorted_accesses"] == 0 and stats["exhausted"] == 1


def test_epsilon_stops_earlier_and_is_bounded():
    rng = np.random.default_rng(6)
    Y = rng.standard_normal((2000, 48))
    q = rng.standard_normal(48)
    _, exact_scores, exact_stats = _search(Y, q, K=10, epsilon=0.0)
    idxs, scores, stats = _search(Y, q, K=10, epsilon=5.0)
    assert stats["depth"] <= exact_stats["depth"]
    s = Y.astype(np.float64) @ q.astype(np.float64)
    brute_kth = np.sort(s)[::-1][9]
    # Guarantee: returned kth is within epsilon of the true kth.
    assert scores[9] >= brute_kth - 5.0 - 1e-4
    # Returned scores are still the true scores of the returned ids.
    np.testing.assert_allclose(scores[:10], s[idxs[:10]], rtol=1e-4, atol=1e-5)


def test_max_depth_caps_depth():
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((1000, 32))
    q = rng.standard_normal(32)
    idxs, scores, stats = _search(Y, q, K=10, batch=8, max_depth=20)
    assert stats["depth"] <= 20
    valid = idxs[idxs >= 0]
    assert len(np.unique(valid)) == len(valid)
    assert np.all(np.diff(scores[:len(valid)]) <= 1e-6)


def test_stats_consistency():
    rng = np.random.default_rng(8)
    Y = rng.standard_normal((600, 20))
    q = rng.standard_normal(20)
    q[3] = 0.0
    _, _, stats = _search(Y, q, K=10, batch=16)
    n_active = int(np.count_nonzero(q))
    assert stats["sorted_accesses"] == n_active * stats["depth"]
    assert stats["random_accesses"] <= min(600, stats["sorted_accesses"])
    assert stats["rounds"] >= 1
    assert stats["depth"] <= 600


def test_num_threads_invariance():
    rng = np.random.default_rng(9)
    Y = rng.standard_normal((800, 40))
    q = rng.standard_normal(40)
    i1, s1, _ = _search(Y, q, K=10, num_threads=1)
    i4, s4, _ = _search(Y, q, K=10, num_threads=4)
    np.testing.assert_allclose(s1, s4, rtol=1e-6)
    _check_exact_topk(Y, q, i4, s4, 10)


def test_input_validation():
    rng = np.random.default_rng(10)
    Y = np.ascontiguousarray(rng.standard_normal((10, 4)), dtype=np.float32)
    perm = np.argsort(-Y, axis=0, kind="stable")
    order = np.ascontiguousarray(perm.T, dtype=np.int32)
    vals = np.ascontiguousarray(np.take_along_axis(Y, perm, axis=0).T,
                                dtype=np.float32)
    q = np.ascontiguousarray(rng.standard_normal(4), dtype=np.float32)
    with pytest.raises(ValueError):
        _native.fagin_search(Y, order, vals, q, 0, 8, 0.0, 0, 1)   # K < 1
    with pytest.raises(ValueError):
        _native.fagin_search(Y, order, vals, q, 5, 0, 0.0, 0, 1)   # batch < 1
    with pytest.raises(ValueError):
        _native.fagin_search(Y, order, vals, q[:3], 5, 8, 0.0, 0, 1)  # bad q shape
    with pytest.raises(ValueError):
        # .copy() keeps it C-contiguous so the shape check (not the buffer
        # contiguity check) is what rejects it
        _native.fagin_search(Y, order[:, :5].copy(), vals, q, 5, 8, 0.0, 0, 1)
    with pytest.raises(TypeError):
        _native.fagin_search(Y.astype(np.float64), order, vals, q, 5, 8, 0.0, 0, 1)
```

- [ ] **Step 2: Run the tests**

Run: `conda run -n ndocu python -m pytest tests/test_fagin_native.py -v`
Expected: all PASS. If any fails, fix `fagin_ta.h` (rebuild with `conda run -n ndocu python setup.py build_ext --inplace` after each C change) until green — the kernel in Task 1 was written to satisfy these, so failures indicate a real bug, not a missing feature.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fagin_native.py docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h
git commit -m "Cover fagin_search edge cases: negative/zero weights, ties, K>N, epsilon and max_depth knobs, stats consistency"
```

---

### Task 3: FaginIndex (build / save / load / search)

**Files:**
- Create: `docuverse/engines/retrieval/fagin/__init__.py` (placeholder — final export added in Task 5)
- Create: `docuverse/engines/retrieval/fagin/fagin_index.py`
- Test: `tests/test_fagin_index.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fagin_index.py`:

```python
"""FaginIndex build/persist/search tests."""
import numpy as np
import pytest

pytest.importorskip("docuverse.engines.retrieval.simdq._simdq_native")

from docuverse.engines.retrieval.fagin.fagin_index import FaginIndex


def _rand(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d)).astype(np.float32),
            rng.standard_normal(d).astype(np.float32))


def test_build_shapes_and_dtypes():
    Y, _ = _rand()
    ix = FaginIndex.build(Y, encoder_id="test-enc")
    assert ix.n_vectors == 200 and ix.dim == 16
    assert ix.Y.shape == (200, 16) and ix.Y.dtype == np.float32
    assert ix.order.shape == (16, 200) and ix.order.dtype == np.int32
    assert ix.vals.shape == (16, 200) and ix.vals.dtype == np.float32
    # each dimension's vals must be descending
    assert np.all(np.diff(ix.vals, axis=1) <= 0)
    # order must be a permutation per dimension
    for j in range(16):
        assert sorted(ix.order[j].tolist()) == list(range(200))


def test_search_matches_brute_force():
    Y, q = _rand()
    ix = FaginIndex.build(Y)
    idxs, scores, stats = ix.search(q, K=10)
    s = Y.astype(np.float64) @ q.astype(np.float64)
    desc = np.sort(s)[::-1]
    np.testing.assert_allclose(np.sort(scores)[::-1], desc[:10], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(scores, s[idxs], rtol=1e-4, atol=1e-5)
    assert stats["depth"] >= 1


def test_save_load_roundtrip(tmp_path):
    Y, q = _rand()
    ix = FaginIndex.build(Y, encoder_id="enc-x")
    ix.save(tmp_path / "idx")
    lx = FaginIndex.load(tmp_path / "idx")
    assert lx.n_vectors == ix.n_vectors and lx.dim == ix.dim
    assert lx.encoder_id == "enc-x"
    i1, s1, _ = ix.search(q, K=8)
    i2, s2, _ = lx.search(q, K=8)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(s1, s2, rtol=1e-6)


def test_save_overwrites_existing(tmp_path):
    Y, q = _rand()
    ix = FaginIndex.build(Y)
    ix.save(tmp_path / "idx")
    ix.save(tmp_path / "idx")            # second save must not fail
    lx = FaginIndex.load(tmp_path / "idx")
    assert lx.n_vectors == 200


def test_build_rejects_bad_input():
    with pytest.raises(ValueError, match="2-D"):
        FaginIndex.build(np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError, match="N must be >= 1"):
        FaginIndex.build(np.zeros((0, 4), dtype=np.float32))


def test_search_rejects_bad_q():
    Y, _ = _rand()
    ix = FaginIndex.build(Y)
    with pytest.raises(ValueError, match="shape"):
        ix.search(np.zeros(3, dtype=np.float32), K=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n ndocu python -m pytest tests/test_fagin_index.py -v`
Expected: collection ERROR with `ModuleNotFoundError: No module named 'docuverse.engines.retrieval.fagin'`

- [ ] **Step 3: Write the package init and index**

Create `docuverse/engines/retrieval/fagin/__init__.py` (placeholder for now — Task 5 replaces it with the engine export):

```python
```

(empty file)

Create `docuverse/engines/retrieval/fagin/fagin_index.py`:

```python
"""FaginIndex — per-dimension sorted lists + fp32 matrix for Fagin's
Threshold Algorithm (see docs/superpowers/specs/2026-07-06-fagin-threshold-
engine-design.md).

On-disk layout:

    <index_path>/
    ├── meta.json    # {format_version, n_vectors, dim, encoder_id}
    ├── Y.npy        # (N, D) fp32 row-major — random-access side
    ├── order.bin    # (D, N) int32 — per-dim argsort of Y[:, j], descending
    └── vals.bin     # (D, N) fp32  — Y[order[j], j] (sorted values)

All three arrays are mmap-loaded at search time. The native kernel
(_simdq_native.fagin_search) does the per-query TA scan.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from docuverse.engines.retrieval.simdq import _simdq_native as _native

FORMAT_VERSION = 1


@dataclass
class FaginIndex:
    Y: np.ndarray          # (N, D) fp32, C-contiguous
    order: np.ndarray      # (D, N) int32
    vals: np.ndarray       # (D, N) fp32
    n_vectors: int
    dim: int
    encoder_id: Optional[str] = None

    # ----- build -----

    @classmethod
    def build(cls, vectors: np.ndarray,
              encoder_id: Optional[str] = None) -> "FaginIndex":
        if vectors.ndim != 2:
            raise ValueError(f"fagin build: vectors must be 2-D; got shape "
                             f"{vectors.shape}")
        if vectors.shape[0] == 0:
            raise ValueError("fagin build: N must be >= 1; got an empty corpus")
        Y = np.ascontiguousarray(vectors, dtype=np.float32)
        N, D = Y.shape
        # Row indices sorting each column descending; stable for ties.
        perm = np.argsort(-Y, axis=0, kind="stable")               # (N, D)
        vals = np.take_along_axis(Y, perm, axis=0)                 # (N, D)
        order = np.ascontiguousarray(perm.T, dtype=np.int32)       # (D, N)
        vals = np.ascontiguousarray(vals.T, dtype=np.float32)      # (D, N)
        return cls(Y=Y, order=order, vals=vals, n_vectors=int(N), dim=int(D),
                   encoder_id=encoder_id)

    # ----- persist -----

    def save(self, path: os.PathLike) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        with open(tmp / "meta.json", "w") as f:
            json.dump({"format_version": FORMAT_VERSION,
                       "n_vectors": self.n_vectors,
                       "dim": self.dim,
                       "encoder_id": self.encoder_id}, f)
        np.save(tmp / "Y.npy", self.Y)
        np.ascontiguousarray(self.order).tofile(tmp / "order.bin")
        np.ascontiguousarray(self.vals).tofile(tmp / "vals.bin")
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: os.PathLike) -> "FaginIndex":
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"fagin load: unsupported format_version "
                             f"{meta.get('format_version')}")
        N, D = int(meta["n_vectors"]), int(meta["dim"])
        Y = np.load(path / "Y.npy", mmap_mode="r")
        order = np.memmap(path / "order.bin", dtype=np.int32, mode="r",
                          shape=(D, N))
        vals = np.memmap(path / "vals.bin", dtype=np.float32, mode="r",
                         shape=(D, N))
        return cls(Y=Y, order=order, vals=vals, n_vectors=N, dim=D,
                   encoder_id=meta.get("encoder_id"))

    # ----- search -----

    def search(self, q: np.ndarray, K: int, batch: int = 64,
               epsilon: float = 0.0, max_depth: int = 0,
               num_threads: int = 0) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Run TA for one query. Returns (idxs int64[K], scores fp32[K],
        stats dict); unfilled slots (K > N) have idx -1."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"fagin search: q must have shape ({self.dim},); "
                             f"got {q.shape}")
        scores_b, idx_b, stats = _native.fagin_search(
            self.Y, self.order, self.vals, q,
            int(K), int(batch), float(epsilon), int(max_depth),
            int(num_threads))
        scores = np.frombuffer(scores_b, dtype=np.float32)
        idxs = np.frombuffer(idx_b, dtype=np.int64)
        return idxs, scores, stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n ndocu python -m pytest tests/test_fagin_index.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/retrieval/fagin/__init__.py \
        docuverse/engines/retrieval/fagin/fagin_index.py \
        tests/test_fagin_index.py
git commit -m "Add FaginIndex: per-dimension sorted lists with mmap persistence wrapping the native TA kernel"
```

---

### Task 4: fagin_* config fields

**Files:**
- Modify: `docuverse/engines/search_engine_config_params.py` (insert after the `simdq_ivf_nprobe` field, currently ending at line 597, before `def __post_init__`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fagin_index.py`:

```python
def test_retrieval_arguments_have_fagin_fields():
    from docuverse.engines.search_engine_config_params import RetrievalArguments
    cfg = RetrievalArguments()
    assert cfg.fagin_batch_rows == 64
    assert cfg.fagin_epsilon == 0.0
    assert cfg.fagin_max_depth == 0
    assert cfg.fagin_num_threads == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ndocu python -m pytest tests/test_fagin_index.py::test_retrieval_arguments_have_fagin_fields -v`
Expected: FAIL with `AttributeError: ... 'fagin_batch_rows'`

- [ ] **Step 3: Add the fields**

In `docuverse/engines/search_engine_config_params.py`, directly after the `simdq_ivf_nprobe` field block (line 593-597) and before `def __post_init__`, insert:

```python
    # ----- fagin threshold engine -----

    fagin_batch_rows: int = field(
        default=64,
        metadata={"help": "fagin: rows of sorted access taken from each active "
                          "dimension's list per TA round (B). Larger = fewer "
                          "threshold checks but more overshoot past the exact "
                          "stopping depth."}
    )

    fagin_epsilon: float = field(
        default=0.0,
        metadata={"help": "fagin: additive halting slack — stop when the kth best "
                          "score >= T - epsilon. 0.0 = exact Threshold Algorithm. "
                          "Additive rather than multiplicative because inner-product "
                          "thresholds can be negative."}
    )

    fagin_max_depth: int = field(
        default=0,
        metadata={"help": "fagin: cap on sorted-access depth (rows per dimension). "
                          "0 = unlimited (exact). Nonzero makes results approximate."}
    )

    fagin_num_threads: int = field(
        default=0,
        metadata={"help": "fagin: OpenMP thread count for the TA scan kernel; "
                          "0 = OMP default."}
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ndocu python -m pytest tests/test_fagin_index.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/search_engine_config_params.py tests/test_fagin_index.py
git commit -m "Add fagin_* retrieval config fields (batch rows, epsilon, max depth, num threads)"
```

---

### Task 5: FaginThresholdEngine

**Files:**
- Create: `docuverse/engines/retrieval/fagin/fagin_engine.py`
- Modify: `docuverse/engines/retrieval/fagin/__init__.py`

No standalone unit test in this task: the engine needs an encoder, so its
behavior is covered by the dispatch test (Task 6, no encoder) and the
end-to-end test (Task 7, tiny real encoder). This matches how `SimdqEngine`
is tested.

- [ ] **Step 1: Write the engine**

Create `docuverse/engines/retrieval/fagin/fagin_engine.py`:

```python
"""FaginThresholdEngine — DocUVerse engine running Fagin's Threshold
Algorithm (Fagin, Lotem, Naor, PODS'01) over per-dimension sorted lists.

Mirrors SimdqEngine: in-process, file-backed (no client/server split). At
ingest time we accumulate corpus-encoded fp32 vectors across all batches,
then build (NumPy argsort + gather) and save the index in one shot. At
search time the index is mmap-loaded once on the first query and each
query runs the native TA kernel — exact top-k inner product by default;
fagin_epsilon / fagin_max_depth trade exactness for speed.

Persist layout:
    <persist_directory>/fagin_data/<index_name>/
        meta.json
        Y.npy
        order.bin
        vals.bin
        engine_metadata.json      # id_map + per-doc metadata sidecar
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from docuverse.engines.retrieval.fagin.fagin_index import FaginIndex
from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import _trim_json, get_param
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction
from docuverse.utils.timer import timer

_STAT_KEYS = ("depth", "sorted_accesses", "random_accesses", "rounds", "exhausted")


class FaginThresholdEngine(RetrievalEngine):
    """In-process exact top-k engine using Fagin's Threshold Algorithm.

    Reads fagin_* fields from RetrievalArguments. Index lives at
    ``<persist_directory>/fagin_data/<index_name>/``.
    """

    SUBDIR = "fagin_data"

    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)
        self.model: Optional[DenseEmbeddingFunction] = None
        self.hidden_dim: Optional[int] = None
        self.index: Optional[FaginIndex] = None
        self.id_map: List[str] = []
        self.metadata_store: Dict[str, dict] = {}
        # Optional precomputed query embeddings, keyed by id(query); see
        # precompute_query_embeddings().
        self._query_emb_cache: Dict[int, np.ndarray] = {}
        # Aggregated TA access counters across all queries (thread-safe;
        # search_all runs searches from a thread pool). Exposed via info().
        self._stats_lock = threading.Lock()
        self.ta_stats: Dict[str, int] = {"queries": 0,
                                         **{k: 0 for k in _STAT_KEYS}}

        self.load_model_config(config_params)
        self.text_header = "text"
        self.title_header = "title"
        self.id_header = "id"
        self.extra_fields = get_param(self.config.data_template, "extra_fields", [])
        self.persist_directory = get_param(self.config, "project_dir", "/tmp")

        # Defer the (GPU) model load until first use — same rationale as
        # SimdqEngine (fork-friendly preprocessing).
        self._init_model_kwargs = kwargs
        self.init_client()

    # ===== Init =====

    def init_model(self, **kwargs):
        if self.model is not None:
            return
        self.model = DenseEmbeddingFunction(
            self.config.model_name,
            **self.config.__dict__,
        )
        self.hidden_dim = self.model.embedding_dim

    def _ensure_model(self):
        if self.model is None:
            self.init_model(**self._init_model_kwargs)

    def init_client(self):
        os.makedirs(os.path.join(self.persist_directory, self.SUBDIR), exist_ok=True)

    def check_client(self):
        pass

    # ===== Paths / has-index =====

    def _index_dir(self, index_name: Optional[str] = None) -> str:
        if index_name is None:
            index_name = self.config.index_name
        return os.path.join(self.persist_directory, self.SUBDIR, index_name)

    def _metadata_path(self, index_name: Optional[str] = None) -> str:
        return os.path.join(self._index_dir(index_name), "engine_metadata.json")

    def has_index(self, index_name: str) -> bool:
        return os.path.exists(os.path.join(self._index_dir(index_name), "meta.json"))

    def create_index(self, index_name: Optional[str] = None, **kwargs):
        # No-op: index files materialize at ingest end. We just clear any
        # existing partial directory.
        d = self._index_dir(index_name)
        if os.path.exists(d):
            shutil.rmtree(d)

    def delete_index(self, index_name: str, **kwargs):
        d = self._index_dir(index_name)
        if os.path.exists(d):
            shutil.rmtree(d)
            logging.info(f"Deleted fagin index: {index_name}")
        self.index = None
        self.id_map = []
        self.metadata_store = {}

    # ===== Ingest =====

    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs) -> bool:
        self.check_client()
        self._ensure_model()
        fmt = "\n=== {:30} ==="
        still_create_index = self.create_update_index(fmt=fmt, do_update=update)
        if not still_create_index:
            return None

        tm = timer("fagin::ingest")
        corpus_size = len(corpus)
        batch = self.ingestion_batch_size or self.config.bulk_batch or 256

        all_vecs: List[np.ndarray] = []
        all_ids: List[str] = []
        self.metadata_store = {}

        tq = tqdm(desc="fagin ingest", total=corpus_size, leave=True)
        for i in range(0, corpus_size, batch):
            chunk = corpus[i:min(i + batch, corpus_size)]
            texts: List[str] = []
            ids:   List[str] = []
            for j, doc in enumerate(chunk):
                t = _trim_json(get_param(doc, self.text_header, ""),
                               max_string_len=self.config.max_text_size)
                if not t:
                    continue
                doc_id = get_param(doc, self.id_header, f"doc_{i+j}")
                ids.append(doc_id)
                texts.append(t)
                meta = {
                    "text": t,
                    "title": get_param(doc, self.title_header, ""),
                }
                for f in self.extra_fields:
                    meta[f] = str(get_param(doc, f, ""))
                self.metadata_store[doc_id] = meta
            if not texts:
                tq.update(len(chunk))
                continue
            embs = self.model.encode(texts, show_progress_bar=False,
                                     _batch_size=len(texts), tm=tm)
            all_vecs.append(np.asarray(embs, dtype=np.float32))
            all_ids.extend(ids)
            tm.add_timing("encoding")
            tq.update(len(chunk))
        tq.close()

        if not all_vecs:
            raise RuntimeError("fagin ingest: no documents survived text filtering")
        Y = np.vstack(all_vecs)
        if Y.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f"fagin ingest: encoder produced dim {Y.shape[1]} but "
                f"hidden_dim={self.hidden_dim}"
            )

        self.index = FaginIndex.build(vectors=Y, encoder_id=self.config.model_name)
        out_dir = self._index_dir()
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        self.index.save(out_dir)
        # Side-car: id_map + metadata for rehydration.
        with open(self._metadata_path(), "w") as f:
            json.dump({"id_map": all_ids, "metadata": self.metadata_store}, f)
        self.id_map = all_ids
        tm.add_timing("build_save")
        logging.info(f"Ingested {corpus_size} docs into fagin index "
                     f"{self.config.index_name}")
        return True

    # ===== Search =====

    def _ensure_loaded(self):
        if self.index is None:
            self.index = FaginIndex.load(self._index_dir())
            with open(self._metadata_path()) as f:
                side = json.load(f)
            self.id_map = side["id_map"]
            self.metadata_store = side["metadata"]

    def precompute_query_embeddings(self, queries) -> None:
        """Batch-encode all query texts in one GPU forward pass (see
        SimdqEngine.precompute_query_embeddings for the rationale)."""
        self._ensure_loaded()
        self._ensure_model()
        items = list(queries)
        texts = [(q.text if hasattr(q, "text") else q) for q in items]
        if not texts:
            return
        tm = timer("fagin::precompute_query_embeddings")
        embs = self.model.encode(texts, show_progress_bar=False,
                                 prompt_name="query", tm=tm)
        embs = np.asarray(embs, dtype=np.float32)
        self._query_emb_cache = {id(q): embs[i] for i, q in enumerate(items)}
        tm.add_timing("encode_batch")

    def search_all(self, queries, num_threads: int = 1) -> List[SearchResult]:
        """Batch-encode all queries on the GPU, then scan in parallel on the
        CPU (thread pool; the native TA scan releases the GIL). Same two-phase
        structure as SimdqEngine.search_all."""
        self._ensure_loaded()
        items = list(queries)
        if not items:
            return []
        if num_threads is not None and num_threads < 0:
            num_threads = os.cpu_count() or 1

        self.precompute_query_embeddings(items)

        if num_threads is None or num_threads <= 1:
            return [self.search(q, scan_threads=1)
                    for q in tqdm(items, desc="fagin search", leave=True)]

        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: List[Optional[SearchResult]] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futs = {ex.submit(self.search, q, scan_threads=1): i
                    for i, q in enumerate(items)}
            with tqdm(total=len(items), desc="fagin search", leave=True) as tk:
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()
                    tk.update(1)
        return results

    def _accumulate_stats(self, stats: dict) -> None:
        with self._stats_lock:
            self.ta_stats["queries"] += 1
            for k in _STAT_KEYS:
                self.ta_stats[k] += int(stats.get(k, 0))

    def search(self, query: SearchQueries.Query, scan_threads: Optional[int] = None,
               **kwargs) -> SearchResult:
        tm = timer("fagin::search")
        self._ensure_loaded()
        tm.add_timing("load")

        cached = self._query_emb_cache.get(id(query))
        if cached is not None:
            q = np.asarray(cached, dtype=np.float32)
        else:
            self._ensure_model()
            text = query.text if hasattr(query, "text") else query
            emb = self.model.encode([text], show_progress_bar=False,
                                    prompt_name="query", tm=tm)[0]
            q = np.asarray(emb, dtype=np.float32)
        tm.add_timing("encode")

        K = int(self.config.top_k)
        idxs, scores, stats = self.index.search(
            q, K=K,
            batch=self.config.fagin_batch_rows,
            epsilon=self.config.fagin_epsilon,
            max_depth=self.config.fagin_max_depth,
            num_threads=(self.config.fagin_num_threads
                         if scan_threads is None else scan_threads),
        )
        self._accumulate_stats(stats)
        tm.add_timing("scan")

        passages = []
        for k_idx, score in zip(idxs, scores):
            if int(k_idx) < 0:
                continue
            doc_id = self.id_map[int(k_idx)]
            meta = self.metadata_store.get(doc_id, {})
            passages.append({
                "id": doc_id,
                "text": meta.get("text", ""),
                "title": meta.get("title", ""),
                "score": float(score),
                **{f: meta.get(f, "") for f in self.extra_fields},
            })
        tm.add_timing("result")
        return SearchResult(query, passages)

    # ===== Info =====

    def info(self) -> Dict[str, Any]:
        out = {
            "retriever_type": "FaginThresholdEngine",
            "index_name": self.config.index_name,
            "model": self.config.model_name,
            "dimension": self.hidden_dim,
            "batch_rows": self.config.fagin_batch_rows,
            "epsilon": self.config.fagin_epsilon,
            "max_depth": self.config.fagin_max_depth,
            "ta_stats": dict(self.ta_stats),
        }
        try:
            with open(os.path.join(self._index_dir(), "meta.json")) as f:
                out["index_meta"] = json.load(f)
        except FileNotFoundError:
            pass
        return out
```

- [ ] **Step 2: Export from the package**

Replace the contents of `docuverse/engines/retrieval/fagin/__init__.py` with:

```python
from docuverse.engines.retrieval.fagin.fagin_engine import FaginThresholdEngine
from docuverse.engines.retrieval.fagin.fagin_index import FaginIndex

__all__ = ["FaginThresholdEngine", "FaginIndex"]
```

- [ ] **Step 3: Sanity-check the import**

Run: `conda run -n ndocu python -c "from docuverse.engines.retrieval.fagin import FaginThresholdEngine; print(FaginThresholdEngine.SUBDIR)"`
Expected: prints `fagin_data`

- [ ] **Step 4: Run the existing fagin tests (no regression)**

Run: `conda run -n ndocu python -m pytest tests/test_fagin_index.py tests/test_fagin_native.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/retrieval/fagin/fagin_engine.py \
        docuverse/engines/retrieval/fagin/__init__.py
git commit -m "Add FaginThresholdEngine: in-process TA retrieval engine mirroring the SimdqEngine flow"
```

---

### Task 6: Dispatcher registration

**Files:**
- Modify: `docuverse/utils/retrievers.py:79-96`
- Modify: `tests/test_engine_dispatch.py:27-57`

- [ ] **Step 1: Write the failing test**

In `tests/test_engine_dispatch.py`, extend `KNOWN_NAMES` (after the
`"lancedb_hybrid",` entry, before the closing bracket) with:

```python
    "fagin",
    "fagin-threshold",
    "fagin_threshold",
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ndocu python -m pytest tests/test_engine_dispatch.py -v -k fagin`
Expected: 3 FAILs — `create_retrieval_engine` raises
`ValueError: Unknown or unsupported db_engine: 'fagin'` (which the test does
not catch).

- [ ] **Step 3: Add the dispatch branch**

In `docuverse/utils/retrievers.py`, after the `elif name == 'simdq':` block
(ends with `raise e` at line 86) and before `elif name.startswith("file:"):`,
insert (note: this file uses **3-space** indentation):

```python
   elif name in ['fagin', 'fagin-threshold', 'fagin_threshold']:
       try:
           from docuverse.engines.retrieval.fagin import FaginThresholdEngine
           engine = FaginThresholdEngine(retriever_config)
       except ImportError as e:
           print("fagin engine: native extension not built. Run `pip install -e .` "
                 "from the repo root to compile docuverse.engines.retrieval.simdq._native.")
           raise e
```

And update the supported-engines error string at the bottom of the function
from:

```python
           "(e.g. 'simdq', not 'simqd') against the supported engines: "
           "elastic-*, milvus-*, lancedb-*, chromadb, faiss, simdq, file:."
```

to:

```python
           "(e.g. 'simdq', not 'simqd') against the supported engines: "
           "elastic-*, milvus-*, lancedb-*, chromadb, faiss, simdq, fagin, file:."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n ndocu python -m pytest tests/test_engine_dispatch.py -v`
Expected: all PASS (the fagin cases reach the engine constructor and either
instantiate or die on the stub config with an accepted exception type)

- [ ] **Step 5: Commit**

```bash
git add docuverse/utils/retrievers.py tests/test_engine_dispatch.py
git commit -m "Register fagin/fagin-threshold engine names in the retrieval dispatcher"
```

---

### Task 7: End-to-end engine tests

**Files:**
- Create: `tests/test_fagin_engine.py`

- [ ] **Step 1: Write the end-to-end tests**

Create `tests/test_fagin_engine.py`:

```python
"""End-to-end FaginThresholdEngine tests with a real (tiny) encoder.

Mirrors tests/test_simdq_engine.py: skipped when sentence_transformers or
the native extension is unavailable."""
import os
import tempfile

import numpy as np
import pytest

from docuverse.engines.data_template import default_query_template
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_engine_config_params import RetrievalArguments
from docuverse.utils.retrievers import create_retrieval_engine

pytest.importorskip("docuverse.engines.retrieval.simdq._simdq_native")

TINY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d


class FakeCorpus:
    """Minimal corpus shim satisfying SearchCorpus's interface (len + slice)."""

    def __init__(self, docs):
        self._docs = docs

    def __len__(self):
        return len(self._docs)

    def __getitem__(self, index):
        return self._docs[index]


def _make_config(td: str) -> RetrievalArguments:
    cfg = RetrievalArguments()
    cfg.db_engine = "fagin"
    cfg.model_name = TINY_MODEL
    cfg.index_name = "test_fagin_e2e"
    cfg.project_dir = td
    cfg.top_k = 5
    cfg.ingestion_batch_size = 16
    cfg.max_text_size = 256
    return cfg


def _planted_corpus_docs():
    return [
        {"id": f"d{i}", "text": t, "title": ""}
        for i, t in enumerate([
            "the quick brown fox jumps over the lazy dog",
            "machine learning algorithms are statistical pattern matchers",
            "the cat sat on the warm mat in the morning sun",
            "neural networks approximate functions through layered transforms",
            "lions roar at dawn in the African savannah",
            "python is a high-level interpreted programming language",
            "transformers attend to all tokens via dot-product attention",
            "rabbits live in burrows and eat grass and clover",
            "BERT and GPT differ in their pretraining objectives",
            "elephants have the longest gestation of any mammal",
            "convolutional networks share weights across spatial positions",
            "the wolf is a social predator that hunts in packs",
        ])
    ] + [
        {"id": f"pad{j}", "text": f"padding doc {j}", "title": ""}
        for j in range(4)
    ]


@pytest.fixture
def fake_corpus():
    return FakeCorpus(_planted_corpus_docs())


def test_dispatch_and_round_trip(fake_corpus):
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        assert type(eng).__name__ == "FaginThresholdEngine"
        ok = eng.ingest(fake_corpus, update=False)
        assert ok is True
        meta_path = os.path.join(td, "fagin_data", "test_fagin_e2e", "meta.json")
        assert os.path.exists(meta_path)

        q = SearchQueries.Query(template=default_query_template, id="q1",
                                text="neural networks deep learning")
        res = eng.search(q)
        ids = [p["id"] for p in res.retrieved_passages]
        relevant = {"d1", "d3", "d6", "d8", "d10"}
        assert relevant.intersection(ids), \
            f"none of {relevant} in top-{cfg.top_k}: {ids}"
        # exact TA over 16 docs with K=5 must return exactly 5 results
        assert len(ids) == 5
        # stats accumulated
        assert eng.ta_stats["queries"] == 1
        assert eng.ta_stats["random_accesses"] >= 5


def test_search_matches_brute_force_over_encoded_corpus(fake_corpus):
    """The engine's exact mode must return exactly the brute-force top-k
    over the same encoded vectors — the core TA correctness claim, e2e."""
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        eng.ingest(fake_corpus, update=False)

        q = SearchQueries.Query(template=default_query_template, id="q1",
                                text="attention mechanism in transformers")
        res = eng.search(q)
        got_ids = [p["id"] for p in res.retrieved_passages]

        # Brute force over the persisted fp32 matrix (same vectors TA used).
        eng._ensure_loaded()
        Y = np.asarray(eng.index.Y, dtype=np.float64)
        qv = eng._query_emb_cache.get(id(q))
        if qv is None:
            eng._ensure_model()
            qv = np.asarray(eng.model.encode([q.text], show_progress_bar=False,
                                             prompt_name="query")[0])
        s = Y @ np.asarray(qv, dtype=np.float64)
        brute = [eng.id_map[i] for i in np.argsort(-s, kind="stable")[:cfg.top_k]]
        # Compare score-sets (tie-safe): every returned id must score >= kth
        kth = np.sort(s)[::-1][cfg.top_k - 1]
        idx_of = {d: i for i, d in enumerate(eng.id_map)}
        assert all(s[idx_of[d]] >= kth - 1e-5 for d in got_ids), \
            f"TA returned {got_ids}, brute force top-{cfg.top_k} is {brute}"


def test_search_all_parallel_matches_sequential(fake_corpus):
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        eng.ingest(fake_corpus, update=False)

        queries = [
            SearchQueries.Query(template=default_query_template, id=f"q{i}", text=t)
            for i, t in enumerate([
                "neural networks deep learning",
                "wild animals in the savannah",
                "interpreted programming languages",
                "attention mechanism in transformers",
                "small furry mammals",
            ])
        ]
        seq = [[p["id"] for p in eng.search(q).retrieved_passages] for q in queries]
        for nt in (1, 4):
            res = eng.search_all(queries, num_threads=nt)
            got = [[p["id"] for p in r.retrieved_passages] for r in res]
            assert got == seq, f"search_all(num_threads={nt}) diverged"


def test_delete_then_reingest(fake_corpus):
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        eng.ingest(fake_corpus, update=False)
        assert eng.has_index(cfg.index_name)
        eng.delete_index(cfg.index_name)
        assert not eng.has_index(cfg.index_name)
        eng2 = create_retrieval_engine(cfg)
        eng2.ingest(fake_corpus, update=False)
        assert eng2.has_index(cfg.index_name)


def test_info_reports_ta_stats(fake_corpus):
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        eng.ingest(fake_corpus, update=False)
        q = SearchQueries.Query(template=default_query_template, id="q1",
                                text="wolves hunting")
        eng.search(q)
        info = eng.info()
        assert info["retriever_type"] == "FaginThresholdEngine"
        assert info["ta_stats"]["queries"] == 1
        assert info["ta_stats"]["depth"] >= 1
        assert "index_meta" in info
```

- [ ] **Step 2: Run the tests**

Run: `CUDA_VISIBLE_DEVICES=1 conda run -n ndocu python -m pytest tests/test_fagin_engine.py -v`
Expected: all PASS (or all SKIP if `sentence_transformers`/the tiny model is
unavailable — in that case note it and rely on Tasks 1-6 coverage). If a
failure points at the engine (not the test), fix `fagin_engine.py` and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fagin_engine.py
git commit -m "Add end-to-end FaginThresholdEngine tests: round trip, brute-force agreement, parallel search, reingest, stats"
```

---

### Task 8: Full-suite verification

**Files:** none new.

- [ ] **Step 1: Run the full test suite**

Run: `CUDA_VISIBLE_DEVICES=1 conda run -n ndocu python -m pytest tests/ -q`
Expected: no new failures relative to the pre-plan baseline (run
`git stash && conda run -n ndocu python -m pytest tests/ -q && git stash pop`
first if a baseline is needed; several suites may skip on missing optional
backends — that is normal).

- [ ] **Step 2: Confirm benchmark hookup**

Run: `conda run -n ndocu python -c "
from docuverse.engines.search_engine_config_params import RetrievalArguments
c = RetrievalArguments(); c.db_engine='fagin'
from docuverse.utils.retrievers import create_retrieval_engine
print('dispatch OK:', type(create_retrieval_engine(c)).__name__)"`
Expected: prints `dispatch OK: FaginThresholdEngine` (or a clean config error
naming a missing field — not an unknown-engine error). This confirms existing
`ingest_and_test` configs work by switching `db_engine: fagin`.

- [ ] **Step 3: Final commit (if any stragglers)**

```bash
git status --short   # should show nothing fagin-related uncommitted
```
