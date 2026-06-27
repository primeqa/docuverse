# simdq 768-dim slowness — investigation report

_Started 2026-06-27. Goal: explain why simdq b=2 scan is ~5× slower than Milvus
FLAT at 768-dim (49 ms vs 9.4 ms) while being ~2.3× **faster** at 384-dim. The
384→768 jump (3.2→49 ms) is ~15× for a 2× dim increase — strongly non-linear,
so something beyond "twice the work" is going on._

## Environment

- CPU: AMD Ryzen 9 9950X3D, 16 physical cores / 32 logical, 2 CCDs (one with
  3D V-cache). L1d 48 KB/core, L2 1 MB/core, L3 96 MB (shared, 1 instance).
- SIMD: AVX-512F/DQ/BW/VL **+ avx512vbmi + gfni + avx512_vbmi2 + vpclmulqdq**
  (i.e. `vpermb`/GFNI in-register 2-bit expansion is available but unused).
- Build flags: `-O3 -march=native -fopenmp` → AVX-512 16-lane kernel path.
- Corpus reference size N = 276,007 → `row_bytes = (N+3)/4 = 69,002` B/dim.
  Code footprint: d=384 → 26.5 MB, d=768 → 53 MB (both fit the 96 MB L3).

## Hot path (from code read)

`scan_asym_b2_shard_topk` (`simdq_kernels_asym_b2.h:38`): per dim `w`, per
16-doc block it (1) does a **strided** read `codes[w*row_bytes + (ii>>2)]`
(consecutive dims 69 KB apart), and (2) unpacks 16 2-bit codes via a **scalar
loop into a stack array `vf[16]`**, then `_mm512_loadu_ps(vf)` — a store→load
roundtrip that serializes the FMA pipeline. Both costs scale with `d`.

Leads going in:
- **L1 (compute):** scalar unpack via stack `vf[]` defeats vectorization.
- **L2 (memory):** SoA strided layout; per-block footprint ~`d` cache lines
  (24 KB at d=384 fits L1d; ~48 KB at d=768 is at/over L1d → spills to L2).
- **L3 (threading):** `num_threads=0` spans both CCDs + SMT; cross-CCD L3 +
  critical-section merge may make "all cores" a loss.
- **L4 (waste):** identity projection still runs a full `W@q` matmul.

---

## Phase 1 — measurements

All numbers: `scripts/bench_simdq_scan.py`, N=276,007, b=2, K=100, K'=256,
store_floats=True, fp16 rescore on. Median ms/query, scan-only (no GPU encode,
no parallel_process). "default BLAS" = OpenBLAS picks its own thread count.

### 1.1 / 1.2 — dim × thread sweep (default BLAS)

| dim | t=1 | t=2 | t=4 | t=8 | t=16 | t=all |
|---|---|---|---|---|---|---|
| 384d | 26.5 | 13.3 | 6.9 | 4.4 | **3.1** | 3.1 |
| 768d | 75.7 | 69.1 | 56.9 | 46.5 | **43.7** | 44.8 |

- 384d scales **8.5×** across 16 threads. 768d scales only **1.7×** — it hits a
  ~40 ms floor that does not parallelize.
- Single-threaded, 768d is only **2.85×** of 384d (75.7 vs 26.5) — i.e. roughly
  linear in dim. **The headline 15× gap is NOT in the kernel; it is a
  thread-scaling collapse that appears only at 768d.**

### 1.3 — component decomposition (probe.py, separate timing loops)

| dim | nt | full | native scan | proj W@q | python rest |
|---|---|---|---|---|---|
| 384 | 1 | 27.3 | 24.5 | 0.00 | 2.8 |
| 384 | 16 | 2.9 | 2.7 | 0.01 | 0.2 |
| 768 | 1 | 75.1 | 74.0 | 6.6* | (noise) |
| 768 | 16 | 61.9 | 29.4 | 0.14 | **32.3** |

The native scan at 768d/16t is only **29 ms**, yet full search is ~62 ms: ~32 ms
of non-kernel overhead appears **only at high dim + high thread count** (it is
~0 at 384d/16t and absent in spirit at 768d/1t). `proj` measured 6.6 ms* at
768d/1t is a BLAS-thread-spinup artifact, which itself flags the culprit.

### 1.4 — ROOT CAUSE: OpenBLAS thread contention

numpy here links **scipy-openblas (OpenBLAS 0.3.31, DYNAMIC_ARCH)**. The
per-query projection `W @ q` ((768,768)·(768,)) and rescore
`cand_floats @ q_proj` ((256,768)·(768,)) are run by numpy through OpenBLAS.
At 768d these GEMV/GEMM sizes cross OpenBLAS's auto-parallel threshold, so
OpenBLAS spins up **its own ~16-thread pool per query**, on top of the scan's
16 OpenMP threads → 32+ threads oversubscribing 16 physical cores → cache/
scheduler thrash. At 384d the matrices stay below the threshold, OpenBLAS runs
single-threaded, and there is no contention (hence 384d scales cleanly).

**Decisive test — pin BLAS with `OPENBLAS_NUM_THREADS=1`:**

| dim | config | t=1 | t=4 | t=16 | t=all |
|---|---|---|---|---|---|
| 768d | default BLAS | 75.7 | 56.9 | 43.7 | 44.8 |
| 768d | **BLAS=1** | 53.5 | 14.1 | **6.3** | 6.4 |
| 384d | default BLAS | 26.5 | — | 3.1 | 3.1 |
| 384d | **BLAS=1** | 26.4 | — | 3.0 | 3.0 |

- 768d best case: **43.7 → 6.3 ms (7×)** — now **faster than Milvus FLAT
  (9.4 ms)**, not 5× slower.
- 768d now scales 8.5× across 16 threads (53.5 → 6.3), like 384d.
- 384d is unchanged (3.1 → 3.0 ms): the fix costs nothing where it wasn't needed.

---

## Conclusion (Phase 1)

The 768-dim "slowness" is **not** a compute-bound 2-bit unpack kernel, as the
status report assumed. It is **OpenBLAS auto-parallelizing the tiny per-query
projection/rescore GEMVs and contending with the scan's OpenMP threads**. The
single-thread kernel scales linearly with dim and is healthy; the native scan
at 768d/16t is only ~29 ms and drops to ~6 ms once BLAS stops oversubscribing.

**The 49 ms vs 9.4 ms result in the status report (`§3`) was measured under
thread oversubscription and should be retracted/re-run.** With BLAS pinned,
simdq b=2 at 768d is ~6.3 ms — competitive with / faster than Milvus FLAT.

### Recommended fixes (in order)
1. **Pin BLAS threads for the per-query GEMVs.** These ops are ~0.6 MFLOP and
   never benefit from threading. Either set `OPENBLAS_NUM_THREADS=1`/use
   `threadpoolctl.threadpool_limits(1)` around `SimdqIndex.search`, or do the
   matvecs without BLAS. This is the whole fix; everything else is incremental.
2. **Skip `W @ q` entirely when projection is identity** (`simdq_index.py:273`)
   — at 768d identity it's a wasted 768×768 matmul per query.
3. **Re-run and correct the status report's §3 speed table** with BLAS pinned.
4. (Deferred) The VBMI/GFNI in-register unpack and `d=D/2` projection ideas are
   still valid kernel optimizations but are no longer urgent — the kernel is not
   the bottleneck once contention is removed.

### Open follow-ups for Phase 2
- Confirm fix end-to-end (full `simdq_engine` path, not just `bench_simdq_scan`),
  since the engine's `parallel_process` adds another threading layer that can
  re-introduce oversubscription.
- Size sweep at 768d/BLAS=1 to recheck the simdq-vs-Milvus crossover.
- Check whether `precompute_query_embeddings` batched rescore would amortize the
  GEMV better than per-query calls.

---

## Phase 2 — end-to-end verification on the real engine path

The engine adds a **third** threading layer: `SearchEngine` runs
`parallel_process` over queries (`num_search_threads: -1` = all cores), each
query calls `index.search` (OpenMP scan, `simdq_num_threads: 0` = all cores),
which calls OpenBLAS (its own pool). So the real path stacks query-threads ×
scan-OMP × BLAS.

Confirmation that the §3 number came from this path: the recorded run
`output/...granite311m...timing.json` reports `simdq::search::scan` median
**44.78 ms** over 2578 queries — i.e. the contended per-call time, matching the
49 ms headline.

### 2.1 — real index (N=276,007, D=768, b=2) replayed through a query thread pool

Loaded the **actual on-disk index** and drove `index.search` via a
`ThreadPoolExecutor` mimicking `parallel_process`. Throughput as ms/query:

| config | default BLAS | **BLAS pinned=1** |
|---|---|---|
| engine real: scan=all, 16 query-threads | 9.72 (103 q/s) | **3.50 (286 q/s)** |
| scan=1, 16 query-threads | 11.49 | 3.96 |
| scan=all, **serial** queries | **49.32** | 6.45 |
| scan=1, serial queries | 81.50 | 51.69 |

### 2.2 — conclusions

1. **The 49 ms in §3 is the serial, default-BLAS scan** (49.32 here). It is not
   what the engine actually delivers: with `parallel_process` the real path is
   already 9.72 ms/q at default BLAS, and the per-call `scan` timer reads ~45 ms
   only because each contended thread's individual call is slow.
2. **Pinning BLAS gives a further 2.8× on the real engine path: 9.72 → 3.50
   ms/q (286 q/s)** — comfortably faster than Milvus FLAT (9.4 ms). The fix holds
   end-to-end, not just in the isolated bench.
3. **Nested OpenMP is unnecessary.** scan=all × 16 query-threads (256 OS threads)
   is only marginally faster than scan=1 + query-level parallelism (3.50 vs 3.96
   ms). The robust choice is **`simdq_num_threads=1` (single-threaded scan) +
   query-level `parallel_process` + pinned BLAS** — same speed, no 256-thread
   oversubscription that misbehaves on a busy box.
4. Single-threaded raw kernel (scan=1, serial, BLAS=1) = 51.7 ms — the true
   kernel cost. Everything above it was thread-pool interaction, not the kernel.

### Recommended engine settings
- Pin BLAS around `SimdqIndex.search` (`threadpool_limits(1)`), OR set
  `OPENBLAS_NUM_THREADS=1` for the run.
- Default `simdq_num_threads` to `1` (let `parallel_process` own the
  parallelism) instead of `0`.
- Skip `W @ q` when projection is identity.

---

## Phase 3 — fix implemented

Isolated which GEMV caused the contention (real index, default BLAS, 16
query-threads):

| variant | ms/query | q/s |
|---|---|---|
| baseline (`W@q` + `cand@q_proj`) | 8.80 | 114 |
| **skip `W@q` (identity)**, keep `cand@q_proj` | **3.34** | 299 |
| skip `W@q` + einsum rescore (no BLAS GEMV) | 3.41 | 293 |

So the **only** offender is the `W@q` projection (a 768×768 identity matmul
OpenBLAS parallelizes). The rescore `cand@q_proj` (256×768) stays below
OpenBLAS's threshold and is harmless. At 384d the projection matrix is also
below the threshold — which is exactly why 384d never showed the problem.

### Change (`docuverse/engines/retrieval/simdq/simdq_index.py`)
- New `SimdqIndex._project_query`: **skips the matmul entirely when projection
  is identity** (`q_proj = q`, bit-exact: `max|W@q − q| = 0.0`). For non-identity
  projections it keeps the matmul but wraps it in `_single_threaded_blas()`
  (`threadpoolctl.threadpool_limits(1, "blas")`, a no-op if threadpoolctl is
  absent) so the small GEMV can't spawn a competing thread pool.

### Verification (real NQ 768d index)
- **Correctness:** new (identity-skip) vs old (`W@q`) at `num_threads=1`
  (deterministic) → **50/50 identical**. The 1–4/50 differences seen at
  `num_threads=0` are pre-existing parallel-scan tie-break non-determinism (old
  path differs from *itself* 1/50 across re-runs), not caused by this change.
- **Speed:** real `SimdqIndex.search`, scan=all, 16 query-threads, **default
  BLAS (no env var)**: 8.80 → **3.92 ms/q (255 q/s)**. The fix no longer
  depends on setting `OPENBLAS_NUM_THREADS`.
- Test suite: `tests/test_simdq_*` green (see run).

---

## Phase 4 — fixing query-level parallelism (single-GPU)

The old engine path encoded queries inside `parallel_process`, which defaults to
**multiprocessing with `fork`**. `precompute_query_embeddings` creates a CUDA
context in the main process, then `parallel_process` forks — fork-after-CUDA is
unsupported, so workers can't safely use the GPU. Net effect: query-level
parallelism only paid off with **one GPU per worker**.

**Fix:** `SimdqEngine.search_all(queries, num_threads)` makes the two phases
explicit and keeps the GPU in the main process:
1. **Pool all query texts → one batched GPU forward pass** (no fork after CUDA).
2. **Parallelize only the CPU scan with a thread pool** — the native scan
   releases the GIL, so threads give real parallelism while sharing the
   in-process embedding cache (no pickling, no model reload). Each scan runs
   single-threaded (`scan_threads=1`) to avoid nested-OMP oversubscription.

`SearchEngine.search` now delegates to `retriever.search_all` when present
(else the old `precompute + parallel_process` path). Verified equivalent to
per-query `search` at `num_threads` ∈ {1, 4}
(`tests/test_simdq_engine.py::test_search_all_parallel_matches_sequential`).
This reuses the exact threaded-scan pattern measured in §2.1 (3.9 ms/q, 255 q/s
on the real 768d index).

---

### Status report corrections needed (`docs/simdq_status_report.md` §3, §6.1)
- The "768d simdq b=2 = 49 ms, ~5× slower than Milvus FLAT" result was measured
  under BLAS×OpenMP thread oversubscription. **Real number is ~3.9 ms/q**
  (faster than Milvus FLAT's 9.4 ms). Update the table and remove the "768-dim
  scan kernel speed" from the primary open-items list — the kernel was never the
  bottleneck.
