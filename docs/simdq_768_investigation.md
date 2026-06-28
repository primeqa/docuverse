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

## Phase 5 — Lever 1: vectorized sub-byte unpack (kernel speedup)

The `b2`/`b4` AVX-512 kernels unpacked codes with a per-dim **scalar loop into a
stack `vf[16]` then `_mm512_loadu_ps`** — a store-forwarding stall that dominated
the inner loop. Replaced with an in-register unpack:

- **`vpmultishiftqb`** (`_mm_multishift_epi64_epi8`, AVX512-VBMI) extracts the 16
  packed codes into 16 bytes in one instruction (control = per-lane bit offset,
  data = the packed word broadcast to both qwords).
- **`vpshufb`** (`_mm_shuffle_epi8`) maps code → level via a 4-entry (`b2`) /
  16-entry (`b4`) LUT, then `cvtepi8_epi32` + `cvtepi32_ps` feed the existing FMA.

Bit-identical to the scalar path (same level floats, same lane order, same FMA
sequence); a scalar fallback is kept under `#if !(__AVX512VBMI__ && __AVX512VL__)`.
Files: `simdq_kernels_asym_b2.h`, `simdq_kernels_asym_b4.h`.

> Gotcha: `_mm_multishift_epi64_epi8(a, b)` takes **a = control, b = data**
> (verified empirically; the Intel guide prose is ambiguous). Getting it backwards
> compiles and runs but returns wrong candidates.

### Verification
- `b2` and `b4` vs a numpy level-dot reference: **top-200 candidate set identical**,
  max score error ~3e-6 (fp32 reduction-order only). Single-thread == all-threads.
- Full simdq suite green (**108 passed**).

### Speed (single-thread scan, N=276,007, BLAS pinned)

| dim | before (scalar unpack) | after (Lever 1) | speedup |
|---|---|---|---|
| 384d b=2 | 26.4 ms | **5.9 ms** | 4.5× |
| 768d b=2 | 52 ms | **13.4 ms** | 3.9× |

### End-to-end on the real 768d NQ index (default BLAS, with the projection fix)

| config | ms/query | q/s |
|---|---|---|
| scan=1, serial | 13.5 | 74 |
| scan=all, serial | 1.48 | 677 |
| **scan=1, 16 query-threads** | **0.95** | **1053** |
| scan=all, 16 query-threads | 0.99 | 1014 |

**768d cumulative: 49 ms → 3.9 ms (projection fix) → 0.95 ms (Lever 1).**
Next levers if needed: VNNI `vpdpbusd` MAC and multiple FMA accumulators.

### Head-to-head vs Milvus FLAT (one identical harness)

Re-benchmarked both with the **same harness**: search-only (queries pre-encoded),
same 276k vectors, same machine, Milvus standalone on :19530, 16-way concurrent.
ms/query:

| encoder | Milvus FLAT | simdq b=2 | speedup |
|---|---|---|---|
| granite-97m (384d) | 6.3 | **0.42** (2373 q/s) | ~15× |
| granite-311m (768d) | 12.2 | **0.89** (1128 q/s) | ~14× |

The Milvus figures reproduce the earlier ~7.6 / 9.4 ms within noise. simdq's codes
are also 16× smaller in RAM (96 vs 1536 B/doc at 384d).

---

### Status report corrections needed (`docs/simdq_status_report.md` §3, §6.1)
- The "768d simdq b=2 = 49 ms, ~5× slower than Milvus FLAT" result was measured
  under BLAS×OpenMP thread oversubscription. **Real number is ~3.9 ms/q**
  (faster than Milvus FLAT's 9.4 ms). Update the table and remove the "768-dim
  scan kernel speed" from the primary open-items list — the kernel was never the
  bottleneck.

---

## Phase 6 — Hamming family benchmark (quality + speed)

Built Hamming indexes from the **same stored embeddings** as the asym-b2 indexes
(no re-encode) for granite-97m (384d) and granite-311m (768d). Hamming = 1-bit
sign codes (AoS), query sign-quantized at search, scored by negative Hamming
distance, then fp32-rescored over the top-K' candidates (α=10, K'=256).

### Quality — real NQ dev eval (retrieve+eval via the engine, vs qrels)

768d (granite-311m), identical pipeline to the asym-b2 / fp32 runs:

| metric | fp32 | asym b=2 | **hamming** |
|---|---|---|---|
| NDCG@10 | 0.607 | 0.607 | **0.607** |
| NDCG@100 | 0.634 | 0.634 | **0.634** |
| Match@100 | 0.983 | 0.984 | **0.984** |
| MRR@10 | — | 0.554 | **0.554** |

**Hamming + fp32 rescore is lossless** — identical to asym-b2 and fp32. The 1-bit
ranking is coarse but good enough to pull the true top-100 into the top-256
candidate set, and the fp32 rescore fixes the final order.

> Caveat learned the hard way: quality **must** use real query embeddings.
> Recall-vs-exact with *random* query vectors gave ~0.15 (meaningless) because
> random queries have near-tied top-K that any quantizer reorders.

### Speed — same clean harness (search-only, same vectors, 16-way concurrent)

| dim | asym b=2 | hamming | fp32 (Milvus) |
|---|---|---|---|
| 384d | 0.41 ms/q | 1.14 ms/q | 6.3 ms/q |
| 768d | 0.81 ms/q | 3.5 ms/q | 12.2 ms/q |

**Hamming is slower than asym-b2 in practice**, despite 1-bit codes being half the
size of b=2. Decomposing the 768d hamming path:
- native popcount scan: 3.1 ms serial → **4.5 ms at 16-way concurrent** (negative
  scaling) — it's **memory/latency-bound**, so concurrent queries contend rather
  than scale. asym-b2's vectorized compute-bound scan parallelizes well instead.
- Python query sign-packing: 0.138 ms/query, **49× slower** than a vectorized
  `(signs.reshape(-1,64) * 2**arange(64)).sum()` (0.003 ms) and GIL-bound — minor
  next to the scan, but it inflates the engine's per-call `scan` timer badly under
  the threaded path (123 ms median observed).

### Verdict
Hamming's niche is **RAM**, not speed: 96 / 48 B/doc (768d / 384d) vs asym-b2's
192 / 96 and fp32's 3072 / 1536, at **identical retrieval quality** once rescored.
If latency/throughput matter, asym-b2 wins (~4× faster at 768d concurrent).
Open levers for hamming: vectorize the query pack (trivial, 49×), and the
memory-bound scan needs an outer/coarse index to scan fewer codes (a faster
kernel won't help a bandwidth wall).

---

## Phase 7 — IVF outer index for Hamming (scan fewer codes)

Hamming's scan is memory-bound (Phase 6): every query streams **all** N codes,
and the native `scan_hamming` even transposes the whole AoS corpus to SoA per
call. Added an **IVF (inverted-file) outer index** so a query only scans the
`nprobe` nearest clusters.

**Build** (`SimdqIndex.build(..., ivf_nlist=, ivf_nprobe=)`, hamming only):
k-means (faiss, sklearn fallback) into `nlist` clusters; reorder the AoS codes so
each cluster is a contiguous range (store cluster offsets + a permutation back to
original ids; floats stay in original order).

**Search** (`_search_hamming_ivf`): route by L2 to centroids (`argmax q·c − ½‖c‖²`),
**gather** the selected clusters' code ranges into one buffer, run a **single**
native scan, then fp32-rescore. The single gathered scan is essential — scanning
clusters with one native call each pays the per-call alloc+transpose `nprobe`
times and is *slower* than flat (measured: np=16 went from 5.8 ms → 1.4 ms after
switching to gather+single-call).

### Quality vs speed (real NQ dev eval, 768d, nlist=√N≈525)

| nprobe | % corpus | NDCG@10 | Match@100 | speed 16-conc |
|---|---|---|---|---|
| 16 | 3% | 0.591 | 0.949 | 1.38 ms |
| 32 | 6% | 0.601 | 0.969 | 2.22 ms |
| 64 | 12% | 0.605 | 0.979 | 4.05 ms |
| 128 | 24% | 0.607 | 0.982 | ~8 ms |
| flat (no IVF) | 100% | 0.607 | 0.984 | 3.36 ms |

IVF gives a real **speed/quality knob**: nprobe=32 is ~1.5× faster than flat
Hamming at ~99% of its NDCG@10; nprobe=16 is ~2.4× faster at ~97%. Past ~nprobe=64
the gather cost erases the win (use flat instead). Default `simdq_ivf_nprobe=32`.

### Honest verdict
IVF makes Hamming competitive with **itself**, not with asym-b2: even nprobe=16
(1.38 ms) is still slower than asym-b2's lossless 0.78 ms at 768d, and lossy. So
the practical frontier is unchanged — **asym-b2 for speed, Hamming(+IVF) only when
1-bit RAM (96/48 B/doc) is the hard constraint**, trading a little recall for a
smaller scan. A bigger structural win would be storing Hamming codes SoA once at
build to kill the per-query transpose (orthogonal to IVF).

Config: `simdq_ivf_nlist` (0 = off), `simdq_ivf_nprobe`. Wired through
`SimdqEngine`. Test: `tests/test_simdq_index.py::test_hamming_ivf_round_trip_and_full_probe_matches_flat`.

---

## Phase 8 — SoA storage kills the per-query transpose (Hamming)

`scan_hamming` consumed an SoA layout (`dbT[w*N+i]`) but the codes were stored
**AoS**, so the binding transposed the whole corpus AoS→SoA (a fresh `N*words*8`
alloc + copy) **on every query** — a non-shareable, memory-bound cost that was
most of Hamming's slowness and the reason it didn't scale across concurrent
queries (each query rebuilt its own SoA buffer).

**Fix:** store Hamming codes SoA at build time (transpose once), and add a
no-transpose, range-capable binding `scan_hamming_soa(codes, N, D, q, K, nt, i0, i1)`
backed by a new `scan_hamming_topk_parallel_range` kernel. Flat search scans
`[0, N)`; IVF gathers the selected clusters with one fancy-index and scans the
gathered buffer once. Old AoS indexes are transposed once on load (back-compat
via `meta.code_layout`). `pack_hamming` and the native C tests are untouched.

### Result — flat Hamming (real 768d NQ index, 16-way concurrent)

| | before (AoS, transpose/query) | after (SoA) |
|---|---|---|
| flat Hamming | 3.36 ms/q | **0.39 ms/q (8.6×)** |

Quality is **unchanged and lossless** (NDCG@10 0.607, NDCG@100 0.634, Match@100
0.984 — identical to fp32/asym-b2; SoA is just a layout change).

### This flips the verdict (supersedes Phase 6/7)

| encoder/mode (768d) | speed 16-conc | quality | bytes/doc |
|---|---|---|---|
| Milvus FLAT fp32 | 12.2 ms | exact | 3072 |
| asym b=2 | 0.81 ms | lossless | 192 |
| **Hamming flat (SoA)** | **0.39 ms** | **lossless** | **96** |
| Hamming IVF np=16/32 | 1.49 / 2.30 ms | 0.591 / 0.601 NDCG@10 | 96 (+centroids/perm) |

**Flat Hamming + fp32 rescore is now the fastest *and* smallest lossless option**
— faster than asym-b2 at half the code size. The earlier "Hamming is ~4× slower,
its niche is RAM not speed" conclusion was an artifact of the per-query transpose,
now removed.

### Is IVF still useful? Not at this scale.
With codes stored SoA, the whole 26.5 MB corpus is read once per query and stays
**L3-resident** (96 MB X3D) shared across concurrent queries, so the full scan is
0.39 ms — cheaper than IVF's gather+scan (1.49 ms at np=16) plus a recall loss.
IVF only pays off when the SoA codes **exceed L3** (corpora ≫ ~10–30 M × 96 B),
where a single full pass becomes DRAM-bound; it's kept (and tested) for that
regime. A native multi-range scan would cut its per-query overhead further.
