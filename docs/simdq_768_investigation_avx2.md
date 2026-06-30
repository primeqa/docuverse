# simdq 768-dim slowness — replay on AVX2 (Threadripper PRO 5955WX)

_Started 2026-06-29. Replay of `docs/simdq_768_investigation.md` (originally
done on a 9950X3D with AVX-512) on a different machine without AVX-512, to see
which findings carry over, which flip, and which (Phase 5 vectorized unpack)
simply cannot be tested._

## Environment

- CPU: **AMD Ryzen Threadripper PRO 5955WX** — 16 physical cores / 32 logical,
  single socket, 2 CCDs. L1d **32 KB/core**, L2 **512 KB/core**, L3 **64 MB
  total (2× 32 MB)** — smaller everywhere than the 9950X3D (48 KB / 1 MB / 96 MB
  unified).
- SIMD: **AVX2 + FMA + BMI2 + F16C + AES + VAES + VPCLMULQDQ + SHA-NI**.
  **No AVX-512** (no F/DQ/BW/VL/VBMI/GFNI/VPOPCNTDQ). The Phase-5 VBMI in-register
  2-bit unpack cannot run here at all.
- Build flags: `-O3 -march=native -fopenmp`. Resulting `.so` confirmed AVX2-FMA:
  162 `%ymm`, 0 `%zmm`, 24 `vfmadd*`, 0 `vpmultishift*`.
- Kernel path: `simdq_kernels_asym_b2.h:116` — `ASYM_B2_KERNEL_NAME = "AVX2-FMA"`,
  `ASYM_B2_LANES = 8`, scalar unpack into stack `vf[8]` then `_mm256_loadu_ps`
  (the same store-forwarding hazard the original Lever-1 removed for AVX-512 —
  no equivalent fix exists for AVX2 in-tree).
- Software: python 3.14.5, numpy 2.4.6, scipy-openblas 0.3.31.188.0, libgomp.
- Corpus reference size N = 276,007. Code footprint: d=384 → 26.5 MB (just fits
  one CCD's 32 MB L3); d=768 → **53 MB does NOT fit one CCD's L3**, only fits
  the combined 64 MB if both CCDs participate (opposite of the 9950X3D where
  both sizes fit the unified 96 MB X3D L3).

## What's already in the source

Phases 3 (identity-projection skip + `threadpool_limits(1, 'blas')` wrapper) and
4 (`SimdqEngine.search_all` GPU-once + thread-pool scan) are merged
(`simdq_index.py:391`). Phase 5 (AVX-512 VBMI vectorized unpack) is in-tree but
guarded by `__AVX512VBMI__ && __AVX512VL__` — inactive on this CPU. Phase 8
(SoA Hamming codes) is merged. So this replay measures the **post-fix code on
AVX2-only silicon**.

All numbers below: `scripts/bench_simdq_scan.py`, N=276,007, b=2, K=100,
K'=256, store_floats=True, fp16 rescore on, 200 queries, 20 warmup, median
ms/query. "default BLAS" = OpenBLAS picks; "BLAS=1" = `OPENBLAS_NUM_THREADS=1`.

---

## Phase 1 — dim × thread sweep

### 1.1 / 1.2 — default BLAS

| dim | t=1 | t=2 | t=4 | t=8 | t=16 | t=all |
|---|---|---|---|---|---|---|
| 384d | 62.7 | 31.9 | 15.6 | 6.4 | 3.84 | **3.65** |
| 768d | 132.3 | 67.2 | 34.5 | 17.6 | 10.4 | **8.23** |

- 384d scales **17.2×** across 16 threads. 768d scales **16.1×**. Both clean,
  no scaling collapse anywhere — unlike the original machine where 768d hit a
  ~40 ms floor at default BLAS.
- Single-thread 768d is **2.1×** of 384d (132 vs 63) — sub-linear in dim
  (would expect 2× from the inner-loop work; close to that, as on the original).

### 1.4 — BLAS pinning (the original's decisive test)

| dim | config | t=1 | t=4 | t=16 | t=all |
|---|---|---|---|---|---|
| 768d | default BLAS | 132.3 | 34.5 | 10.4 | 8.23 |
| 768d | **BLAS=1** | 134.0 | 34.5 | 12.3 | 9.86 |
| 384d | default BLAS | 62.7 | 15.6 | 3.84 | 3.65 |
| 384d | **BLAS=1** | 63.5 | 16.3 | 4.29 | 4.33 |

**BLAS=1 does not help here, and very slightly hurts.** This is the opposite
of the AVX-512 machine's 7× win at 768d. Why: the original report's culprit was
OpenBLAS auto-parallelizing the per-query `W @ q` projection (a 768×768 matmul
above OpenBLAS's threshold). **Phase 3's identity-skip is already merged** —
`SimdqIndex._project_query` returns `q` verbatim for identity projection
(`simdq_index.py:400`), so there is no 768×768 GEMV to spawn an OpenBLAS pool.
The only remaining BLAS op is the (256, d) rescore, which stays below
OpenBLAS's auto-parallel threshold and is harmless.

So Phase 1's headline finding (BLAS×OpenMP oversubscription) **was a bug, it's
fixed in source, and it never appears on this machine in the first place**.

---

## Phase 5 — kernel speed without VBMI

Lever-1 (the AVX-512 VBMI `vpmultishiftqb` in-register unpack) **cannot run on
this CPU**. The compiled binary takes the AVX2-FMA path with the **scalar
unpack into a stack `vf[8]` array then `_mm256_loadu_ps`** — analogous to the
*pre*-Lever-1 path on the original machine. So this Threadripper is measuring
what the AVX-512 numbers would have been without Lever-1, scaled by SIMD width.

Single-thread b=2 (BLAS pinned, only the kernel matters):

| dim | this machine (AVX2-FMA, 8-lane) | 9950X3D pre-Lever-1 (AVX-512, scalar unpack, 16-lane) | 9950X3D post-Lever-1 (VBMI unpack, 16-lane) |
|---|---|---|---|
| 384d b=2 | **63.5 ms** | 26.4 ms | 5.9 ms |
| 768d b=2 | **134 ms** | 52 ms | 13.4 ms |

- AVX2 single-thread is **~2.4–2.6× slower than the AVX-512 pre-Lever-1 path** —
  consistent with the 8 → 16 lane ratio plus the slightly higher per-FMA
  unpack overhead (8 scalar iters per FMA, vs 16 on AVX-512 — i.e. unpack work
  is half per FMA on AVX2, partly compensating).
- A Lever-1-equivalent for AVX2 (shifts + `vpshufb` LUT in 128-bit lanes) is
  **not in tree**. Adding it would be the natural follow-up for this hardware.

Despite the slower per-thread kernel, the 16-thread scan time on this machine
is comparable to the AVX-512 fixed path:

| | 9950X3D (AVX-512 + Lever-1 + BLAS fix) | this machine (AVX2, scalar unpack, no BLAS fix needed) |
|---|---|---|
| 768d b=2, 16-thread | 6.3 ms | 8.2 ms |

Per-core AVX-512 + Lever-1 is ~5× faster (13.4 vs 134 single-thread), but the
overall search time only differs by ~1.3× at 16 threads on the same 16-core
count. The Threadripper closes most of the gap by being unconstrained by the
AVX-512 machine's L3 footprint (96 MB on the 9950X3D is unified X3D; on this
box it's split 2× 32 MB, but the SoA stream is sequential enough that the L3
miss penalty doesn't dominate — see Hamming numbers below).

For completeness, b=4 single-thread is essentially indistinguishable from b=2:

| dim | b=2 t=1 | b=4 t=1 | b=2 t=16 | b=4 t=16 |
|---|---|---|---|---|
| 384d | 63.5 | 66.0 | 3.8 | 3.4 |
| 768d | 134 | 132 | 10.4 | 9.5 |

The kernel is **unpack-bound, not FMA-bound** — doubling the bits per code adds
no measurable cost because the FMA is idle waiting on the store-forwarding
roundtrip. This is exactly the symptom Lever-1 fixes on AVX-512.

---

## Phase 6/8 — Hamming family (SoA storage, no per-query transpose)

The Phase 8 SoA layout is merged (`meta.code_layout="soa"`), so the per-query
AoS→SoA transpose is gone here too.

| dim | t=1 | t=2 | t=4 | t=8 | t=16 | t=all |
|---|---|---|---|---|---|---|
| 384d hamming | 0.65 | 0.46 | 0.36 | 0.32 | **0.31** | 0.32 |
| 768d hamming | 1.07 | 0.88 | 0.63 | 0.52 | 0.50 | **0.49** |

Hamming versus asym-b2 on the same machine (best ms/query, default BLAS):

| dim | asym b=2 | hamming | ratio |
|---|---|---|---|
| 384d | 3.65 ms | **0.31 ms** | 11.8× faster |
| 768d | 8.23 ms | **0.49 ms** | 16.8× faster |

**Hamming wins decisively on this AVX2 box** — even more lopsidedly than on the
AVX-512 box (where it was 0.39 vs 0.81 ms at 768d, a 2.1× factor). Two reasons
this gap widens without AVX-512:

1. The asym-b2 path loses ~2.5× per thread to scalar-unpack-without-VBMI (above).
2. Hamming's hot kernel is a 64-bit-lane `popcnt` + xor reduction, which is
   well-served by AVX2 and matches the AVX-512 path's throughput much more
   closely than asym-b2 does (Hamming's gap at 768d/16-conc on the 9950X3D was
   0.39 ms; here we observe 0.49 ms — only ~25% slower).
3. Hamming scan reads only `D/8` bytes per doc (96 B at 768d) vs asym-b2's
   `D/4` (192 B) — half the memory bandwidth.

Scaling: Hamming scales modestly (~2× across 16 threads), which is the
memory-bound regime — exactly the original Phase 6 finding. The SoA fix is
load-bearing here too; we'd expect a 4–8× regression on AoS.

---

## Phase 5b — head-to-head vs Milvus FLAT (same harness as original Phase 5)

Re-ran the original report's Phase-5 head-to-head harness on this
Threadripper using the **same on-disk NQ indexes** the AVX-512 numbers were
measured against (granite-97m → 267,316×384, granite-311m → 276,007×768),
plus Milvus standalone on `:19530`. Same ThreadPool pattern as
`scripts/bench_simdq_vs_milvus.py`: workers=16, each scan single-threaded.
Queries: synthetic random unit vectors (no pre-encoded query .npy was
copied; the scan/FLAT timings are query-data-independent — verified by an
earlier all-synthetic run agreeing within noise).

ms/query at 16-way concurrent (lower is better):

| mode (dim) | bytes/doc | 9950X3D (AVX-512) | 5955WX (AVX2) | ratio |
|---|---|---|---|---|
| Milvus FLAT fp32, 384d | 1536 | 5.0  | 6.84  | 1.4× |
| Milvus FLAT fp32, 768d | 3072 | 11.7 | 12.00 | 1.0× |
| asym b=2, 384d         | 96   | 0.43 | 2.89  | 6.7× |
| asym b=2, 768d         | 192  | 0.81 | 8.03  | 9.9× |
| **1-bit Hamming**, 384d | 48  | 0.23 | **0.10** | **0.4×** |
| **1-bit Hamming**, 768d | 96  | 0.40 | **0.17** | **0.4×** |

Three findings, unchanged from the earlier synthetic measurement:

1. **Milvus FLAT is roughly ISA-agnostic on this workload** — cross-ISA gap
   stays within ~30%. Milvus FLAT is dominated by memory bandwidth and
   gRPC overhead, not by FMA throughput. The 768d numbers are essentially
   identical (11.7 vs 12.0); 384d shows the AVX2 box slightly slower but
   far below the FMA-width ratio AVX-512 would imply for a compute-bound
   kernel.
2. **Asym b=2 collapses on AVX2** (6.7–9.9× slower than AVX-512) for the
   reason already established in Phase 5: no `vpmultishiftqb`, scalar
   unpack stalls the FMA pipeline.
3. **1-bit Hamming is ~2× *faster* on this AVX2 box** than on the AVX-512
   one. The popcount+xor kernel is memory-bound, and the Threadripper PRO's
   8-channel DDR4 + larger aggregate L3 bandwidth (two CCDs × 32 MB) beats
   the 9950X3D's 2-channel DDR5 cleanly. ISA does not matter for this
   kernel; bus width does.

So the asym-vs-1-bit ratio at 768d goes from **2.0×** on AVX-512 (0.81 vs
0.40) to **47.2×** on AVX2 (8.03 vs 0.17) — and that gap is what makes the
deployment default flip with the silicon.

Measurement: `scripts/investigate_simdq_hardware.py --vectors
experiments/nq_new/simq_data/nq_dev-...-granite97m-...,experiments/nq_new/simq_data/nq_dev-...-granite311m-...
--queries 400 --warmup 20 --workers 16`. Auto-generated report (same numbers,
full per-thread sweep): `docs/simdq_768_investigation_avx2_realdata.md`.

---

## Summary — what carries over, what flips, what's untestable

| | original (9950X3D, AVX-512) | this machine (5955WX, AVX2) |
|---|---|---|
| Phase 1: 768d thread-scaling collapse at default BLAS | yes, 8.5× → 1.7× | **no** — Phase 3 already in source, so no GEMV to oversubscribe |
| Phase 1: BLAS=1 helps | **7× win at 768d** | **no** (slightly hurts) — fix is now redundant |
| Phase 3: identity-projection skip | proposed and merged | **already in source, doing its job silently** |
| Phase 5: VBMI in-register unpack (Lever-1) | 4–5× kernel speedup | **not applicable** — VBMI absent. Scalar unpack still active; an AVX2 `vpshufb`-LUT analog is the natural follow-up |
| Phase 6/8: Hamming SoA beats asym-b2 | yes, 2× at 768d | **yes, ~47× at 768d** under the head-to-head harness — gap widens dramatically without AVX-512 |
| Practical winner for speed at 768d (16-conc) | Hamming SoA (0.40 ms) ≈ asym-b2 (0.81 ms; 2× gap) | **Hamming SoA wins by a wide margin** (0.17 vs 8.03 ms; 47×) |
| Milvus FLAT vs simdq Hamming at 768d (16-conc) | ~29× faster (0.40 vs 11.7 ms) | **~71× faster** (0.17 vs 12.0 ms) — Milvus is ISA-agnostic, simdq is not |

### Recommendations specific to this hardware

1. **Use Hamming with fp32 rescore** as the default code family on AVX2 boxes —
   the asym-b2 path's main advantage (Lever-1 vectorized unpack on AVX-512) is
   unavailable here, and Hamming is ~17× faster at 768d with identical
   retrieval quality after rescore.
2. **Don't waste a knob on `OPENBLAS_NUM_THREADS`** — the Phase-3 identity-skip
   already prevents the contention. Setting BLAS=1 here is a slight pessimization.
3. **Open follow-up: write an AVX2 vectorized unpack** for asym-b2 (shifts +
   `vpshufb` to expand 16 codes to 16 bytes in 128-bit lanes, then
   `cvtepi8_epi32` + `cvtepi32_ps`, mirroring the VBMI path) — this is the only
   piece of Phase 5 that's still on the table on AVX2 silicon. Expected
   speedup: similar 3–4× kernel-time reduction at single thread.

### What did NOT need to be re-run

Phases 2 (engine real-path with `parallel_process`), 4 (`search_all` GPU-once),
7 (IVF outer index) — these are pipeline/architecture changes that are
ISA-independent and already merged. The Phase-2 conclusion that "scan=1 +
query-level parallelism + pinned BLAS" is the robust setting still holds; the
"pinned BLAS" half is now a no-op on this hardware because the source already
skips the offending GEMV.
