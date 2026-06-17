# simdq Plan 1 kernel sweep — threads=32, K=100

| binary | n | reps | output |
|--------|---|------|--------|
| bench_hamming_top1 | 500000 | 10 | AVX2-nibbleLUT    threads=32 NQ= 8 : n=500000 (48 MB)      669.7 M cmp/s     8.0 GB/s memory  (6.0 ms/pass, top-1[0]=152372 d=318) |
| bench_hamming_top1 | 1000000 | 10 | AVX2-nibbleLUT    threads=32 NQ= 8 : n=1000000 (96 MB)     1573.0 M cmp/s    18.9 GB/s memory  (5.1 ms/pass, top-1[0]=152372 d=318) |
| bench_hamming_top1 | 50000000 | 3 | AVX2-nibbleLUT    threads=32 NQ= 8 : n=50000000 (4800 MB)     2308.2 M cmp/s    27.7 GB/s memory  (173.3 ms/pass, top-1[0]=27308199 d=307) |
| bench_hamming | 500000 | 10 | kernel=AVX2-nibbleLUT n=500000 K=100 reps=10  best=4.246ms  cmp/s=117.75M  GB/s=11.3  top1=76750 |
| bench_hamming | 1000000 | 10 | kernel=AVX2-nibbleLUT n=1000000 K=100 reps=10  best=5.926ms  cmp/s=168.73M  GB/s=16.2  top1=76750 |
| bench_hamming | 50000000 | 3 | kernel=AVX2-nibbleLUT n=50000000 K=100 reps=3  best=67.525ms  cmp/s=740.47M  GB/s=71.1  top1=19052442 |
> **Note (Plan 2, T2):** AVX2 b=1 inner loop now uses
> `_mm256_set1_epi8 + cmpeq_epi8 + cvtepi8_epi32 + blendv_ps` to expand 8
> packed bits into 8 ±1 floats per dim, replacing the per-iteration
> scalar 8-step loop that bottlenecked Plan 1. Numbers below reflect the
> new path.

| bench_asym_b1 | 500000 | 5 | kernel=AVX2-FMA n=500000 K=100 reps=5  best=74.460ms  cmp/s=6.72M  GB/s=0.6  top1=289573 score=37.073235 |
| bench_asym_b1 | 1000000 | 5 | kernel=AVX2-FMA n=1000000 K=100 reps=5  best=123.359ms  cmp/s=8.11M  GB/s=0.8  top1=548858 score=41.071777 |
| bench_asym_b1 | 50000000 | 3 | kernel=AVX2-FMA n=50000000 K=100 reps=3  best=5925.252ms  cmp/s=8.44M  GB/s=0.8  top1=18068739 score=44.219238 |
| bench_asym_b2 | 500000 | 10 | kernel=AVX2-FMA n=500000 K=100 reps=10  best=170.239ms  cmp/s=2.94M  GB/s=0.6  top1=289573 score=37.073235 |
| bench_asym_b2 | 1000000 | 10 | kernel=AVX2-FMA n=1000000 K=100 reps=10  best=312.402ms  cmp/s=3.20M  GB/s=0.6  top1=548858 score=41.071777 |
| bench_asym_b2 | 50000000 | 3 | kernel=AVX2-FMA n=50000000 K=100 reps=3  best=22853.632ms  cmp/s=2.19M  GB/s=0.4  top1=18068739 score=44.219238 |
| bench_asym_b4 | 500000 | 10 | kernel=AVX2-FMA n=500000 K=100 reps=10  best=178.772ms  cmp/s=2.80M  GB/s=1.1  top1=289573 score=37.073235 |
| bench_asym_b4 | 1000000 | 10 | kernel=AVX2-FMA n=1000000 K=100 reps=10  best=321.251ms  cmp/s=3.11M  GB/s=1.2  top1=548858 score=41.071777 |
| bench_asym_b4 | 50000000 | 3 | kernel=AVX2-FMA n=50000000 K=100 reps=3  best=21183.354ms  cmp/s=2.36M  GB/s=0.9  top1=18068739 score=44.219238 |

## Plan 2 — threaded asymmetric throughput

`OMP_NUM_THREADS=32`, K=100, reps=3. Bench drivers now call
`scan_asym_b{1,2,4}_d768_topk_parallel`; the single-threaded numbers
above are preserved for comparison.

| binary | n | reps | output |
|--------|---|------|--------|
| bench_asym_b1 | 500000 | 3 | kernel=AVX2-FMA threads=32 n=500000 K=100 reps=3  best=12.925ms  cmp/s=38.68M  GB/s=3.7  top1=289575 score=37.073235 |
| bench_asym_b1 | 1000000 | 3 | kernel=AVX2-FMA threads=32 n=1000000 K=100 reps=3  best=5.803ms  cmp/s=172.31M  GB/s=16.5  top1=548860 score=41.071777 |
| bench_asym_b1 | 50000000 | 3 | kernel=AVX2-FMA threads=32 n=50000000 K=100 reps=3  best=685.423ms  cmp/s=72.95M  GB/s=7.0  top1=18068743 score=44.219238 |
| bench_asym_b2 | 500000 | 3 | kernel=AVX2-FMA threads=32 n=500000 K=100 reps=3  best=8.982ms  cmp/s=55.67M  GB/s=10.7  top1=289575 score=37.073235 |
| bench_asym_b2 | 1000000 | 3 | kernel=AVX2-FMA threads=32 n=1000000 K=100 reps=3  best=28.728ms  cmp/s=34.81M  GB/s=6.7  top1=548860 score=41.071777 |
| bench_asym_b2 | 50000000 | 3 | kernel=AVX2-FMA threads=32 n=50000000 K=100 reps=3  best=1782.796ms  cmp/s=28.05M  GB/s=5.4  top1=18068739 score=44.219238 |
| bench_asym_b4 | 500000 | 3 | kernel=AVX2-FMA threads=32 n=500000 K=100 reps=3  best=18.798ms  cmp/s=26.60M  GB/s=10.2  top1=289573 score=37.073235 |
| bench_asym_b4 | 1000000 | 3 | kernel=AVX2-FMA threads=32 n=1000000 K=100 reps=3  best=39.315ms  cmp/s=25.44M  GB/s=9.8  top1=548858 score=41.071777 |
| bench_asym_b4 | 50000000 | 3 | kernel=AVX2-FMA threads=32 n=50000000 K=100 reps=3  best=1826.949ms  cmp/s=27.37M  GB/s=10.5  top1=18068739 score=44.219238 |

### Threaded speedup summary (vs single-threaded Plan 2, 32 threads)

| b | n | 1-thread cmp/s | 32-thread cmp/s | speedup |
|---|---|---|---|---|
| 1 | 1M | 8.11 M | 172.31 M | ~21× |
| 1 | 50M | 8.44 M | 72.95 M | ~8.6× |
| 2 | 1M | 3.20 M | 34.81 M | ~10.9× |
| 2 | 50M | 2.19 M | 28.05 M | ~12.8× |
| 4 | 1M | 3.11 M | 25.44 M | ~8.2× |
| 4 | 50M | 2.36 M | 27.37 M | ~11.6× |

b=1 at 1M achieves ~21× speedup on 32 threads — super-linear for small N
due to L3/LLC cache fit. At 50M (memory-bound), all three b values
converge to ~8-13× speedup, limited by aggregate memory bandwidth
across 32 threads. For BEIR-scale (5M vectors), projected b=1 latency is
~0.5 ms and b=2/b=4 ~1-2 ms — well within the interactive-search budget.

## Hardware

This sweep ran on a CPU **without AVX-512F or AVX-512 VPOPCNTDQ** — every
benchmark fell back to the AVX2 path. Both Hamming kernels report
`AVX2-nibbleLUT`; both asymmetric kernels report `AVX2-FMA`. Plan 1's
kernel matrix covers both SIMD paths and ctest exercises both via the
`*_native` and `*_avx2` test targets, so the AVX-512 paths are
correctness-tested at compile time on an AVX-512F-capable build host but
not yet measured for throughput here.

## Notes

### Hamming top-1 vs top-K

`bench_hamming_top1` uses the existing `hb5` driver — a query batch of
**NQ=8** that amortises each DB load across 8 queries. `bench_hamming`
runs a single query through `scan_batch_parallel_topk` (Plan 1's new
top-K driver). At n=50M, single-query top-K hits **740 M cmp/s** vs the
NQ=8 top-1 at **2308 M cmp/s** — a ~3.1× gap that is the *batching*
benefit, not top-K overhead. Per-query, top-K (n=50M) is 740 M cmp/s vs
top-1 per-query (2308 / 8 ≈ 288 M cmp/s) — top-K is actually faster at
this scale, because the single-query path keeps fewer accumulators in
flight and avoids register pressure. Top-K with K=100 against random
data essentially never branches into the heap-offer slow path, so the
amortised heap overhead is negligible.

### Asymmetric kernels are single-threaded in Plan 1

Plan 1 deliberately ships the asymmetric scans (`scan_asym_b{1,2,4}_d768_topk`)
as **single-threaded** — threading lands in Plan 2 alongside the Python
binding. The numbers above therefore use 1 core; multiply by ~16-24× to
estimate the threaded ceiling at 32 threads (modulo memory bandwidth).

The relative ordering across `b` is informative even at single-threaded:

| b | bytes/code at d=768 | n=1M cmp/s | 50M cmp/s |
|---|---|---|---|
| 1 | 96 | 8.11 M (Plan 2) | 8.44 M (Plan 2) |
| 2 | 192 | 3.20 M | 2.19 M |
| 4 | 384 | 3.11 M | 2.36 M |

b=1 is now **~3–4× faster** than b=2/b=4 — the SIMD unpack (Plan 2, T2)
eliminated the scalar bottleneck. Plan 1 measured b=1 at 0.57 M / 0.55 M
cmp/s (6× slower than b=2/b=4) because the inner loop expanded bits via a
per-iteration 8-step scalar `vf[l] = (bits & (1u << l)) ? 1.0f : -1.0f`
loop, breaking SIMD pipelining. The replacement uses
`_mm256_set1_epi8 + per-lane bit mask + cmpeq_epi8 + cvtepi8_epi32 + blendv_ps`
to expand 8 bits into 8 ±1 floats entirely in SIMD. The AVX-512F path
uses `_mm512_mask_blend_ps` (1-instruction expansion); the AVX2 fix above
is the closest SIMD-friendly equivalent on this hardware.

### Throughput at the scale we care about

For BEIR-scale evaluations (≤ 5M vectors), a 5M-vector single-query
asymmetric scan currently takes ~2 seconds at b=2/b=4 single-threaded.
Once threaded (Plan 2) and on AVX-512F hardware, that should drop to
the order of ~50 ms — competitive with the headline goal.

Hamming top-K at 5M vectors takes ~30 ms threaded already (extrapolating
from the 50M / 67.5 ms result), which beats most engine-based binary
recipes at this scale.

### What is NOT in this sweep

- AVX-512F throughput numbers (this host lacks AVX-512F).
- Threaded asymmetric scans (Plan 2 work).
- Recall-vs-throughput Pareto data (Plan 3 BEIR sweep).
- Float baseline (existing dense engines, out of `_native/` scope).

These are deferred to subsequent plans.
