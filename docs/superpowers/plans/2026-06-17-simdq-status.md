# `simdq` — plan & implementation status

**Last updated:** 2026-06-17
**Branch:** `v0.1.2`
**Spec:** [`docs/superpowers/specs/2026-06-15-simdq-engine-design.md`](../specs/2026-06-15-simdq-engine-design.md)

## Multi-plan roadmap

The `simdq` engine is split into **three sequential plans**, each shipping
working, testable software on its own.

| Plan | Scope | Status |
|---|---|---|
| **Plan 1** — kernels & standalone benches | C/SIMD kernels, top-K, asymmetric `b∈{1,2,4}` for D=768, ctest, benchmark binaries. No Python. | **Complete** ([plan](2026-06-15-simdq-plan-1-kernels.md)) |
| **Plan 2** — Python integration | CPython C-API binding, `SimdqIndex` class, numpy ingest (quantization + random-orthogonal projection), on-disk format, mmap rescore tier, threaded asymmetric scan driver. End-to-end at D=768. | Not started |
| **Plan 3** — engine + multi-D + BEIR sweep | `SimdqEngine` (DocUVerse `SearchEngine` subclass), factory dispatch, config dataclass, kernel templates for `D ∈ {384, 1024, 1536}`, recipe sweep script producing R0–R6 CSV. | Not started |

## Plan 1 — final status

### Tasks

| # | Task | Commit(s) | Status |
|---|------|-----------|--------|
| T1 | Port hb5 Hamming kernel into `_native/` | `47a6752`, `62324ee` | ✅ |
| T2 | Bounded max-heap top-K (`simdq_topk_t`) | `7b94333`, `f22fe5f` | ✅ |
| T3 | Hamming top-K kernels (AVX-512 + AVX2 + threaded) | `4d0ceeb`, `938e5ef` | ✅ |
| T4 | `bench_hamming` top-K driver | `3774326` | ✅ |
| T5 | b=1/2/4 SoA pack helpers + scales | `3a5856d`, `96c147d` | ✅ |
| T6 | b=1 asymmetric scan kernel | `8b0f284`, `79fc4cd` | ✅ |
| T7 | `bench_asym_b1` | `f1737e3` | ✅ |
| T8 | b=2 asymmetric scan kernel | `b8e3620` | ✅ |
| T9 | `bench_asym_b2` | `b11fa28` | ✅ |
| T10 | b=4 asymmetric scan kernel | `f4457d2` | ✅ |
| T11 | `bench_asym_b4` | `180bfea` | ✅ |
| T12 | Sweep script + `BENCHMARKS.md` | `f3a20f3` | ✅ |
| T13 | Final integration check | (no new commit; verification only) | ✅ |

**17 commits, 2844 insertions, all confined to `docuverse/engines/retrieval/simdq/_native/`.**

### What's in the tree (`docuverse/engines/retrieval/simdq/_native/`)

```
include/
  simdq_common.h                ported helpers (RNG, alloc, fill_soa_parallel)
  simdq_topk.h                   bounded max-heap, caller-allocated, asserts k>0/<=256
  simdq_kernels_hamming.h        ported top-1 Hamming (AVX-512 + AVX2 paths)
  simdq_kernels_hamming_topk.h   top-K Hamming + scan_batch_parallel_topk (OpenMP)
  simdq_pack.h                   simdq_pack_b{1,2,4} + simdq_unpack_b{1,2,4}
                                 + simdq_pack_scales (||y||/sqrt(d))
  simdq_kernels_asym_b1.h        scan_asym_b1_d768_topk, both SIMD paths
  simdq_kernels_asym_b2.h        scan_asym_b2_d768_topk, both SIMD paths
  simdq_kernels_asym_b4.h        scan_asym_b4_d768_topk, both SIMD paths
bench/
  bench_hamming_top1.c           = the existing hb5 driver
  bench_hamming.c                top-K driver (single query, threaded)
  bench_asym_b1.c                single-threaded
  bench_asym_b2.c                single-threaded
  bench_asym_b4.c                single-threaded
tests/
  test_kernels_hamming.c         ported, both SIMD paths
  test_topk.c                     8 sub-tests
  test_kernels_hamming_topk.c    K=1, planted, random, threaded
  test_pack.c                     b=1/2/4 round-trip + b=4 saturation clamp
  test_kernels_asym_b1.c         random + planted-best
  test_kernels_asym_b2.c         random + planted-best
  test_kernels_asym_b4.c         random + planted-best
scripts/
  run_sweep.sh                   runs every bench at {500k, 1M, 50M}
CMakeLists.txt
BENCHMARKS.md                    captured sweep + interpretation
```

### Test coverage

12 ctest targets, all passing on every build:

```
kernels_hamming_native      kernels_hamming_avx2
kernels_hamming_topk_native kernels_hamming_topk_avx2
kernels_asym_b1_native      kernels_asym_b1_avx2
kernels_asym_b2_native      kernels_asym_b2_avx2
kernels_asym_b4_native      kernels_asym_b4_avx2
topk                         pack
```

The `*_native` targets compile with `-march=native` and pick AVX-512F if
available; the `*_avx2` targets force `-mavx2 -mpopcnt` (or `-mavx2 -mfma`
for the asymmetric kernels) so the AVX2 fallback is exercised even on
AVX-512 hosts.

### Headline benchmark numbers (AVX2-only host, 32 threads, K=100)

| Bench | n | cmp/s | GB/s | Notes |
|---|---|---|---|---|
| `bench_hamming_top1` (NQ=8 batched) | 50M | 2308 M | 27.7 | Carry-over of existing `hb5` performance |
| `bench_hamming` (top-K=100, single query) | 50M | 740 M | 71.1 | Per-query throughput ~2.5× higher than top-1 NQ=8 due to lower register pressure |
| `bench_asym_b1` (single-threaded) | 50M | 0.55 M | 0.1 | Limited by AVX2 scalar bit-expansion in inner loop |
| `bench_asym_b2` (single-threaded) | 50M | 2.19 M | 0.4 | AVX2-FMA, 16-bit unpack |
| `bench_asym_b4` (single-threaded) | 50M | 2.36 M | 0.9 | AVX2-FMA, 32-bit unpack |

Full sweep at 500k / 1M / 50M is captured in
[`_native/BENCHMARKS.md`](../../../docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md).

### Known follow-ups (deferred to Plan 2 / 3)

1. **AVX2 b=1 unpack bottleneck** — the asymmetric b=1 AVX2 path
   expands packed bits via a per-iteration 8-step scalar loop. Replace
   with `_mm256_set1_epi8(bits)` + `vpshufb` / movemask trick. Plan 2
   should fix this before benchmarking R1/R2 recipes.

2. **AVX-512F throughput numbers** — the dev host lacks AVX-512F, so the
   AVX-512 inline kernel paths are correctness-tested but not measured.
   Re-run `scripts/run_sweep.sh` on an AVX-512F-capable box (e.g. Zen 4
   or Skylake-X) to populate the missing rows.

3. **Threaded asymmetric driver** — Plan 1 ships the asymmetric scans
   single-threaded only. Plan 2 will add a `scan_asym_b<b>_d768_topk_parallel`
   wrapper following the existing T3 OpenMP pattern (per-thread top-K
   heap + `omp critical` merge with `INT64_MAX`/-1 init for partial-fill
   safety).

4. **Macro renames for Python binding** — `KERNEL_NAME`, `LANES`,
   `ASYM_KERNEL_NAME`, `ASYM_LANES` are defined in multiple kernel
   headers with the same names. Plan 1 binaries each include only one
   header; Plan 2's Python binding will pull all four into a single
   translation unit and must rename them (e.g. `HAMMING_TOPK_LANES`,
   `ASYM_B2_LANES`) first.

5. **Code-review cleanups deferred from T1** — `-O3 -march=native` is
   set on every target unconditionally; gating `-O3` to release/relwithdebinfo
   would unblock future Debug/ASAN builds. No urgency for Plan 1.

## Plan 2 — preview (not started)

From the spec, Plan 2 will deliver:

```
docuverse/engines/retrieval/simdq/
├── __init__.py
├── simdq_index.py            # Python wrapper around the C extension
├── projection.py             # identity / random-orthogonal W ∈ R^{d×D}
├── quantization.py           # numpy-side scale fitting + b-bit pack call
└── _native/
    └── bindings/             # CPython C-API ext (no pybind11 dep)
```

Public API (deferred design — exact shape may shift after Plan 1 review):

```python
from docuverse.engines.retrieval.simdq import SimdqIndex

index = SimdqIndex.build(
    vectors=Y,                 # numpy (N, D) fp32
    b=2, d=384,                # halve dims, double bits
    projection="random_orthogonal", projection_seed=42,
    store_floats=True,         # mmap'd fp16 rescore tier
)
index.save("/scratch/idx-b2-d384")

index = SimdqIndex.load("/scratch/idx-b2-d384")
indices, scores = index.search(query_fp32, K=100, K_prime=1000)  # two-stage
indices, scores = index.search(query_fp32, K=100, K_prime=100)   # codes-only
```

On-disk index layout (per the spec):

```
<index_path>/
├── meta.json
├── W.npy
├── codes.bin           # SoA-packed codes
├── scales.bin          # per-vector fp16
└── floats.bin          # optional fp16 rescore tier, mmap'd at search time
```

Plan-2 scope also includes the threaded asymmetric driver (item 3 above)
and the macro renames (item 4). The AVX2 b=1 unpack fix (item 1) should
land here too — or earlier as a stand-alone hotfix if it blocks
benchmarking decisions.

## Plan 3 — preview (not started)

- `SimdqEngine(SearchEngine)` — DocUVerse `SearchEngine` subclass.
- `SimdqConfig` dataclass added to `search_engine_config_params.py`.
- Factory dispatch: `"simdq"` registered in `create_retrieval_engine`.
- Kernel templates for `D ∈ {384, 1024, 1536}`.
- `scripts/bench_simdq_beir.py` — runs the R0–R6 recipe sweep through
  the existing `ingest_and_test` CLI on BEIR datasets, captures
  NDCG@10 / recall@100 / latency / index size in CSV.

The headline research question Plan 3 answers:

> **R0 vs R3 at the same byte budget** — does asymmetric b=2 alone
> match or beat 1-bit Hamming + float rescore?
