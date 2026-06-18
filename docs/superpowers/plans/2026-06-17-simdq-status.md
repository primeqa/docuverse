# `simdq` — plan & implementation status

**Last updated:** 2026-06-18 (Plan 3 complete)
**Branch:** `v0.1.2`
**Spec:** [`docs/superpowers/specs/2026-06-15-simdq-engine-design.md`](../specs/2026-06-15-simdq-engine-design.md)

## Multi-plan roadmap

The `simdq` engine is split into **three sequential plans**, each shipping
working, testable software on its own.

| Plan | Scope | Status |
|---|---|---|
| **Plan 1** — kernels & standalone benches | C/SIMD kernels, top-K, asymmetric `b∈{1,2,4}` for D=768, ctest, benchmark binaries. No Python. | **Complete** ([plan](2026-06-15-simdq-plan-1-kernels.md)) |
| **Plan 2** — Python integration | CPython C-API binding, `SimdqIndex` class, numpy ingest (quantization + random-orthogonal projection), on-disk format, mmap rescore tier, threaded asymmetric scan driver. End-to-end at D=768. | **Complete** ([plan](2026-06-17-simdq-plan-2-python.md)) |
| **Plan 3** — engine + multi-D + BEIR sweep | `SimdqEngine` (DocUVerse `SearchEngine` subclass), factory dispatch, config dataclass, runtime-`d` generalization for `D ∈ {384, 768, 1024, 1536}` with `d ∈ {D, D/2}`, recipe sweep script producing R0–R6 CSV. | **Complete** ([plan](2026-06-18-simdq-plan-3-engine.md)) |

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

1. ~~**AVX2 b=1 unpack bottleneck**~~ — resolved in Plan 2 T2 (commit
   `51eb47a`). AVX2 b=1 inner loop now uses `_mm256_set1_epi8 +
   cmpeq_epi8 + cvtepi8_epi32 + blendv_ps`; bench at n=50M went from
   0.55 M cmp/s → 8.44 M cmp/s on the AVX2 dev box.

2. **AVX-512F throughput numbers** — still open. The dev host lacks
   AVX-512F, so the AVX-512 inline kernel paths are correctness-tested
   but not measured. Re-run `scripts/run_sweep.sh` on an AVX-512F-capable
   box (e.g. Zen 4 or Skylake-X) to populate the missing rows.

3. ~~**Threaded asymmetric driver**~~ — resolved in Plan 2 T3 (commit
   `df76441`). Each `scan_asym_b{1,2,4}_d768_topk_parallel` mirrors the
   Hamming top-K pattern (per-thread heap + `omp critical` merge with
   `idx=-1` init for partial-fill safety). 1/2/8-thread parity verified
   in `tests/test_kernels_asym_parallel.c`.

4. ~~**Macro renames for Python binding**~~ — resolved in Plan 2 T1
   (commits `8c81fa7` and `83140cb`). All four kernel headers now use
   per-kernel-prefixed macros (`HAMMING_TOPK_*`, `ASYM_B{1,2,4}_*`) and
   compose cleanly in `bindings/module.c`.

5. **Code-review cleanups deferred from T1** — still open. `-O3
   -march=native` is set on every target unconditionally; gating `-O3`
   to release/relwithdebinfo would unblock future Debug/ASAN builds.

## Plan 2 — final status

### Tasks

| # | Task | Commit(s) | Status |
|---|------|-----------|--------|
| T1 | Rename per-kernel macros (`HAMMING_TOPK_*`, `ASYM_B{1,2,4}_*`) | `8c81fa7`, `83140cb` | ✅ |
| T2 | AVX2 b=1 SIMD bit-to-float unpack | `51eb47a` | ✅ |
| T3 | `(i0,i1)` shard refactor + `scan_asym_b{1,2,4}_d768_topk_parallel` | `df76441` | ✅ |
| T4 | CPython C-API binding `_simdq_native` | `7362b42` | ✅ |
| T5 | `setup.py` shim wiring the extension | `fa320e3` | ✅ |
| T6 | `projection.py` (identity / random_orthogonal) | `2f525c1` | ✅ |
| T7 | `quantization.py` (numpy `fit_scales` / `pack`) | `4733270` | ✅ |
| T8 | `SimdqIndex` (build/save/load/search) | `187c3dd` | ✅ |
| T9 | Status doc update | (this commit) | ✅ |

**9 commits across `_native/`, `bindings/`, and the new
`docuverse/engines/retrieval/simdq/` Python package.**

### What's in the tree

```
docuverse/engines/retrieval/simdq/
├── __init__.py                    # public re-export of SimdqIndex
├── simdq_index.py                 # build/save/load/search, mmap'd rescore
├── projection.py                  # identity, random_orthogonal, apply
├── quantization.py                # fit_scales, pack, unpack_levels
└── _native/
    └── bindings/
        └── module.c               # CPython C-API: 7 entry points

setup.py                            # setuptools shim declaring the Extension

tests/
├── test_simdq_projection.py       # 7 tests
├── test_simdq_quantization.py     # 6 tests
└── test_simdq_index.py            # 6 tests (round-trip, mode parity, etc.)
```

Plus 1 new ctest target (`kernels_asym_parallel_native` /
`kernels_asym_parallel_avx2`) and modifications to the three asym
kernel headers, three asym bench drivers, `CMakeLists.txt`,
`run_sweep.sh`, and `BENCHMARKS.md`.

### Test coverage

- **C ctest** — 14 targets (12 from Plan 1 + 2 new parallel-parity tests),
  all passing on every clean build.
- **Python pytest** — 19 tests across the three new files, all passing.

### Headline benchmark numbers (AVX2-only host, K=100)

| Bench | n | threads | cmp/s | GB/s | Notes |
|---|---|---|---|---|---|
| `bench_asym_b1` (Plan 2 T2 fix) | 50M | 1 | 8.44 M | 0.8 | 15× speedup vs Plan 1's 0.55 M |
| `bench_asym_b1` (Plan 2 T3 threaded) | 50M | 32 | 73 M | 7.0 | ~8.6× over single-thread |
| `bench_asym_b2` (Plan 2 T3 threaded) | 50M | 32 | 28 M | 5.4 | memory-bound |
| `bench_asym_b4` (Plan 2 T3 threaded) | 50M | 32 | 27 M | 10.5 | memory-bound, larger codes |

Single-threaded baselines and full sweep at 500k / 1M / 50M live in
[`_native/BENCHMARKS.md`](../../../docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md).

### Public API shipped

```python
import numpy as np
from docuverse.engines.retrieval.simdq import SimdqIndex

# build
Y = ...                                                # (N, 768) fp32
idx = SimdqIndex.build(vectors=Y, b=2, store_floats=True)
idx.save("/scratch/myindex")

# load and search
idx = SimdqIndex.load("/scratch/myindex")
indices, scores = idx.search(query_fp32, K=10, K_prime=100)   # two-stage
indices, scores = idx.search(query_fp32, K=10, K_prime=10)    # codes-only
```

### Open for Plan 3

- **`d` fixed at 768.** Plan 2's `SimdqIndex.build` rejects `d != D`
  with a clear message. Plan 3 will template the kernels for
  `D ∈ {384, 1024, 1536}` and enable `d = D/2` (the "halve dims, double
  bits" recipe).
- **No `SimdqEngine`/factory dispatch yet.** `simdq` is not registered
  in `create_retrieval_engine`; YAML configs cannot select it. Plan 3
  adds the `SearchEngine` subclass + factory entry + `SimdqConfig`.
- **No NQ-batched scan kernel.** `search_batch` will be a Python loop in
  Plan 3 until the batched kernel lands.
- **Per-vector scale applied post-scan in Python.** Codes-only mode
  ranks by raw `<q', v_i>` then multiplies by `scales[idx]`; for
  near-uniform per-vector norms (unit-norm embeddings) the ranking is
  unchanged. For variable-norm corpora, two-stage rescore with
  `K_prime > K` is the documented mitigation. Kernel-side scale
  application is a Plan 3 follow-up if BEIR sweep shows it matters.
- **`global` quantization mode deferred.** Spec section 6 mentions
  `per_vector` (shipped) and `global` (one fp32 in metadata). Plan 2
  ships only `per_vector`; `global` not implemented.
- **b=4 quantizer step-size mismatch with N(0,1) per-dim
  distributions.** The `simdq_pack_b4` formula uses a step-2 grid in
  `yn` units and is well-tuned for embeddings with heavier-tailed
  per-dim distributions; for isotropic unit-norm Gaussians it
  under-utilizes its 16 levels (most values land in {-1, +1}). Recall
  on the test's synthetic Gaussian inputs caps around 50% even at
  `K_prime=256`. Real embeddings (granite, ST) exercise the range
  better — Plan 3's BEIR sweep will measure this directly.

## Plan 3 — final status

### Tasks

| # | Task | Commit(s) | Status |
|---|------|-----------|--------|
| T1 | Generalize asym kernels for runtime `d` | `ff6d73f`, `8f7dd7f` | ✅ |
| T2 | Generalize Hamming kernels for runtime `words` | `3ad02a2`, `c8210cb` | ✅ |
| T3 | Multi-D ctest spot-check | `c6a598f` | ✅ |
| T4 | CPython binding: runtime `d` + Hamming entry points | `fa31feb`, `1b06fd0` | ✅ |
| T5 | `SimdqIndex`: `D ∈ {384,768,1024,1536}`, `family` axis | `77288cf` | ✅ |
| T6 | Pytest parametrize multi-D + hamming family | `c73ba9f` | ✅ |
| T7 | `simdq_*` fields on `RetrievalArguments` | `324a6a9` | ✅ |
| T8 | `SimdqEngine` `SearchEngine` subclass | `bf4bd20` | ✅ |
| T9 | Factory dispatch for `simdq` | `c816469` | ✅ |
| T10 | End-to-end `SimdqEngine` pytest | `ab05fd9` | ✅ |
| T11 | `scripts/bench_simdq_beir.py` R0–R6 sweep | `67fce62` | ✅ |
| T12 | Status doc update | (this commit) | ✅ |

**14 commits across `_native/` (kernel headers, bindings, multi-D ctest), the
Python `simdq/` package (`projection.py`, `quantization.py`, `simdq_index.py`,
new `simdq_engine.py`), `RetrievalArguments`, `create_retrieval_engine`,
three updated pytests + one new pytest, and the new `scripts/bench_simdq_beir.py`
+ `config/beir_simdq_base.yaml`.**

### What's in the tree (added this plan)

```
docuverse/engines/retrieval/simdq/
├── simdq_engine.py                  # NEW: SimdqEngine(RetrievalEngine)
├── simdq_index.py                   # rewritten: family axis, D ∈ supported set
├── projection.py                    # SUPPORTED_D, SUPPORTED_d, validate_D_d
├── quantization.py                  # pack_hamming wrapper; D_FIXED removed
└── _native/
    ├── include/                     # all kernel headers: runtime d / words
    ├── bindings/module.c            # runtime d on every entry; pack/scan_hamming
    └── tests/test_kernels_multi_D.c # NEW: spot-check all (d, words) variants

docuverse/engines/search_engine_config_params.py    # +8 simdq_* fields
docuverse/utils/retrievers.py                       # +simdq factory branch

tests/
├── test_simdq_projection.py         # +parametrize over (D, d)
├── test_simdq_quantization.py       # +multi-d round-trip + hamming round-trip
├── test_simdq_index.py              # +multi-D asym + hamming family
└── test_simdq_engine.py             # NEW: dispatch + ingest + search

scripts/bench_simdq_beir.py          # NEW: R0–R6 recipe sweep driver
config/beir_simdq_base.yaml          # NEW: base YAML for the sweep
```

### Test coverage

- **C ctest** — 16 targets (14 from Plan 2 + 2 new `kernels_multi_D_*`),
  all passing on every clean build.
- **Python pytest** — 64 tests across the four simdq test files
  (61 from `test_simdq_{projection,quantization,index}.py` after
  parametrization + 3 from new `test_simdq_engine.py`), all passing.
  `test_simdq_engine.py` `pytest.importorskip`s if sentence-transformers
  is unavailable.

### Architectural deviations from spec

- **Runtime-`d` kernel parameterization** rather than compile-time-D
  templates (spec §6). Justification: at `d ≥ 384` the inner loop is too
  long for full unroll; benchmark-driven follow-up if perf shows otherwise.
  The macro template re-include path remains a clean deferred follow-up.
- **`SimdqConfig` realized as `simdq_*` fields on `RetrievalArguments`**
  rather than a standalone dataclass (spec §8). Matches existing engine
  conventions (`milvus_*`, `lancedb_*`, …).

### Headline benchmark numbers

To be populated after running `scripts/bench_simdq_beir.py` on a real
BEIR dataset. Suggested first run: FiQA (~57k passages, encoder =
granite-embedding-278m).

### Open for v2 / Plan 3.5

- Streaming-build path (avoids `4·N·D` peak RAM at ingest time).
- Compile-time D templates if BEIR sweep shows runtime-d loses ≥10%
  on memory-bound recipes.
- Learned projection (full ASH training loop).
- `global` quantization mode (one fp32 in metadata header) — Plan 2
  shipped only `per_vector`.

### The headline research question Plan 3 answers

> **R0 vs R3 at the same byte budget** — does asymmetric b=2 alone
> match or beat 1-bit Hamming + float rescore?

Awaits the first `bench_simdq_beir.py` run on a real dataset to
populate the answer in the recipe-sweep CSV.
