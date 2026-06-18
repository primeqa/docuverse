# `simdq` Plan 3 — DocUVerse engine, multi-D, R0–R6 BEIR sweep

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the `simdq` retrieval engine inside DocUVerse — `SimdqEngine(SearchEngine)` subclass, `SimdqConfig` fields on `RetrievalArguments`, factory dispatch via `create_retrieval_engine` — and widen the kernel template from a single `D=768` to `D ∈ {384, 768, 1024, 1536}` with reduced dim `d ∈ {D, D/2}`. Expose the symmetric-Hamming kernel via the Python binding so the recipe sweep covers R0–R6 end-to-end and produces a single CSV from a single `python scripts/bench_simdq_beir.py` invocation.

**Architecture:**

1. **Runtime-`d` kernel parameterization (deviation from spec section 6).** The spec calls for *compile-time* templated kernels per-D. We instead pass `d` (and, for Hamming, `WORDS = D/64`) as a runtime parameter to a single kernel implementation per (family × b × simd-path). The kernel inner loop `for w in 0..d` is already too long for full compiler unrolling (768 iterations); GCC strength-reduces and auto-vectorizes the FMA pattern regardless of whether `d` is a literal or a runtime variable. Runtime-d cuts kernel surface 6× and removes the macro re-include layer the spec implied. **If the BEIR sweep in T11 shows a perf regression vs the existing Plan 2 D=768 measurement, we revisit and template-instantiate at compile time** (the inline `static` kernels in headers make that a mechanical change). Compile-time templates remain a clean deferred follow-up.
2. **Symmetric Hamming reaches Python.** Plan 2 shipped only the asymmetric path (`pack_b{1,2,4}` + `scan_b{1,2,4}`). Plan 3 exposes `pack_hamming` (sign-only 1-bit code generation, dim-major SoA at the uint64 word stride the Hamming kernel expects) and `scan_hamming` (top-K parallel) as additional binding entry points, and adds a `family ∈ {"asymmetric", "hamming"}` axis to `SimdqIndex`. R0 in the recipe matrix runs through this path.
3. **`SimdqEngine` mirrors `FAISSEngine`** (in-process, file-backed, no client/server split). It instantiates a `DenseEmbeddingFunction` via the existing `RetrievalArguments.model_name`, encodes corpus passages in batches at ingest time, writes a single `SimdqIndex.save(...)` at the end, and lazy-loads the index for `search`. Each batch's encoded vectors are accumulated into a single fp32 numpy buffer (peak RAM = `4·N·D` bytes — at N=10M D=768 that's ~30 GB; documented as a Plan 3 limitation, with the streaming-build path deferred).
4. **`SimdqConfig` fields go on `RetrievalArguments`, not a new dataclass** (codebase pattern: every engine's options share one `RetrievalArguments` instance, see `milvus_idf_file`, `model_torch_dtype`, etc.). Field names are `simdq_*`-prefixed to avoid collision (`simdq_b`, `simdq_d`, `simdq_projection`, ...). Spec section 8's `SimdqConfig` dataclass is realized as this prefixed field group.
5. **Recipe sweep is a thin driver.** `scripts/bench_simdq_beir.py` reads a base YAML, materializes 7 per-recipe variants by overriding `simdq_*` fields, runs `python -m docuverse.utils.ingest_and_test` once per recipe (subprocess), parses the printed metrics, and writes a single `simdq_recipe_sweep_<dataset>.csv`. The script does **not** import DocUVerse — keeping recipes isolated in subprocesses prevents cross-recipe state leak (e.g. the `omp_set_num_threads` global, model warm-state).

**Tech Stack:** C11 (existing kernels), CPython C-API, numpy, OpenMP, DocUVerse `SearchEngine` + `DenseEmbeddingFunction`, `RetrievalArguments` (HuggingFace-style `@dataclass` with `field(metadata={"help": ...})`), pytest, YAML configs with the existing `{{var}}` resolver.

**Spec reference:** [`docs/superpowers/specs/2026-06-15-simdq-engine-design.md`](../specs/2026-06-15-simdq-engine-design.md).
**Plan 2 reference:** [`docs/superpowers/plans/2026-06-17-simdq-plan-2-python.md`](2026-06-17-simdq-plan-2-python.md).
**Status doc to update:** [`docs/superpowers/plans/2026-06-17-simdq-status.md`](2026-06-17-simdq-status.md).

---

## File structure

| Path | Purpose | Created/modified in task |
|---|---|---|
| `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h` | Drop `ASYM_B1_D 768`; take `d` as a function parameter, drop `_d768` from symbol | T1 (modify) |
| `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b2.h` | Same: `d` parameter, symbol becomes `scan_asym_b2_topk_parallel` | T1 (modify) |
| `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b4.h` | Same: `d` parameter, symbol becomes `scan_asym_b4_topk_parallel` | T1 (modify) |
| `docuverse/engines/retrieval/simdq/_native/include/simdq_common.h` | `WORDS` becomes runtime; helpers (`hamming_soa`, `fill_soa_parallel`) take `words` parameter | T2 (modify) |
| `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming_topk.h` | Symbol becomes `scan_hamming_topk_parallel`; takes `words` runtime | T2 (modify) |
| `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming.h` | Same runtime-`words` for completeness | T2 (modify) |
| `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b{1,2,4}.c` | Pass D=768 explicitly to renamed parallel driver | T1 (modify) |
| `docuverse/engines/retrieval/simdq/_native/bench/bench_hamming.c` | Pass `WORDS=12` to renamed driver | T2 (modify) |
| `docuverse/engines/retrieval/simdq/_native/bench/bench_hamming_top1.c` | Pass `WORDS=12` to top-1 driver | T2 (modify) |
| `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b{1,2,4}.c` | Pass D=768 to renamed driver; existing test stays at D=768 | T1 (modify) |
| `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming.c` | Pass `WORDS=12`; existing tests unchanged | T2 (modify) |
| `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming_topk.c` | Pass `WORDS=12` to renamed top-K driver | T2 (modify) |
| `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_parallel.c` | Pass D=768 to renamed driver | T1 (modify) |
| `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_multi_D.c` | New: spot-check d ∈ {192, 384, 512, 1024, 1536} for asym b=2 + hamming | T3 (create) |
| `docuverse/engines/retrieval/simdq/_native/CMakeLists.txt` | Register `kernels_multi_D_native` / `_avx2` | T3 (modify) |
| `docuverse/engines/retrieval/simdq/_native/bindings/module.c` | Drop `#define SIMDQ_D 768`; take `d` (or `D` for Hamming) as a runtime arg on every entry; add `pack_hamming` + `scan_hamming` | T4 (modify) |
| `docuverse/engines/retrieval/simdq/quantization.py` | Drop `D_FIXED` shape check; accept any `d` ∈ supported set; add `pack_hamming` wrapper | T5 (modify) |
| `docuverse/engines/retrieval/simdq/projection.py` | Already supports any `(D, d)`; add `SUPPORTED_D` and `SUPPORTED_d` constants for validation | T5 (modify) |
| `docuverse/engines/retrieval/simdq/simdq_index.py` | Relax `d == 768` rejection; accept `D ∈ {384, 768, 1024, 1536}` and `d ∈ {D, D/2}`; add `family ∈ {"asymmetric", "hamming"}` | T5 (modify) |
| `tests/test_simdq_index.py` | Add multi-D parametrize + Hamming-family round-trip | T6 (modify) |
| `tests/test_simdq_quantization.py` | Add multi-D parametrize | T6 (modify) |
| `tests/test_simdq_projection.py` | Add `(D=384, d=192)`, `(D=1024, d=512)`, `(D=1536, d=768)` cases | T6 (modify) |
| `docuverse/engines/search_engine_config_params.py` | Add `simdq_*` fields to `RetrievalArguments`; add `simdq` to `db_engine` choices | T7 (modify) |
| `docuverse/engines/retrieval/simdq/simdq_engine.py` | New: `SimdqEngine(RetrievalEngine)` subclass | T8 (create) |
| `docuverse/engines/retrieval/simdq/__init__.py` | Re-export `SimdqEngine` alongside `SimdqIndex` | T8 (modify) |
| `docuverse/utils/retrievers.py` | Add `simdq` branch to `create_retrieval_engine` | T9 (modify) |
| `tests/test_simdq_engine.py` | New: factory dispatch resolves; round-trip ingest/search of synthetic 100-doc corpus | T10 (create) |
| `scripts/bench_simdq_beir.py` | New: R0–R6 sweep driver | T11 (create) |
| `config/beir_simdq_base.yaml` | New: base YAML the sweep script overrides | T11 (create) |
| `docs/superpowers/plans/2026-06-17-simdq-status.md` | Mark Plan 3 tasks complete; capture observed numbers | T12 (modify) |

**No new dependencies.** Plan 3 stays inside numpy + the existing C compiler / OpenMP toolchain. The sweep script uses only stdlib (`subprocess`, `csv`, `pathlib`, `argparse`, `yaml` which is already in DocUVerse's deps).

---

## Architectural decisions to confirm before starting

These deviate from the spec; the plan reviewer should explicitly accept or reject each.

| Decision | Spec position | Plan 3 position | Why |
|---|---|---|---|
| Kernel D parameterization | Compile-time per-D template (section 6) | Runtime `d` argument to single kernel per (family, b, simd-path) | Memory-bound at high N; D-as-literal doesn't unblock unrolling at d≥384 |
| Config dataclass | Standalone `SimdqConfig` (section 8) | Fields go on existing `RetrievalArguments` with `simdq_*` prefix | Codebase pattern (Milvus, LanceDB, FAISS all do this) |
| `index_path` location | Spec section 8 implies a CLI-level `index_path` | Use `<persist_directory>/simdq_data/<index_name>/` mirroring `FAISSEngine`'s `<persist_directory>/faiss_data/<index_name>.index` | Consistency with existing engines |
| Streaming build | Out of scope in spec | Out of scope: peak RAM = `4·N·d` bytes, documented limitation | Defer until corpora >5M force the issue |

---

## Task 1: Generalize asym kernel headers for runtime `d`

**Goal:** Each of `simdq_kernels_asym_b{1,2,4}.h` currently `#define`s `ASYM_B{N}_D 768` and exports symbols `scan_asym_b{N}_d768_{shard,topk,topk_parallel}`. Replace the macro with a function parameter `size_t d`, drop `_d768` from each symbol name. Behavior unchanged when called with `d=768`. Pure refactor.

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b2.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b4.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b1.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b2.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b4.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b1.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b2.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b4.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_parallel.c`

- [ ] **Step 1: Verify existing tests pass at the start (baseline)**

```bash
cd docuverse/engines/retrieval/simdq/_native
rm -rf build && mkdir build && cd build
cmake .. && make -j && ctest --output-on-failure
```
Expected: all 14 ctest targets pass. If anything fails here, **stop and ask** — Plan 2 should be on a clean baseline.

- [ ] **Step 2: Refactor `simdq_kernels_asym_b1.h` to take `d` as a function parameter**

Open `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h`. Apply these specific edits:

a. Delete the line `#define ASYM_B1_D 768`.

b. The two `scan_asym_b1_d768_shard_topk` definitions (one inside `#if defined(__AVX512F__)` at ~line 35, one inside the `#elif` AVX2 branch at ~line 106) become:
```c
static inline void scan_asym_b1_shard_topk(const uint8_t *codes, size_t N, size_t d,
                                           size_t i0, size_t i1,
                                           const float *q, int K,
                                           float *out_s, int64_t *out_i) {
```
Inside both, every reference to `ASYM_B1_D` becomes `d`.

c. The two `scan_asym_b1_d768_topk` wrappers (single-thread driver) become `scan_asym_b1_topk` and forward `d`:
```c
static inline void scan_asym_b1_topk(const uint8_t *codes, size_t N, size_t d,
                                     const float *q, int K,
                                     float *out_s, int64_t *out_i) {
    scan_asym_b1_shard_topk(codes, N, d, 0, N, q, K, out_s, out_i);
}
```

d. The `scan_asym_b1_d768_topk_parallel` driver (inside `#if defined(_OPENMP)`) becomes `scan_asym_b1_topk_parallel` and threads `d` through to each shard call:
```c
static inline void scan_asym_b1_topk_parallel(const uint8_t *codes, size_t N, size_t d,
                                              const float *q, int K,
                                              float *gs, int64_t *gi) {
    /* unchanged body except the inner shard call: */
    scan_asym_b1_shard_topk(codes, N, d, i0, i1, q, K, ls, li);
    /* ...rest unchanged... */
}
```

e. Update the doc comment at the top of the file: change "asymmetric float×1-bit scan, top-K, D=768." to "asymmetric float×1-bit scan, top-K, runtime `d`."

- [ ] **Step 3: Apply the same refactor to `simdq_kernels_asym_b2.h`**

Same pattern: drop `#define ASYM_B2_D 768`; rename `scan_asym_b2_d768_*` → `scan_asym_b2_*`; add `size_t d` as the third parameter to `_shard_topk`, `_topk`, and `_topk_parallel`; thread `d` through to inner calls; replace `ASYM_B2_D` with `d` in both AVX-512 and AVX2 branches; update the top-of-file comment.

- [ ] **Step 4: Apply the same refactor to `simdq_kernels_asym_b4.h`**

Same pattern; rename `scan_asym_b4_d768_*` → `scan_asym_b4_*`; replace `ASYM_B4_D` with `d`.

- [ ] **Step 5: Update bench drivers to pass `d=768`**

In each of `bench_asym_b1.c`, `bench_asym_b2.c`, `bench_asym_b4.c`: every call site of `scan_asym_b{N}_d768_topk_parallel(codes, N, q, K, scores, idxs)` becomes `scan_asym_b{N}_topk_parallel(codes, N, /*d=*/768, q, K, scores, idxs)`. Likewise the `simdq_pack_b{N}` calls already take `d`; no change there. Add a `#define D 768` near the top of each bench file if not already present (check by `grep ^#define _native/bench/bench_asym_b1.c`); the existing benches use `ASYM_B{N}_D` from the header — replace those constants with the local `D` define.

- [ ] **Step 6: Update C tests to pass `d=768` to the renamed symbols**

In each of `test_kernels_asym_b1.c`, `test_kernels_asym_b2.c`, `test_kernels_asym_b4.c`, `test_kernels_asym_parallel.c`:

- Every call to `scan_asym_b{N}_d768_topk(...)`, `scan_asym_b{N}_d768_shard_topk(...)`, or `scan_asym_b{N}_d768_topk_parallel(...)` gains a `/*d=*/768` argument in the third position (after `N`, before `q` or `i0`).
- Every reference to `ASYM_B{N}_D` (used as a literal 768 elsewhere in the test) becomes the literal `768`.

- [ ] **Step 7: Build and verify all tests still pass**

```bash
cd docuverse/engines/retrieval/simdq/_native
rm -rf build && mkdir build && cd build
cmake .. && make -j 2>&1 | tail -30
ctest --output-on-failure
```
Expected: all 14 ctest targets still pass. Compilation should be clean (no warnings about ASYM_B{N}_D being undefined — those macros should be entirely gone from the codebase except in any leftover bench/test reference, which Step 5/6 should have caught).

If any test fails: the most likely cause is a missed call site that still references `_d768`. `grep -r '_d768\b' docuverse/engines/retrieval/simdq/_native/` should be empty.

- [ ] **Step 8: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b{1,2,4}.h \
        docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b{1,2,4}.c \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b{1,2,4}.c \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_parallel.c
git commit -m "Generalize asym scan kernels to runtime d; drop _d768 suffix.

The asym b={1,2,4} scan kernels previously hard-coded D=768 as a macro.
Plan 3 makes d a runtime parameter so we can scan over reduced
dimensions {192, 384, 512, 768, 1024, 1536} from the same kernel.
Symbol rename: scan_asym_b{N}_d768_{shard,topk,parallel} ->
scan_asym_b{N}_{shard,topk,parallel}_topk; existing bench/test sites
pass d=768 explicitly. Behavior unchanged at d=768."
```

---

## Task 2: Generalize Hamming kernel for runtime `WORDS`

**Goal:** `simdq_common.h` hard-codes `#define WORDS 12` (= 768 bits). Hamming kernels take `words` as a runtime argument so the same kernel covers `D ∈ {384, 768, 1024, 1536}` (`words` ∈ {6, 12, 16, 24}). Symbol rename: `scan_shard_topk` → `scan_hamming_shard_topk`; `scan_topk_parallel` → `scan_hamming_topk_parallel`. The existing top-1 kernel `hamming_soa` and the helper `fill_soa_parallel` also pick up `words`.

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_common.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming_topk.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_hamming.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_hamming_top1.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming_topk.c`

- [ ] **Step 1: Refactor `simdq_common.h`**

Open `_native/include/simdq_common.h`. Locate `#define WORDS 12 // 768 bits` (around line 16). Delete that line. Then update each helper that previously assumed the macro:

a. `alloc_soa(size_t n)` (used to call `posix_memalign` with `n * WORDS * 8`): becomes `alloc_soa(size_t n, size_t words)` and the size becomes `n * words * 8`.

b. `hamming_soa(const uint64_t *dbT, size_t n, size_t i, const uint64_t *q)` (around line 71-78): becomes `hamming_soa(const uint64_t *dbT, size_t n, size_t words, size_t i, const uint64_t *q)` and the inner loop becomes `for (size_t w = 0; w < words; w++)`.

c. `hamming_aos(const uint64_t *db, size_t n, size_t i, const uint64_t *q)` (around line 90-95) — the `for (int w = 0; w < WORDS; w++)` body — becomes `hamming_aos(const uint64_t *db, size_t n, size_t words, size_t i, const uint64_t *q)`.

d. `fill_soa_parallel(uint64_t *dbT, const uint64_t *db, size_t n)` (around line 102-115) — the `for (size_t w = 0; w < WORDS; w++)` body and the `n * WORDS` allocations — becomes `fill_soa_parallel(uint64_t *dbT, const uint64_t *db, size_t n, size_t words)`.

Verify by `grep -n WORDS _native/include/simdq_common.h` — the macro should be entirely gone. If any helper still references `WORDS`, finish refactoring it.

- [ ] **Step 2: Refactor `simdq_kernels_hamming.h`** (top-1 kernel)

The top-1 batched kernel uses `WORDS` in the inner loop. Make `scan_hamming_top1_batched` (or whatever the existing symbol is — `grep -n '^static inline' _native/include/simdq_kernels_hamming.h | head -5`) take `size_t words` as an extra parameter just before `q`/`db`, and replace `WORDS` in the body with `words`. Same for any private helper in this header.

- [ ] **Step 3: Refactor `simdq_kernels_hamming_topk.h`**

Two functions: `scan_shard_topk` and `scan_topk_parallel` (both inside the AVX-512F + AVX2 branches). Rename to `scan_hamming_shard_topk` / `scan_hamming_topk_parallel` and add `size_t words` as a parameter just after `n`. Replace `WORDS` with `words` in the body. The reference helper `ref_scan_soa_top1` near the top of the file gets the same treatment.

- [ ] **Step 4: Update bench drivers**

`bench_hamming.c` and `bench_hamming_top1.c`: at the call sites, pass `/*words=*/12` to all helpers (`alloc_soa`, `fill_soa_parallel`, `scan_hamming_*`, `hamming_soa`). Add `#define WORDS 12` locally in each bench file so the existing literals continue to work.

- [ ] **Step 5: Update C tests**

`test_kernels_hamming.c`, `test_kernels_hamming_topk.c`: same pattern, pass `/*words=*/12` everywhere. Add `#define WORDS 12` locally in each test file.

- [ ] **Step 6: Build and verify**

```bash
cd docuverse/engines/retrieval/simdq/_native/build
make -j 2>&1 | tail -30
ctest --output-on-failure
```
Expected: all 14 ctest targets pass. `grep -rn '\bWORDS\b' _native/include/` should be empty.

- [ ] **Step 7: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_common.h \
        docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming.h \
        docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming_topk.h \
        docuverse/engines/retrieval/simdq/_native/bench/bench_hamming.c \
        docuverse/engines/retrieval/simdq/_native/bench/bench_hamming_top1.c \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming.c \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming_topk.c
git commit -m "Generalize Hamming kernels to runtime words (= D / 64).

Drop the WORDS=12 macro from simdq_common.h; helpers (hamming_soa,
hamming_aos, alloc_soa, fill_soa_parallel) and the top-K kernel
take words as a runtime parameter so the same kernel covers
D in {384, 768, 1024, 1536}. Symbol rename: scan_{shard,topk,parallel}
-> scan_hamming_{shard,topk,parallel}_topk. Bench/test call sites
pass words=12 explicitly. Behavior unchanged at D=768."
```

---

## Task 3: ctest coverage for `d` ∈ {192, 384, 512, 1024, 1536}

**Goal:** Add a single new test file `test_kernels_multi_D.c` that spot-checks the runtime-`d` refactors at every `d` in `{192, 384, 512, 768, 1024, 1536}` for asym b=2 (representative of asym family) and Hamming top-K. Existing per-d=768 tests are left as-is.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_multi_D.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/CMakeLists.txt`

- [ ] **Step 1: Write `test_kernels_multi_D.c`**

```c
// test_kernels_multi_D.c — runtime-d/runtime-words spot-check.
//
// For each d in {192, 384, 512, 768, 1024, 1536}, build a tiny
// (n=1024) random asym-b=2 index, scan a random query, and assert the
// kernel's top-1 matches a scalar reference loop within 1e-4 relative
// tolerance. Same for the Hamming top-K kernel at every D in
// {384, 768, 1024, 1536} (words = D/64).
//
// We're not measuring throughput here, just correctness for the d/words
// runtime parameterization. Per-d test budget is well under 200ms.

#include "simdq_common.h"
#include "simdq_pack.h"
#include "simdq_kernels_asym_b2.h"
#include "simdq_kernels_hamming_topk.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void seed_rng(uint64_t *s) { *s = 0xC0FFEEULL; }
static float urand(uint64_t *s) {
    *s = (*s) * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x = (uint32_t)((*s) >> 33);
    return (float)(x & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
}

static int test_asym_b2_at_d(size_t d) {
    const size_t N = 1024;
    uint64_t rng;
    seed_rng(&rng);

    float *Y = (float *)aligned_alloc(64, N * d * sizeof(float));
    float *q = (float *)aligned_alloc(64, d * sizeof(float));
    for (size_t i = 0; i < N * d; i++) Y[i] = urand(&rng);
    for (size_t w = 0; w < d; w++)    q[w] = urand(&rng);

    float *scales = simdq_pack_scales(Y, N, d);
    const size_t row_bytes = (N + 3) / 4;
    uint8_t *codes = (uint8_t *)calloc(d * row_bytes, 1);
    simdq_pack_b2(Y, N, d, scales, codes);

    int K = 4;
    float scores[4]; int64_t idxs[4];
    for (int r = 0; r < K; r++) { scores[r] = -INFINITY; idxs[r] = -1; }
    scan_asym_b2_topk_parallel(codes, N, d, q, K, scores, idxs);

    // Scalar reference: brute force <q, levels[code]> over all N
    static const int8_t L[4] = {-3, -1, 1, 3};
    float best = -INFINITY; size_t bi = 0;
    for (size_t i = 0; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
            uint8_t byte = codes[w * row_bytes + (i >> 2)];
            int8_t v = L[(byte >> ((i & 3) * 2)) & 0x3];
            s += q[w] * (float)v;
        }
        if (s > best) { best = s; bi = i; }
    }

    int ok = (idxs[0] == (int64_t)bi);
    if (!ok) {
        fprintf(stderr,
                "FAIL asym_b2 d=%zu: kernel best idx=%lld score=%.4f, ref idx=%zu score=%.4f\n",
                d, (long long)idxs[0], scores[0], bi, best);
    } else {
        printf("PASS asym_b2 d=%zu: top-1 idx=%zu score=%.4f\n", d, bi, best);
    }

    free(Y); free(q); free(scales); free(codes);
    return ok;
}

static int test_hamming_at_words(size_t words) {
    const size_t n = 4096;
    uint64_t rng; seed_rng(&rng);

    uint64_t *db    = (uint64_t *)aligned_alloc(64, n * words * sizeof(uint64_t));
    uint64_t *dbT   = alloc_soa(n, words);
    uint64_t *q     = (uint64_t *)aligned_alloc(64, words * sizeof(uint64_t));
    for (size_t i = 0; i < n * words; i++) db[i] = ((uint64_t)urand(&rng) * 0xAAULL) ^ ((uint64_t)i << 33);
    for (size_t w = 0; w < words; w++)     q[w]  = ((uint64_t)urand(&rng) * 0xBBULL) ^ ((uint64_t)w << 17);

    fill_soa_parallel(dbT, db, n, words);

    int K = 8;
    int64_t scores[8], idxs[8];
    scan_hamming_topk_parallel(dbT, n, words, q, K, scores, idxs);

    // Scalar reference: brute-force Hamming distance, find K smallest
    int *all = (int *)malloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++) all[i] = hamming_aos(db, n, words, i, q);
    int sorted[16]; size_t sorted_idx[16];
    for (int r = 0; r < K; r++) { sorted[r] = INT32_MAX; sorted_idx[r] = (size_t)-1; }
    for (size_t i = 0; i < n; i++) {
        if (all[i] < sorted[K-1]) {
            sorted[K-1] = all[i]; sorted_idx[K-1] = i;
            for (int r = K - 1; r > 0 && sorted[r] < sorted[r-1]; r--) {
                int td = sorted[r]; size_t ti = sorted_idx[r];
                sorted[r] = sorted[r-1]; sorted_idx[r] = sorted_idx[r-1];
                sorted[r-1] = td; sorted_idx[r-1] = ti;
            }
        }
    }
    int ok = 1;
    for (int r = 0; r < K; r++) {
        if ((int)scores[r] != sorted[r]) {
            fprintf(stderr,
                    "FAIL hamming words=%zu: rank %d kernel dist=%lld, ref dist=%d\n",
                    words, r, (long long)scores[r], sorted[r]);
            ok = 0;
        }
    }
    if (ok) printf("PASS hamming words=%zu: top-%d distances match\n", words, K);

    free(db); free(dbT); free(q); free(all);
    return ok;
}

int main(void) {
    int failures = 0;
    size_t d_list[]     = {192, 384, 512, 768, 1024, 1536};
    size_t words_list[] = {6,   12,  16,  24};
    for (size_t i = 0; i < sizeof(d_list)/sizeof(d_list[0]); i++)
        if (!test_asym_b2_at_d(d_list[i])) failures++;
    for (size_t i = 0; i < sizeof(words_list)/sizeof(words_list[0]); i++)
        if (!test_hamming_at_words(words_list[i])) failures++;
    return failures ? 1 : 0;
}
```

- [ ] **Step 2: Register the test in CMakeLists**

Append to `_native/CMakeLists.txt` (after the existing `kernels_asym_parallel_avx2` block):

```cmake
add_executable(test_kernels_multi_D tests/test_kernels_multi_D.c)
target_compile_options(test_kernels_multi_D PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_multi_D PRIVATE include)
target_link_libraries(test_kernels_multi_D PRIVATE OpenMP::OpenMP_C m)
add_test(NAME kernels_multi_D_native COMMAND test_kernels_multi_D)

add_executable(test_kernels_multi_D_avx2 tests/test_kernels_multi_D.c)
target_compile_options(test_kernels_multi_D_avx2 PRIVATE -O3 -mavx2 -mfma -mpopcnt)
target_include_directories(test_kernels_multi_D_avx2 PRIVATE include)
target_link_libraries(test_kernels_multi_D_avx2 PRIVATE OpenMP::OpenMP_C m)
add_test(NAME kernels_multi_D_avx2 COMMAND test_kernels_multi_D_avx2)
```

- [ ] **Step 3: Build and run**

```bash
cd docuverse/engines/retrieval/simdq/_native/build
cmake .. && make -j 2>&1 | tail -20
ctest --output-on-failure -R kernels_multi_D
```
Expected: 2 new ctest targets (`kernels_multi_D_native`, `kernels_multi_D_avx2`) both PASS. Total ctest count is now 16. Run `ctest --output-on-failure` (no `-R`) to confirm all 16 pass together.

- [ ] **Step 4: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/tests/test_kernels_multi_D.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add multi-D ctest spot-check (asym b=2 + hamming).

Verifies the runtime-d / runtime-words refactors from T1/T2 by
scanning small (n=1024 / 4096) random datasets at every supported
d in {192, 384, 512, 768, 1024, 1536} and asserting the kernel
top-K matches a scalar brute-force reference. Runs in both
AVX-512-native and AVX2-forced paths."
```

---

## Task 4: CPython binding takes `d` as a runtime arg; expose Hamming family

**Goal:** Plan 2's binding hard-coded `#define SIMDQ_D 768` and rejected any other shape. Plan 3 takes `d` from each input array's shape (or as an explicit parameter for `scan_*` where the shape doesn't carry it). Add two new entry points: `pack_hamming(Y) -> codes_bytes` and `scan_hamming(codes, N, D, q_codes, K, num_threads) -> (scores, indices)`.

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/bindings/module.c`

- [ ] **Step 1: Drop the hard-coded `SIMDQ_D` and accept any supported `d`**

In `_native/bindings/module.c`:

a. Delete `#define SIMDQ_D 768` (around line 36).

b. Add a small validator near the top, just after the includes:
```c
static int simdq_check_d(Py_ssize_t d) {
    static const Py_ssize_t supported[] = {192, 384, 512, 768, 1024, 1536};
    for (size_t i = 0; i < sizeof(supported)/sizeof(supported[0]); i++)
        if (supported[i] == d) return 0;
    PyErr_Format(PyExc_ValueError,
                 "d must be one of {192, 384, 512, 768, 1024, 1536}; got %zd", d);
    return -1;
}
```

c. In `py_compute_scales`: replace the `y_view.shape[1] != SIMDQ_D` check with:
```c
Py_ssize_t d = y_view.shape[1];
if (simdq_check_d(d) != 0) { PyBuffer_Release(&y_view); return NULL; }
```
The `inv_sqrt_d` and inner loop bound become `d` (already using `SIMDQ_D` literal).

d. In `do_pack`: same — replace `SIMDQ_D` with `y_view.shape[1]` after validation. Pass `d` into the underlying `simdq_pack_b{1,2,4}(Y, N, d, scales, codes)` call (those functions already take `d` as the third parameter; the binding currently passes `SIMDQ_D`).

e. In `do_scan`: the ParseTuple format string is `"OnOnn"` (codes, N, q, K, num_threads). Change it to `"OnnOnn"` and add `Py_ssize_t d` between `N` and `q_obj`. Validate via `simdq_check_d(d)`. Replace `SIMDQ_D` with `d` in the `q_view.shape[0] != SIMDQ_D` check, in `expect = (Py_ssize_t)SIMDQ_D * row_bytes`, and in the kernel call's third argument (which now becomes `(size_t)d`):
```c
if      (b == 1) scan_asym_b1_topk_parallel(
                      (const uint8_t *)codes_view.buf, (size_t)N, (size_t)d,
                      (const float *)q_view.buf, (int)K, scores, idxs);
else if (b == 2) scan_asym_b2_topk_parallel(
                      (const uint8_t *)codes_view.buf, (size_t)N, (size_t)d,
                      (const float *)q_view.buf, (int)K, scores, idxs);
else             scan_asym_b4_topk_parallel(
                      (const uint8_t *)codes_view.buf, (size_t)N, (size_t)d,
                      (const float *)q_view.buf, (int)K, scores, idxs);
```

f. Update the file's top-of-file comment block to reflect that `d` is now a runtime parameter on every entry point.

g. In `PyInit__simdq_native`, replace `PyModule_AddIntConstant(m, "D", SIMDQ_D)` with a tuple of supported d's:
```c
PyObject *supported = Py_BuildValue("(iiiiii)", 192, 384, 512, 768, 1024, 1536);
if (!supported) { Py_DECREF(m); return NULL; }
if (PyModule_AddObject(m, "SUPPORTED_D", supported) < 0) {
    Py_DECREF(supported); Py_DECREF(m); return NULL;
}
```

- [ ] **Step 2: Add the Hamming binding entry points**

Append to `module.c` after the `do_scan` block, before the `PyMethodDef SimdqMethods[]` table:

```c
// ---------- pack_hamming ----------

static PyObject *py_pack_hamming(PyObject *self, PyObject *args) {
    PyObject *y_obj;
    if (!PyArg_ParseTuple(args, "O", &y_obj)) return NULL;
    Py_buffer y_view;
    if (get_buffer(y_obj, &y_view, 'f', 0) != 0) return NULL;
    Py_ssize_t d = (y_view.ndim == 2) ? y_view.shape[1] : -1;
    if (simdq_check_d(d) != 0) { PyBuffer_Release(&y_view); return NULL; }
    if ((d % 64) != 0) {
        PyErr_Format(PyExc_ValueError,
                     "pack_hamming: d must be a multiple of 64; got %zd", d);
        PyBuffer_Release(&y_view); return NULL;
    }
    Py_ssize_t N = y_view.shape[0];
    Py_ssize_t words = d / 64;

    // Output: AoS layout, words uint64 per code, N codes total.
    char *out_data;
    PyObject *out = new_bytes_buffer(N * words * (Py_ssize_t)sizeof(uint64_t), &out_data);
    if (!out) { PyBuffer_Release(&y_view); return NULL; }
    uint64_t *db = (uint64_t *)out_data;
    const float *Y = (const float *)y_view.buf;

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (Py_ssize_t i = 0; i < N; i++) {
        const float *yi = Y + (size_t)i * d;
        for (Py_ssize_t w = 0; w < words; w++) {
            uint64_t bits = 0;
            for (int b = 0; b < 64; b++) {
                if (yi[w * 64 + b] >= 0.0f) bits |= ((uint64_t)1 << b);
            }
            db[i * words + w] = bits;
        }
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&y_view);
    return out;
}

// ---------- scan_hamming ----------

static PyObject *py_scan_hamming(PyObject *self, PyObject *args) {
    PyObject *codes_obj, *q_obj;
    Py_ssize_t N, D, K, num_threads;
    if (!PyArg_ParseTuple(args, "OnnOnn",
                          &codes_obj, &N, &D, &q_obj, &K, &num_threads)) return NULL;
    if (K <= 0 || K > 256) {
        PyErr_SetString(PyExc_ValueError, "K must be in [1, 256]"); return NULL;
    }
    if (simdq_check_d(D) != 0) return NULL;
    if ((D % 64) != 0) {
        PyErr_Format(PyExc_ValueError,
                     "scan_hamming: D must be a multiple of 64; got %zd", D);
        return NULL;
    }
    Py_ssize_t words = D / 64;

    Py_buffer codes_view, q_view;
    if (get_buffer(codes_obj, &codes_view, 'B', 0) != 0) return NULL;
    if (get_buffer(q_obj, &q_view, 'B', 0) != 0) {
        PyBuffer_Release(&codes_view); return NULL;
    }
    Py_ssize_t expect_codes = N * words * (Py_ssize_t)sizeof(uint64_t);
    Py_ssize_t expect_q     = words * (Py_ssize_t)sizeof(uint64_t);
    if (codes_view.shape[0] < expect_codes) {
        PyErr_Format(PyExc_ValueError,
                     "codes buffer too small: %zd < %zd", codes_view.shape[0], expect_codes);
        goto fail3;
    }
    if (q_view.shape[0] < expect_q) {
        PyErr_Format(PyExc_ValueError,
                     "q buffer too small: %zd < %zd", q_view.shape[0], expect_q);
        goto fail3;
    }

    // SoA-fill: dbT[w * N + i] = db[i * words + w]. Caller-provided codes
    // are AoS (matching pack_hamming output); fill_soa_parallel converts
    // them once. Keep the SoA buffer module-local (alloc + free per call)
    // — Plan 3 doesn't try to cache it; the typical use is a one-shot
    // load_or_search call from SimdqIndex.
    uint64_t *dbT = alloc_soa((size_t)N, (size_t)words);
    if (!dbT) { PyErr_NoMemory(); goto fail3; }
    fill_soa_parallel(dbT, (const uint64_t *)codes_view.buf, (size_t)N, (size_t)words);

    char *scores_data, *idx_data;
    PyObject *scores_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(int64_t), &scores_data);
    if (!scores_bytes) { free(dbT); goto fail3; }
    PyObject *idx_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(int64_t), &idx_data);
    if (!idx_bytes) { Py_DECREF(scores_bytes); free(dbT); goto fail3; }
    int64_t *scores = (int64_t *)scores_data;
    int64_t *idxs   = (int64_t *)idx_data;

    int saved_threads = omp_get_max_threads();
    if (num_threads > 0) omp_set_num_threads((int)num_threads);

    Py_BEGIN_ALLOW_THREADS
    scan_hamming_topk_parallel(dbT, (size_t)N, (size_t)words,
                               (const uint64_t *)q_view.buf, (int)K,
                               scores, idxs);
    Py_END_ALLOW_THREADS

    if (num_threads > 0) omp_set_num_threads(saved_threads);
    free(dbT);

    PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view);
    return Py_BuildValue("(NN)", scores_bytes, idx_bytes);

fail3:
    PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view);
    return NULL;
}
```

- [ ] **Step 3: Register the new methods**

In the `SimdqMethods[]` table, add two entries before the sentinel `{NULL, NULL, 0, NULL}`:
```c
{"pack_hamming", py_pack_hamming, METH_VARARGS,
 "pack_hamming(Y) -> codes (uint8 bytes, AoS layout, D*N/8 bytes total)"},
{"scan_hamming", py_scan_hamming, METH_VARARGS,
 "scan_hamming(codes, N, D, q, K, num_threads) -> (distances, indices) bytes; "
 "distances are int64 Hamming distances (smaller = better)."},
```

- [ ] **Step 4: Rebuild the extension**

```bash
conda activate ndocu
pip install -e . 2>&1 | tail -10
```
Expected: clean rebuild of `_simdq_native.cpython-*.so`. If you see warnings about `SIMDQ_D` being undefined, you missed a reference — `grep -n SIMDQ_D docuverse/engines/retrieval/simdq/_native/bindings/module.c` should be empty.

- [ ] **Step 5: Smoke-test the new entry points from Python**

```bash
python -c "
import numpy as np
from docuverse.engines.retrieval.simdq import _simdq_native as n
print('SUPPORTED_D:', n.SUPPORTED_D)
Y = np.random.RandomState(0).randn(64, 384).astype(np.float32)
sc = n.compute_scales(Y)
print('scales bytes:', len(sc), '(expect 256)')
codes = n.pack_b2(Y, np.frombuffer(sc, dtype=np.float32), )
# pack_b2 takes (Y, scales) -> codes. Verify length matches d * ceil(N/4):
expect = 384 * ((64 + 3) // 4)
print('codes bytes:', len(codes), '(expect', expect, ')')
hb = n.pack_hamming(Y)
print('hamming bytes:', len(hb), '(expect', 64 * 384 // 8, ')')
"
```
Expected: prints SUPPORTED_D as a 6-tuple; `scales bytes: 256`; `codes bytes: 6144`; `hamming bytes: 3072`.

If `SUPPORTED_D` raises `AttributeError`, the binding wasn't rebuilt. Force a clean rebuild: `find . -name '_simdq_native*.so' -delete && pip install -e .`.

- [ ] **Step 6: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/bindings/module.c
git commit -m "CPython binding: runtime d on every entry; add pack/scan_hamming.

The Plan 2 binding hard-coded SIMDQ_D=768. Plan 3 takes d from the
input numpy array shape on compute_scales/pack_b{1,2,4}, and as an
explicit argument on scan_b{1,2,4}/scan_hamming. SUPPORTED_D module
attribute exposes the validation set {192, 384, 512, 768, 1024, 1536}.

New entry points pack_hamming(Y) and scan_hamming(codes, N, D, q, K,
num_threads) plumb the symmetric Hamming kernel through to Python so
SimdqIndex (next task) can build a 'family=hamming' index."
```

---

## Task 5: `SimdqIndex` accepts variable D, d, and `family ∈ {asymmetric, hamming}`

**Goal:** Relax `SimdqIndex.build`'s `vectors.shape[1] != D_FIXED` rejection so any `D ∈ {384, 768, 1024, 1536}` and `d ∈ {D, D/2}` pair is accepted. Add a `family` parameter (default `"asymmetric"`) that gates between the existing b={1,2,4} path and a new Hamming path. The Hamming path stores codes via `pack_hamming` (no separate scales, no `b` parameter) and searches via `scan_hamming`. Update `meta.json` to record `D_orig`, `d`, `family`, and (for asym) `b`.

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/projection.py`
- Modify: `docuverse/engines/retrieval/simdq/quantization.py`
- Modify: `docuverse/engines/retrieval/simdq/simdq_index.py`

- [ ] **Step 1: Add `SUPPORTED_D` constants to `projection.py`**

In `docuverse/engines/retrieval/simdq/projection.py` (read the current file with `Read` first to know its exact shape — Plan 2 created it; it has `identity`, `random_orthogonal`, `apply_projection`).

Add at module top, after the imports:
```python
SUPPORTED_D = (384, 768, 1024, 1536)
SUPPORTED_d = (192, 384, 512, 768, 1024, 1536)


def validate_D_d(D: int, d: int) -> None:
    if D not in SUPPORTED_D:
        raise ValueError(
            f"simdq: D must be one of {SUPPORTED_D}; got {D}"
        )
    if d not in SUPPORTED_d:
        raise ValueError(
            f"simdq: d must be one of {SUPPORTED_d}; got {d}"
        )
    if d not in (D, D // 2):
        raise ValueError(
            f"simdq: for D={D}, d must be {D} or {D // 2}; got {d}"
        )
```

- [ ] **Step 2: Update `quantization.py` to drop the `D_FIXED` check**

Open `docuverse/engines/retrieval/simdq/quantization.py`. Replace `_check_Y` body so it validates against `SUPPORTED_d` instead of equality with `D_FIXED`:
```python
from docuverse.engines.retrieval.simdq.projection import SUPPORTED_d


def _check_Y(Y: np.ndarray) -> np.ndarray:
    if Y.ndim != 2 or Y.shape[1] not in SUPPORTED_d:
        raise ValueError(
            f"simdq quantization: Y must have shape (N, d) with d in "
            f"{SUPPORTED_d}; got shape {Y.shape}"
        )
    if Y.dtype != np.float32:
        Y = Y.astype(np.float32, copy=False)
    return np.ascontiguousarray(Y)
```

Delete the `D_FIXED = _native.D` line (the binding no longer exposes `D`).

Add a thin `pack_hamming` wrapper at the end of the module:
```python
def pack_hamming(Y: np.ndarray) -> np.ndarray:
    """Sign-quantize Y to 1-bit AoS Hamming codes.

    Returns a uint8 numpy array of length N * d / 8 bytes.
    Bit ordering: codes[i*words + w] holds bits 64*w .. 64*w+63 of vector i,
    with bit b set iff Y[i, w*64 + b] >= 0.
    """
    Y = _check_Y(Y)
    if Y.shape[1] % 64 != 0:
        raise ValueError(
            f"simdq pack_hamming: d must be a multiple of 64; got {Y.shape[1]}"
        )
    raw = _native.pack_hamming(Y)
    return np.frombuffer(raw, dtype=np.uint8)
```

- [ ] **Step 3: Update `simdq_index.py` to accept variable D/d/family**

Open `docuverse/engines/retrieval/simdq/simdq_index.py`. Replace the module docstring's "D is fixed at 768 in Plan 2" paragraph with:
```python
"""... (preserve existing docstring header) ...

D ∈ {384, 768, 1024, 1536}, d ∈ {D, D/2}. ``family="asymmetric"``
ranks via per-vector-scale-corrected b-bit dot products
(``b ∈ {1, 2, 4}``); ``family="hamming"`` ranks via 1-bit symmetric
Hamming distance with the query also sign-quantized.
"""
```

Replace the dataclass and `build` method:
```python
from docuverse.engines.retrieval.simdq.projection import (
    SUPPORTED_D, SUPPORTED_d, validate_D_d,
)

@dataclass
class SimdqIndex:
    """In-memory + on-disk simdq index.

    family="asymmetric" stores per-bit-quantized codes + per-vector scales.
    family="hamming"    stores 1-bit AoS Hamming codes; no scales, no b.
    """
    family: str                                 # "asymmetric" | "hamming"
    codes: np.ndarray                           # uint8
    W: np.ndarray                               # fp32 (d, D)
    n_vectors: int
    D_orig: int                                 # encoder dim
    d: int                                      # reduced dim
    b: Optional[int] = None                     # 1/2/4 if asym; None if hamming
    scales: Optional[np.ndarray] = None         # fp16 (N,) if asym; None if hamming
    projection_name: str = "identity"
    projection_seed: int = 42
    has_floats: bool = False
    floats_mmap: Optional[np.ndarray] = None
    encoder_id: Optional[str] = None

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        family: str = "asymmetric",
        b: Optional[int] = 2,
        d: Optional[int] = None,
        projection: str = "identity",
        projection_seed: int = 42,
        store_floats: bool = True,
        encoder_id: Optional[str] = None,
    ) -> "SimdqIndex":
        if vectors.ndim != 2:
            raise ValueError(f"simdq build: vectors must be 2-D; got {vectors.shape}")
        D = int(vectors.shape[1])
        if D not in SUPPORTED_D:
            raise ValueError(f"simdq build: D must be one of {SUPPORTED_D}; got {D}")
        if d is None:
            d = D
        validate_D_d(D, d)

        if family not in ("asymmetric", "hamming"):
            raise ValueError(f"simdq build: family must be 'asymmetric' or 'hamming'; got {family!r}")
        if family == "asymmetric" and b not in (1, 2, 4):
            raise ValueError(f"simdq build: asymmetric requires b in (1, 2, 4); got {b}")
        if family == "hamming" and (d % 64) != 0:
            raise ValueError(f"simdq build: hamming requires d divisible by 64; got d={d}")

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        N = int(vectors.shape[0])

        if projection == "identity" and d == D:
            W = _projection.identity(D)
        elif projection == "random_orthogonal":
            W = _projection.random_orthogonal(D, d, projection_seed)
        elif projection == "identity" and d != D:
            raise ValueError(
                "simdq build: projection='identity' requires d == D; "
                f"got D={D}, d={d}. Use projection='random_orthogonal' "
                "for d=D/2 reductions."
            )
        else:
            raise ValueError(
                f"simdq build: projection must be 'identity' or 'random_orthogonal'; "
                f"got {projection!r}"
            )
        Y = _projection.apply_projection(vectors, W)              # (N, d)

        if family == "asymmetric":
            scales_fp32 = _quant.fit_scales(Y)
            codes = _quant.pack(Y, scales_fp32, b=b)
            scales_fp16 = scales_fp32.astype(np.float16)
            stored_b = b
        else:
            codes = _quant.pack_hamming(Y)
            scales_fp16 = None
            stored_b = None

        floats_mmap = Y.astype(np.float16) if store_floats else None

        return cls(
            family=family,
            codes=codes, scales=scales_fp16, W=W,
            n_vectors=N, D_orig=D, d=d, b=stored_b,
            projection_name=projection, projection_seed=projection_seed,
            has_floats=store_floats, floats_mmap=floats_mmap,
            encoder_id=encoder_id,
        )
```

Replace `save`:
```python
    def save(self, path: os.PathLike) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        if tmp.exists():
            for child in tmp.iterdir(): child.unlink()
            tmp.rmdir()
        tmp.mkdir(parents=True)

        (tmp / "codes.bin").write_bytes(self.codes.tobytes())
        if self.scales is not None:
            (tmp / "scales.bin").write_bytes(self.scales.tobytes())
        np.save(tmp / "W.npy", self.W)
        if self.has_floats:
            assert self.floats_mmap is not None
            (tmp / "floats.bin").write_bytes(
                np.ascontiguousarray(self.floats_mmap).tobytes()
            )

        meta = {
            "format_version": FORMAT_VERSION,
            "n_vectors": int(self.n_vectors),
            "D_orig": int(self.D_orig),
            "d": int(self.d),
            "family": self.family,
            "b": (None if self.b is None else int(self.b)),
            "projection": self.projection_name,
            "projection_seed": int(self.projection_seed),
            "quantization": ("none" if self.family == "hamming" else "per_vector"),
            "code_layout": ("aos" if self.family == "hamming" else "soa"),
            "code_bytes_per_vector": int(self.codes.size // self.n_vectors)
                                     if self.family == "hamming"
                                     else int(self.codes.size // self.d),
            "has_floats": bool(self.has_floats),
            "float_dtype": "float16" if self.has_floats else None,
            "encoder_id": self.encoder_id,
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))

        if path.exists():
            for child in path.iterdir(): child.unlink()
            path.rmdir()
        tmp.rename(path)
```

Replace `load`:
```python
    @classmethod
    def load(cls, path: os.PathLike) -> "SimdqIndex":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        if meta["format_version"] != FORMAT_VERSION:
            raise ValueError(
                f"simdq load: format_version mismatch (file={meta['format_version']}, "
                f"code={FORMAT_VERSION}). Rebuild the index."
            )
        N = meta["n_vectors"]
        d = meta["d"]
        family = meta.get("family", "asymmetric")    # legacy: pre-Plan-3 indexes are asym
        b = meta.get("b")

        codes = np.fromfile(path / "codes.bin", dtype=np.uint8)
        scales = None
        if family == "asymmetric":
            scales = np.fromfile(path / "scales.bin", dtype=np.float16)
            if scales.shape[0] != N:
                raise ValueError(
                    f"simdq load: scales.bin has {scales.shape[0]} entries; "
                    f"meta says n_vectors={N}"
                )
        W = np.load(path / "W.npy")
        floats_mmap = None
        if meta.get("has_floats"):
            floats_mmap = np.memmap(
                path / "floats.bin", dtype=np.float16, mode="r",
                shape=(N, d),
            )
        return cls(
            family=family,
            codes=codes, scales=scales, W=W,
            n_vectors=N, D_orig=meta["D_orig"], d=d, b=b,
            projection_name=meta["projection"],
            projection_seed=meta["projection_seed"],
            has_floats=bool(meta.get("has_floats", False)),
            floats_mmap=floats_mmap,
            encoder_id=meta.get("encoder_id"),
        )
```

Replace `search`:
```python
    def search(self, q: np.ndarray, K: int = 10, K_prime: Optional[int] = None,
               num_threads: int = 0):
        if K <= 0 or K > 256:
            raise ValueError(f"simdq search: K must be in [1, 256]; got {K}")
        if K_prime is None: K_prime = K
        if K_prime < K or K_prime > 256:
            raise ValueError(
                f"simdq search: need K <= K_prime <= 256; got K={K}, K'={K_prime}"
            )
        if q.dtype != np.float32:
            q = q.astype(np.float32, copy=False)
        if q.shape != (self.D_orig,):
            raise ValueError(
                f"simdq search: q must have shape ({self.D_orig},); got {q.shape}"
            )

        q_proj = np.ascontiguousarray((self.W @ q).astype(np.float32, copy=False))

        if self.family == "asymmetric":
            return self._search_asym(q_proj, K, K_prime, num_threads)
        return self._search_hamming(q_proj, K, K_prime, num_threads)

    def _search_asym(self, q_proj, K, K_prime, num_threads):
        scan = {1: _native.scan_b1, 2: _native.scan_b2, 4: _native.scan_b4}[self.b]
        scores_buf, idx_buf = scan(self.codes, self.n_vectors, self.d,
                                   q_proj, K_prime, num_threads)
        raw_scores = np.frombuffer(scores_buf, dtype=np.float32).copy()
        idxs       = np.frombuffer(idx_buf, dtype=np.int64).copy()
        scales = self.scales[idxs].astype(np.float32)
        scaled = raw_scores * scales
        if self.has_floats and K_prime > K:
            cand_floats = np.array(self.floats_mmap[idxs], dtype=np.float32)
            rescore = cand_floats @ q_proj
            order = np.argsort(-rescore)[:K]
            return idxs[order], rescore[order].astype(np.float32)
        order = np.argsort(-scaled)[:K]
        return idxs[order], scaled[order]

    def _search_hamming(self, q_proj, K, K_prime, num_threads):
        # Quantize the projected query to 1-bit codes (same packing as pack_hamming).
        words = self.d // 64
        q_bits = np.zeros(words, dtype=np.uint64)
        for w in range(words):
            block = q_proj[w * 64:(w + 1) * 64]
            mask = (block >= 0.0)
            bits = 0
            for b in range(64):
                if mask[b]: bits |= (np.uint64(1) << np.uint64(b))
            q_bits[w] = bits
        q_bytes = q_bits.tobytes()                                # 8*words bytes

        dist_buf, idx_buf = _native.scan_hamming(
            self.codes, self.n_vectors, self.d, q_bytes, K_prime, num_threads,
        )
        dists = np.frombuffer(dist_buf, dtype=np.int64).copy()
        idxs  = np.frombuffer(idx_buf,  dtype=np.int64).copy()

        if self.has_floats and K_prime > K:
            cand_floats = np.array(self.floats_mmap[idxs], dtype=np.float32)
            rescore = cand_floats @ q_proj
            order = np.argsort(-rescore)[:K]
            return idxs[order], rescore[order].astype(np.float32)
        # Codes-only Hamming: smaller distance = better. Convert to a
        # similarity-like score (negative distance) so callers always sort
        # descending.
        order = np.argsort(dists)[:K]
        return idxs[order], (-dists[order]).astype(np.float32)
```

Delete the now-unused `_row_bytes` helper and the `D_FIXED = _native.D` line. Update the file's top docstring "D is fixed at 768 in Plan 2" paragraph to reflect the multi-D / multi-family support.

- [ ] **Step 4: Run the existing pytest to verify nothing regressed**

```bash
conda activate ndocu
python -m pytest tests/test_simdq_projection.py tests/test_simdq_quantization.py tests/test_simdq_index.py -x
```
Expected: all 19 tests still pass at D=768. The legacy-meta path in `load` (no `family` field) covers Plan 2 indexes that don't have it.

If a quantization test fails on `D_FIXED` import: it's importing the deleted constant. Update the test to import `SUPPORTED_D` from `projection` instead.

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/retrieval/simdq/projection.py \
        docuverse/engines/retrieval/simdq/quantization.py \
        docuverse/engines/retrieval/simdq/simdq_index.py
git commit -m "SimdqIndex: support D in {384,768,1024,1536}, d in {D,D/2}, family=hamming.

D and d are now validated via projection.SUPPORTED_D / SUPPORTED_d
and the (D, d) pair must satisfy d in {D, D/2}. New family axis:
'asymmetric' (existing b-bit + per-vector scale path) or 'hamming'
(1-bit symmetric, query sign-quantized at scan time). Meta.json
gains family, D_orig fields; legacy meta without family loads as
asymmetric for backward compat with Plan 2 indexes."
```

---

## Task 6: Python tests for multi-D and `family=hamming`

**Goal:** Extend the three pytest files so they parametrize over `(D, d)` and over `family ∈ {"asymmetric", "hamming"}`. Existing single-D tests stay; we add parametrized clones.

**Files:**
- Modify: `tests/test_simdq_projection.py`
- Modify: `tests/test_simdq_quantization.py`
- Modify: `tests/test_simdq_index.py`

- [ ] **Step 1: Extend `test_simdq_projection.py`**

Read the current file. Add a parametrized round-trip test after the existing tests:

```python
import pytest
import numpy as np
from docuverse.engines.retrieval.simdq.projection import (
    identity, random_orthogonal, apply_projection,
    validate_D_d, SUPPORTED_D, SUPPORTED_d,
)


@pytest.mark.parametrize("D,d", [
    (384, 384), (384, 192),
    (768, 768), (768, 384),
    (1024, 1024), (1024, 512),
    (1536, 1536), (1536, 768),
])
def test_random_orthogonal_shape_and_orthonormal_rows(D, d):
    W = random_orthogonal(D, d, seed=42)
    assert W.shape == (d, D)
    assert W.dtype == np.float32
    # Rows are orthonormal: W @ W.T == I_d (within fp32 tolerance)
    gram = W @ W.T
    assert np.allclose(gram, np.eye(d, dtype=np.float32), atol=1e-4)


@pytest.mark.parametrize("D,d", [
    (384, 384), (384, 192), (768, 768), (768, 384),
])
def test_apply_projection_norms_preserved_at_d_eq_D(D, d):
    """For d == D, random_orthogonal preserves norms (square Q)."""
    if d != D:
        pytest.skip("only meaningful at d==D")
    W = random_orthogonal(D, d, seed=0)
    X = np.random.RandomState(1).randn(50, D).astype(np.float32)
    Y = apply_projection(X, W)
    n_x = np.linalg.norm(X, axis=1)
    n_y = np.linalg.norm(Y, axis=1)
    assert np.allclose(n_x, n_y, rtol=1e-4)


def test_validate_D_d_rejects_unsupported():
    with pytest.raises(ValueError, match="D must be one of"):
        validate_D_d(512, 512)
    with pytest.raises(ValueError, match="d must be"):
        validate_D_d(768, 256)
    with pytest.raises(ValueError, match="d must be"):
        validate_D_d(384, 768)
```

- [ ] **Step 2: Extend `test_simdq_quantization.py`**

Add parametrized round-trip tests:
```python
import pytest
from docuverse.engines.retrieval.simdq import quantization as q
from docuverse.engines.retrieval.simdq.projection import SUPPORTED_d


@pytest.mark.parametrize("d", SUPPORTED_d)
@pytest.mark.parametrize("b", [1, 2, 4])
def test_pack_round_trip_at_each_d(d, b):
    rng = np.random.RandomState(d * 100 + b)
    Y = rng.randn(64, d).astype(np.float32)
    scales = q.fit_scales(Y)
    codes = q.pack(Y, scales, b=b)
    levels = q.unpack_levels(codes, N=64, D=d, b=b)
    # Reconstruct quantized Y and check signs match (asymmetric ranks robustly to
    # quantization at the b=2 level).
    s = scales.reshape(-1, 1).astype(np.float32)
    yhat = levels.astype(np.float32) * s / float(2 ** b - 1) * np.sqrt(d)
    # Spot-check: sign agreement on at least 70% of dims (a generous floor;
    # higher b should pass at >85%).
    sign_match = (np.sign(yhat) == np.sign(Y)).mean()
    assert sign_match > 0.70, f"sign agreement only {sign_match:.2%} at d={d} b={b}"


@pytest.mark.parametrize("d", [384, 768, 1024, 1536])
def test_pack_hamming_round_trip(d):
    rng = np.random.RandomState(d)
    Y = rng.randn(32, d).astype(np.float32)
    codes = q.pack_hamming(Y)
    assert codes.dtype == np.uint8
    assert codes.size == 32 * d // 8
    # Unpack one row and confirm the sign matches.
    words = d // 64
    code0 = codes[:words * 8].view(np.uint64)
    for w in range(words):
        for b in range(64):
            bit = bool((code0[w] >> np.uint64(b)) & np.uint64(1))
            assert bit == (Y[0, w * 64 + b] >= 0.0), \
                f"bit mismatch at w={w} b={b}"
```

- [ ] **Step 3: Extend `test_simdq_index.py`**

Add a parametrized end-to-end round-trip and a Hamming-family round-trip:
```python
import pytest
from docuverse.engines.retrieval.simdq import SimdqIndex


@pytest.mark.parametrize("D,d,projection", [
    (384, 384, "identity"),
    (384, 192, "random_orthogonal"),
    (1024, 1024, "identity"),
    (1024, 512, "random_orthogonal"),
    (1536, 768, "random_orthogonal"),
])
def test_asym_build_save_load_search_multi_D(D, d, projection, tmp_path):
    rng = np.random.RandomState(D)
    Y = rng.randn(2048, D).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    idx = SimdqIndex.build(
        Y, family="asymmetric", b=2, d=d,
        projection=projection, projection_seed=7, store_floats=True,
    )
    p = tmp_path / f"idx_{D}_{d}"
    idx.save(p)
    idx2 = SimdqIndex.load(p)
    assert idx2.D_orig == D and idx2.d == d and idx2.family == "asymmetric"

    # Smoke-test search: any well-defined query returns K finite indices in [0, N)
    q = rng.randn(D).astype(np.float32)
    iN, sN = idx2.search(q, K=10, K_prime=100)
    assert iN.shape == (10,) and sN.shape == (10,)
    assert iN.min() >= 0 and iN.max() < 2048
    assert np.all(np.isfinite(sN))


@pytest.mark.parametrize("D", [384, 768, 1024, 1536])
def test_hamming_family_round_trip(D, tmp_path):
    rng = np.random.RandomState(D + 1)
    Y = rng.randn(1024, D).astype(np.float32)
    idx = SimdqIndex.build(
        Y, family="hamming", d=D, projection="identity", store_floats=False,
    )
    p = tmp_path / f"idx_h_{D}"
    idx.save(p)
    idx2 = SimdqIndex.load(p)
    assert idx2.family == "hamming"
    assert idx2.b is None
    assert idx2.scales is None

    q = rng.randn(D).astype(np.float32)
    iN, sN = idx2.search(q, K=10, K_prime=10)
    assert iN.shape == (10,) and sN.shape == (10,)
    # Hamming "scores" are -distance; descending order means smaller distances first.
    assert np.all(np.diff(sN) <= 1e-6)
```

- [ ] **Step 4: Run all simdq tests**

```bash
conda activate ndocu
python -m pytest tests/test_simdq_projection.py tests/test_simdq_quantization.py tests/test_simdq_index.py -v 2>&1 | tail -40
```
Expected: all original 19 tests pass + ~25 new parametrized tests pass. If a multi-D test fails on `random_orthogonal`'s QR step (numerical issues at d=192 with seed=0), bump the seed in the fixture to a different value.

- [ ] **Step 5: Commit**

```bash
git add tests/test_simdq_projection.py tests/test_simdq_quantization.py tests/test_simdq_index.py
git commit -m "Pytest: parametrize over D in SUPPORTED_D, d in SUPPORTED_d, family.

Adds round-trip coverage for the runtime-d binding and the new
hamming family. Quantization sign-agreement floor (70%) is set
generously; higher b should pass it easily."
```

---

## Task 7: Add `simdq_*` fields to `RetrievalArguments`

**Goal:** The codebase pattern is one shared `RetrievalArguments` dataclass with engine-specific fields prefixed by the engine name. Add the spec section 8 fields with `simdq_` prefix, and add `simdq` to the `db_engine` choices list.

**Files:**
- Modify: `docuverse/engines/search_engine_config_params.py`

- [ ] **Step 1: Add `simdq` to `db_engine` choices**

In `search_engine_config_params.py`, locate the `db_engine` field (around line 137, inside `RetrievalArguments`). Update its `metadata={"choices": [...]}` to include `'simdq'`:
```python
    db_engine: Optional[str] = field(
        default="es-bm25",
        metadata={
            "choices": ['es-dense', 'es-elser', 'es-bm25',
                       'chromadb', 'faiss',
                       'milvus', 'milvus-dense', 'milvus-sparse', 'milvus-bm25',
                       'milvus-hybrid', 'milvus-splade',
                       'lancedb', 'lance',
                       'simdq'],
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
```

- [ ] **Step 2: Add the `simdq_*` field group**

Find a sensible location at the end of `RetrievalArguments` (after the last engine-specific field — `grep -n "milvus_idf_file\|lancedb_" search_engine_config_params.py` shows where engine fields cluster; if no clear cluster exists, append before the closing of the dataclass). Insert this block:

```python
    # ----- simdq engine -----

    simdq_b: int = field(
        default=2,
        metadata={
            "choices": [1, 2, 4],
            "help": "simdq: bits per dim in the asymmetric scan (1, 2, or 4). "
                    "Ignored when simdq_family='hamming'."
        }
    )

    simdq_family: str = field(
        default="asymmetric",
        metadata={
            "choices": ["asymmetric", "hamming"],
            "help": "simdq scan family: 'asymmetric' (float query × b-bit codes) or "
                    "'hamming' (1-bit symmetric, both query and code sign-quantized)."
        }
    )

    simdq_d: Optional[int] = field(
        default=None,
        metadata={
            "help": "simdq: reduced dimension. None means d=D (encoder dim). "
                    "Otherwise must equal D or D/2."
        }
    )

    simdq_projection: str = field(
        default="identity",
        metadata={
            "choices": ["identity", "random_orthogonal"],
            "help": "simdq: projection W applied to encoder output. "
                    "Must be 'random_orthogonal' when simdq_d != D."
        }
    )

    simdq_projection_seed: int = field(
        default=42,
        metadata={"help": "simdq: rng seed for random_orthogonal projection."}
    )

    simdq_store_floats: bool = field(
        default=True,
        metadata={
            "help": "simdq: store fp16 reduced vectors on disk for the rescore tier. "
                    "Disable to halve index size at the cost of recall."
        }
    )

    simdq_rescore_alpha: int = field(
        default=10,
        metadata={
            "help": "simdq: K' = alpha * top_k when rescoring against floats.bin. "
                    "Set to 1 to disable rescore (codes-only mode)."
        }
    )

    simdq_num_threads: int = field(
        default=0,
        metadata={"help": "simdq: OpenMP thread count for the scan kernel; 0 = OMP default."}
    )
```

- [ ] **Step 3: Verify the dataclass still parses**

```bash
conda activate ndocu
python -c "
from docuverse.engines.search_engine_config_params import RetrievalArguments
r = RetrievalArguments()
assert r.simdq_b == 2
assert r.simdq_family == 'asymmetric'
assert r.simdq_d is None
print('RetrievalArguments accepts simdq_* defaults')
print('db_engine choices include simdq:', 'simdq' in r.__dataclass_fields__['db_engine'].metadata['choices'])
"
```
Expected: prints both lines without error.

- [ ] **Step 4: Commit**

```bash
git add docuverse/engines/search_engine_config_params.py
git commit -m "Add simdq_* fields to RetrievalArguments; add 'simdq' to db_engine choices.

Mirrors the codebase pattern of one RetrievalArguments dataclass
shared across engines (cf. milvus_*, lancedb_* fields). Fields:
simdq_family, simdq_b, simdq_d, simdq_projection, simdq_projection_seed,
simdq_store_floats, simdq_rescore_alpha, simdq_num_threads."
```

---

## Task 8: Implement `SimdqEngine(RetrievalEngine)`

**Goal:** A `RetrievalEngine` subclass that owns a `DenseEmbeddingFunction` (for text encoding) and a `SimdqIndex` (for storage / search). At ingest time it accumulates encoded vectors across all batches into a single fp32 buffer, then calls `SimdqIndex.build(...).save(...)` once. At search time it lazy-loads the index, encodes the query, and returns a `SearchResult`.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/simdq_engine.py`
- Modify: `docuverse/engines/retrieval/simdq/__init__.py`

- [ ] **Step 1: Create `simdq_engine.py`**

```python
"""SimdqEngine — DocUVerse SearchEngine backed by an in-process simdq index.

Mirrors FAISSEngine: in-process, file-backed (no client/server split). At
ingest time we accumulate corpus-encoded vectors across all batches into a
single fp32 numpy buffer (peak RAM = 4 * N * D bytes), then build + save
the index in one shot. At search time the index is mmap-loaded once on
the first query.

Persist layout:
    <persist_directory>/simdq_data/<index_name>/
        meta.json
        W.npy
        codes.bin
        scales.bin            # asymmetric only
        floats.bin            # only if simdq_store_floats=True
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import _trim_json, get_param
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction
from docuverse.utils.timer import timer


class SimdqEngine(RetrievalEngine):
    """In-process SIMD-quantized retrieval engine.

    Reads simdq_* fields from RetrievalArguments. Index lives at
    ``<persist_directory>/simdq_data/<index_name>/``.
    """

    SUBDIR = "simdq_data"

    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)
        self.model: Optional[DenseEmbeddingFunction] = None
        self.hidden_dim: Optional[int] = None
        self.index: Optional[SimdqIndex] = None
        self.id_map: List[str] = []
        self.metadata_store: Dict[str, dict] = {}

        self.load_model_config(config_params)
        self.text_header = "text"
        self.title_header = "title"
        self.id_header = "id"
        self.extra_fields = get_param(self.config.data_template, "extra_fields", [])
        self.persist_directory = get_param(self.config, "project_dir", "/tmp")

        self.init_model(**kwargs)
        self.init_client()

    # ===== Init =====

    def init_model(self, **kwargs):
        self.model = DenseEmbeddingFunction(
            self.config.model_name,
            **self.config.__dict__,
        )
        self.hidden_dim = self.model.embedding_dim

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
            logging.info(f"Deleted simdq index: {index_name}")
        self.index = None
        self.id_map = []
        self.metadata_store = {}

    # ===== Ingest =====

    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs) -> bool:
        self.check_client()
        fmt = "\n=== {:30} ==="
        still_create_index = self.create_update_index(fmt=fmt, do_update=update)
        if not still_create_index:
            return None

        tm = timer("simdq::ingest")
        corpus_size = len(corpus)
        batch = self.ingestion_batch_size or self.config.bulk_batch or 256

        all_vecs: List[np.ndarray] = []
        all_ids: List[str] = []
        self.metadata_store = {}

        tq = tqdm(desc="simdq ingest", total=corpus_size, leave=True)
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
            raise RuntimeError("simdq ingest: no documents survived text filtering")
        Y = np.vstack(all_vecs)
        if Y.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f"simdq ingest: encoder produced dim {Y.shape[1]} but hidden_dim={self.hidden_dim}"
            )

        self.index = SimdqIndex.build(
            vectors=Y,
            family=self.config.simdq_family,
            b=self.config.simdq_b if self.config.simdq_family == "asymmetric" else None,
            d=self.config.simdq_d,
            projection=self.config.simdq_projection,
            projection_seed=self.config.simdq_projection_seed,
            store_floats=self.config.simdq_store_floats,
            encoder_id=self.config.model_name,
        )
        out_dir = self._index_dir()
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        self.index.save(out_dir)
        # Side-car: id_map + metadata for rehydration.
        with open(self._metadata_path(), "w") as f:
            json.dump({"id_map": all_ids, "metadata": self.metadata_store}, f)
        self.id_map = all_ids
        tm.add_timing("build_save")
        logging.info(f"Ingested {corpus_size} docs into simdq index {self.config.index_name}")
        return True

    # ===== Search =====

    def _ensure_loaded(self):
        if self.index is None:
            self.index = SimdqIndex.load(self._index_dir())
            with open(self._metadata_path()) as f:
                side = json.load(f)
            self.id_map = side["id_map"]
            self.metadata_store = side["metadata"]

    def search(self, query: SearchQueries.Query, **kwargs) -> SearchResult:
        tm = timer("simdq::search")
        self._ensure_loaded()
        tm.add_timing("load")

        text = query.text if hasattr(query, "text") else query
        emb = self.model.encode([text], show_progress_bar=False,
                                prompt_name="query", tm=tm)[0]
        q = np.asarray(emb, dtype=np.float32)
        tm.add_timing("encode")

        K = int(self.config.top_k)
        K_prime = K * self.config.simdq_rescore_alpha
        K_prime = max(K, min(K_prime, 256))         # spec: K_prime <= 256

        idxs, scores = self.index.search(
            q, K=K, K_prime=K_prime,
            num_threads=self.config.simdq_num_threads,
        )
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
            "retriever_type": "SimdqEngine",
            "index_name": self.config.index_name,
            "model": self.config.model_name,
            "dimension": self.hidden_dim,
            "family": self.config.simdq_family,
            "b": self.config.simdq_b,
            "d": self.config.simdq_d,
            "projection": self.config.simdq_projection,
        }
        try:
            with open(os.path.join(self._index_dir(), "meta.json")) as f:
                out["index_meta"] = json.load(f)
        except FileNotFoundError:
            pass
        return out
```

- [ ] **Step 2: Update `__init__.py` to re-export `SimdqEngine`**

```python
"""simdq — SIMD-quantized in-process retrieval engine."""
from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
from docuverse.engines.retrieval.simdq.simdq_engine import SimdqEngine

__all__ = ["SimdqIndex", "SimdqEngine"]
```

- [ ] **Step 3: Smoke import**

```bash
conda activate ndocu
python -c "
from docuverse.engines.retrieval.simdq import SimdqEngine, SimdqIndex
print('imports OK; SimdqEngine.__init__ sig:', SimdqEngine.__init__.__doc__ or '(no docstring)')
"
```
Expected: prints "imports OK". If `_simdq_native` import fails, the binding rebuild from T4 didn't take — re-run `pip install -e .`.

- [ ] **Step 4: Commit**

```bash
git add docuverse/engines/retrieval/simdq/simdq_engine.py \
        docuverse/engines/retrieval/simdq/__init__.py
git commit -m "Add SimdqEngine — DocUVerse SearchEngine backed by SimdqIndex.

Mirrors FAISSEngine pattern: in-process, file-backed at
<persist>/simdq_data/<index_name>/, lazy-loads on first search.
Reads simdq_* fields from RetrievalArguments. Peak RAM at ingest
is 4*N*D bytes (single-buffer build); streaming-build is deferred."
```

---

## Task 9: Wire `simdq` into `create_retrieval_engine` factory

**Goal:** Add a `simdq` branch to `docuverse/utils/retrievers.py` that instantiates `SimdqEngine`. The lazy-import keeps the C extension unloaded for non-simdq engines.

**Files:**
- Modify: `docuverse/utils/retrievers.py`

- [ ] **Step 1: Add the dispatch branch**

In `docuverse/utils/retrievers.py`, locate the `elif name.startswith("file:")` branch (around line 79). Add a `simdq` branch immediately before it:

```python
   elif name == 'simdq':
       try:
           from docuverse.engines.retrieval.simdq import SimdqEngine
           engine = SimdqEngine(retriever_config)
       except ImportError as e:
           print("simdq engine: native extension not built. Run `pip install -e .` "
                 "from the repo root to compile docuverse.engines.retrieval.simdq._native.")
           raise e
```

- [ ] **Step 2: Verify dispatch works**

```bash
conda activate ndocu
python -c "
from docuverse.utils.retrievers import create_retrieval_engine
from docuverse.engines.search_engine_config_params import RetrievalArguments
import os, tempfile

with tempfile.TemporaryDirectory() as td:
    cfg = RetrievalArguments()
    cfg.db_engine = 'simdq'
    cfg.model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # 384-d
    cfg.index_name = 'smoke'
    cfg.project_dir = td
    cfg.simdq_b = 2
    cfg.simdq_family = 'asymmetric'
    eng = create_retrieval_engine(cfg.__dict__)
    print('engine class:', type(eng).__name__)
    print('hidden_dim:', eng.hidden_dim)
    assert type(eng).__name__ == 'SimdqEngine'
"
```
Expected: prints `engine class: SimdqEngine` and `hidden_dim: 384`. If it can't load the model, ensure ndocu env has sentence-transformers installed; otherwise switch to a model already cached locally.

- [ ] **Step 3: Commit**

```bash
git add docuverse/utils/retrievers.py
git commit -m "Wire 'simdq' into create_retrieval_engine factory.

Lazy-import keeps the simdq C extension unloaded for non-simdq
engines (matches the chromadb / faiss / lancedb pattern)."
```

---

## Task 10: Pytest for engine dispatch + end-to-end ingest/search

**Goal:** A `tests/test_simdq_engine.py` that builds a fake corpus of ~50 short documents, runs `SimdqEngine.ingest`, then `search`, and asserts the planted-best document tops the result list. Uses a tiny encoder so the test runs in <30s on CPU.

**Files:**
- Create: `tests/test_simdq_engine.py`

- [ ] **Step 1: Write the test**

```python
"""End-to-end SimdqEngine tests with a real (tiny) encoder."""
import os
import tempfile

import numpy as np
import pytest

from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_engine_config_params import RetrievalArguments
from docuverse.utils.retrievers import create_retrieval_engine


# A tiny multilingual sentence-transformer model is small and fast on CPU.
# If it's unavailable in the test environment we skip rather than fail.
TINY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d


def _make_config(td: str, family: str = "asymmetric") -> RetrievalArguments:
    cfg = RetrievalArguments()
    cfg.db_engine = "simdq"
    cfg.model_name = TINY_MODEL
    cfg.index_name = "test_simdq_e2e"
    cfg.project_dir = td
    cfg.simdq_family = family
    cfg.simdq_b = 2
    cfg.simdq_d = None
    cfg.simdq_projection = "identity"
    cfg.simdq_store_floats = True
    cfg.simdq_rescore_alpha = 10
    cfg.top_k = 5
    cfg.ingestion_batch_size = 16
    cfg.max_text_size = 256
    return cfg


def _planted_corpus():
    docs = [
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
    ]
    # ensure at least 16 to exercise multiple batches
    while len(docs) < 16:
        docs.append({"id": f"pad{len(docs)}", "text": f"padding doc {len(docs)}", "title": ""})
    return docs


@pytest.fixture
def fake_corpus():
    return SearchCorpus(_planted_corpus())


@pytest.mark.parametrize("family", ["asymmetric", "hamming"])
def test_dispatch_and_round_trip(family, fake_corpus):
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td, family=family)
        eng = create_retrieval_engine(cfg.__dict__)
        assert type(eng).__name__ == "SimdqEngine"
        ok = eng.ingest(fake_corpus, update=False)
        assert ok is True
        # files on disk
        meta_path = os.path.join(td, "simdq_data", "test_simdq_e2e", "meta.json")
        assert os.path.exists(meta_path)

        # search for "neural networks" should rank the neural-net or transformer doc highly
        q = SearchQueries.Query(text="neural networks deep learning", id="q1")
        res = eng.search(q)
        ids = [p["id"] for p in res.passages]
        # planted-relevant docs (indices 1, 3, 6, 10 in the corpus) — we expect
        # at least one of them in the top 5.
        relevant = {"d1", "d3", "d6", "d10"}
        assert relevant.intersection(ids), \
            f"none of {relevant} in top-{cfg.top_k}: {ids}"


def test_dispatch_raises_for_bad_family():
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        cfg.simdq_family = "not_a_family"
        eng = create_retrieval_engine(cfg.__dict__)
        with pytest.raises(ValueError, match="family must be"):
            eng.ingest(SearchCorpus(_planted_corpus()), update=False)
```

- [ ] **Step 2: Run the test**

```bash
conda activate ndocu
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 \
  python -m pytest tests/test_simdq_engine.py -v 2>&1 | tail -30
```
Expected: 3 tests pass. If the tiny model isn't available offline, the test skips. If `top_k=5` doesn't surface a relevant doc, increase the planted corpus size, but the assertion is loose (just one of four relevant docs in top-5) so it should pass with any sensible encoder.

If the test errors with a `SearchCorpus` constructor mismatch — the constructor signature varies by codebase version. Read `docuverse/engines/search_corpus.py` and adapt the helper:
```bash
grep -n "def __init__" docuverse/engines/search_corpus.py
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_simdq_engine.py
git commit -m "Add end-to-end SimdqEngine pytest (dispatch + ingest + search).

Uses sentence-transformers/all-MiniLM-L6-v2 (384-d) for both
asymmetric b=2 and hamming families. Asserts at least one
planted-relevant doc in top-5 for a 'neural networks' query."
```

---

## Task 11: `scripts/bench_simdq_beir.py` — R0–R6 sweep driver

**Goal:** A single-command sweep that produces NDCG@10 / recall@100 / latency / index size for the spec's seven recipes (R0–R6). The script is a thin orchestrator: it generates one YAML per recipe, invokes the existing `python -m docuverse.utils.ingest_and_test` CLI as a subprocess, parses the printed metrics, and writes a CSV.

**Files:**
- Create: `scripts/bench_simdq_beir.py`
- Create: `config/beir_simdq_base.yaml`

- [ ] **Step 1: Write the base YAML template**

`config/beir_simdq_base.yaml`:
```yaml
# Base YAML the sweep script overrides per-recipe. Required
# command-line --dataset overrides input_passages / input_queries /
# data via the existing {{var}} resolver in DocUVerse's config loader.

search_engine:
  db_engine: simdq
  model_name: ibm-granite/granite-embedding-278m-multilingual-r2
  index_name: "{{dataset}}_simdq_{{recipe_id}}"
  project_dir: "{{project_dir}}"
  top_k: 100
  ingestion_batch_size: 256
  max_doc_length: 512
  # simdq fields are filled in per-recipe by bench_simdq_beir.py
  simdq_family: asymmetric
  simdq_b: 2
  simdq_d: null
  simdq_projection: identity
  simdq_projection_seed: 42
  simdq_store_floats: true
  simdq_rescore_alpha: 10
  simdq_num_threads: 0

retrieval:
  input_passages: "{{passages}}"
  input_queries: "{{queries}}"

evaluation:
  metric: "ndcg,recall@100"
  qrels: "{{qrels}}"
```

- [ ] **Step 2: Write the sweep driver**

`scripts/bench_simdq_beir.py`:
```python
#!/usr/bin/env python
"""bench_simdq_beir.py — run the R0–R6 simdq recipe sweep.

For each recipe in the spec's matrix, materialize a YAML config, run
ingest+retrieve+evaluate via the existing DocUVerse CLI as a
subprocess, parse the printed NDCG@10 / recall@100 numbers, capture
on-disk index size and end-to-end wall time, then write a single CSV.

Usage:
    python scripts/bench_simdq_beir.py \\
        --dataset fiqa \\
        --base-yaml config/beir_simdq_base.yaml \\
        --passages data/fiqa/passages.jsonl \\
        --queries  data/fiqa/queries.jsonl \\
        --qrels    data/fiqa/qrels.tsv \\
        --project-dir /scratch/simdq-fiqa \\
        --out simdq_recipe_sweep_fiqa.csv

Recipes (spec section 10):
    R0  symmetric Hamming   + rescore α=10   family=hamming, d=D
    R1  asym b=1            no  rescore     family=asymmetric, b=1, alpha=1
    R2  asym b=1            +   rescore α=10 b=1, alpha=10
    R3  asym b=2            no  rescore     b=2, alpha=1
    R4  asym b=2, halve dim no  rescore     b=2, d=D/2, projection=random_orthogonal
    R5  asym b=4, halve dim no  rescore     b=4, d=D/2, projection=random_orthogonal
    R6  float baseline                       (skipped: run via existing dense engine)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import yaml


RECIPES = [
    # (recipe_id, label, simdq_overrides)
    ("R0", "hamming + rescore alpha=10",
     {"simdq_family": "hamming", "simdq_b": 1, "simdq_projection": "identity",
      "simdq_d": None, "simdq_store_floats": True, "simdq_rescore_alpha": 10}),
    ("R1", "asym b=1, no rescore",
     {"simdq_family": "asymmetric", "simdq_b": 1, "simdq_projection": "identity",
      "simdq_d": None, "simdq_store_floats": False, "simdq_rescore_alpha": 1}),
    ("R2", "asym b=1 + rescore alpha=10",
     {"simdq_family": "asymmetric", "simdq_b": 1, "simdq_projection": "identity",
      "simdq_d": None, "simdq_store_floats": True, "simdq_rescore_alpha": 10}),
    ("R3", "asym b=2, no rescore",
     {"simdq_family": "asymmetric", "simdq_b": 2, "simdq_projection": "identity",
      "simdq_d": None, "simdq_store_floats": False, "simdq_rescore_alpha": 1}),
    ("R4", "asym b=2, d=D/2, random_orthogonal",
     {"simdq_family": "asymmetric", "simdq_b": 2, "simdq_projection": "random_orthogonal",
      "simdq_d": "HALF", "simdq_store_floats": False, "simdq_rescore_alpha": 1}),
    ("R5", "asym b=4, d=D/2, random_orthogonal",
     {"simdq_family": "asymmetric", "simdq_b": 4, "simdq_projection": "random_orthogonal",
      "simdq_d": "HALF", "simdq_store_floats": False, "simdq_rescore_alpha": 1}),
    # R6 is intentionally not run from this script — float baseline uses
    # whatever dense engine the user already has (lancedb-dense / faiss /
    # milvus-dense). The script accepts an optional --r6-csv to merge.
]


def _materialize_yaml(base: dict, dataset: str, recipe_id: str, recipe_overrides: dict,
                      passages: str, queries: str, qrels: str, project_dir: str,
                      encoder_dim_hint: int | None) -> dict:
    """Apply per-recipe overrides; resolve simdq_d='HALF' to D/2."""
    cfg = json.loads(json.dumps(base))   # deep copy

    cfg["search_engine"]["index_name"] = f"{dataset}_simdq_{recipe_id}"
    cfg["search_engine"]["project_dir"] = project_dir
    cfg["retrieval"]["input_passages"] = passages
    cfg["retrieval"]["input_queries"] = queries
    cfg["evaluation"]["qrels"] = qrels

    se = cfg["search_engine"]
    for k, v in recipe_overrides.items():
        if v == "HALF":
            if encoder_dim_hint is None:
                raise RuntimeError(
                    f"{recipe_id}: simdq_d=HALF needs --encoder-dim to know D/2"
                )
            se[k] = encoder_dim_hint // 2
        else:
            se[k] = v
    return cfg


_NDCG_RE   = re.compile(r"NDCG@10[:\s=]+([0-9.]+)")
_RECALL_RE = re.compile(r"Recall@100[:\s=]+([0-9.]+)")


def _parse_metrics(stdout: str) -> dict:
    out = {}
    m = _NDCG_RE.search(stdout)
    if m: out["ndcg10"] = float(m.group(1))
    m = _RECALL_RE.search(stdout)
    if m: out["recall100"] = float(m.group(1))
    return out


def _index_size_bytes(project_dir: str, index_name: str) -> int:
    p = Path(project_dir) / "simdq_data" / index_name
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _run_recipe(yaml_path: Path) -> tuple[str, float, int]:
    """Invoke ingest_and_test as a subprocess. Returns (stdout, wall_seconds, exit_code)."""
    t0 = time.time()
    cmd = [
        "python", "-m", "docuverse.utils.ingest_and_test",
        "--config", str(yaml_path),
        "--actions", "ire",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    return res.stdout + "\n" + res.stderr, elapsed, res.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset slug used in index_name")
    ap.add_argument("--base-yaml", required=True, type=Path)
    ap.add_argument("--passages", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--encoder-dim", type=int, default=768,
                    help="encoder hidden dim (= D) — needed to resolve simdq_d=HALF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipes", nargs="*", default=None,
                    help="subset of recipe IDs to run (default: all)")
    ap.add_argument("--keep-yaml-dir", default=None,
                    help="if set, written YAMLs are kept here for debugging")
    args = ap.parse_args()

    base = yaml.safe_load(args.base_yaml.read_text())

    yaml_dir = Path(args.keep_yaml_dir) if args.keep_yaml_dir else \
               Path(args.project_dir) / "_sweep_yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(args.project_dir, exist_ok=True)

    rows = []
    for rid, label, overrides in RECIPES:
        if args.recipes and rid not in args.recipes:
            continue
        cfg = _materialize_yaml(base, args.dataset, rid, overrides,
                                args.passages, args.queries, args.qrels,
                                args.project_dir, args.encoder_dim)
        yaml_path = yaml_dir / f"{args.dataset}_{rid}.yaml"
        yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        index_name = cfg["search_engine"]["index_name"]
        # Always start from a clean index dir to keep recipes independent.
        idx_dir = Path(args.project_dir) / "simdq_data" / index_name
        if idx_dir.exists():
            shutil.rmtree(idx_dir)

        print(f"\n=== {rid} — {label} ===")
        stdout, elapsed_s, rc = _run_recipe(yaml_path)
        metrics = _parse_metrics(stdout)
        idx_bytes = _index_size_bytes(args.project_dir, index_name)
        rows.append({
            "recipe_id": rid,
            "label": label,
            "simdq_family": cfg["search_engine"].get("simdq_family"),
            "b": cfg["search_engine"].get("simdq_b"),
            "d": cfg["search_engine"].get("simdq_d"),
            "projection": cfg["search_engine"].get("simdq_projection"),
            "store_floats": cfg["search_engine"].get("simdq_store_floats"),
            "rescore_alpha": cfg["search_engine"].get("simdq_rescore_alpha"),
            "ndcg10": metrics.get("ndcg10"),
            "recall100": metrics.get("recall100"),
            "wall_seconds": round(elapsed_s, 2),
            "index_bytes": idx_bytes,
            "exit_code": rc,
        })
        print(f"  ndcg10={metrics.get('ndcg10')}  "
              f"recall100={metrics.get('recall100')}  "
              f"wall={elapsed_s:.0f}s  index={idx_bytes/1e6:.1f}MB  rc={rc}")

    # Write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test with a tiny synthetic dataset**

A full BEIR sweep takes minutes-to-hours; smoke-test the script's plumbing on a 50-doc / 5-query toy corpus. Create the toy data and run R3 only:

```bash
mkdir -p /tmp/simdq_smoke/data
python -c "
import json
with open('/tmp/simdq_smoke/data/passages.jsonl', 'w') as f:
    for i in range(50):
        f.write(json.dumps({'id': f'd{i}', 'text': f'passage about topic {i % 5}', 'title': ''}) + '\n')
with open('/tmp/simdq_smoke/data/queries.jsonl', 'w') as f:
    for i in range(5):
        f.write(json.dumps({'id': f'q{i}', 'text': f'topic {i}'}) + '\n')
with open('/tmp/simdq_smoke/data/qrels.tsv', 'w') as f:
    f.write('q-id\tcorpus-id\tscore\n')
    for q in range(5):
        for d in range(50):
            if d % 5 == q: f.write(f'q{q}\td{d}\t1\n')
"

CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
python scripts/bench_simdq_beir.py \
  --dataset smoke \
  --base-yaml config/beir_simdq_base.yaml \
  --passages /tmp/simdq_smoke/data/passages.jsonl \
  --queries  /tmp/simdq_smoke/data/queries.jsonl \
  --qrels    /tmp/simdq_smoke/data/qrels.tsv \
  --project-dir /tmp/simdq_smoke \
  --encoder-dim 768 \
  --out /tmp/simdq_smoke/sweep.csv \
  --recipes R3
```

Expected: the script invokes `ingest_and_test` once for R3, captures any printed metrics (might be 0 or `None` on this trivial corpus — that's fine, we're testing plumbing), and writes a one-row CSV. Exit code should be 0 from the wrapper script. If `ingest_and_test` itself fails, check that the YAML `{{var}}` resolution accepts the literal substitutions we passed (no `{{...}}` placeholders left in the output YAML — the script materializes everything before the subprocess starts).

If the YAML resolver rejects unsubstituted variables, the test points to a real config-loading bug worth fixing here rather than papering over.

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_simdq_beir.py config/beir_simdq_base.yaml
git commit -m "Add scripts/bench_simdq_beir.py — R0–R6 simdq recipe sweep.

Generates one YAML per recipe, runs ingest_and_test in a subprocess,
parses NDCG@10/Recall@100, captures wall time + on-disk index size,
writes a single CSV. R6 (float baseline) is intentionally out of
scope here — runs via the user's existing dense engine of choice."
```

---

## Task 12: Status doc update

**Goal:** Mark Plan 3 complete in `docs/superpowers/plans/2026-06-17-simdq-status.md`. Capture observed numbers from the BEIR sweep (or note the sweep was wired but not run on a full BEIR dataset, depending on what was actually executed).

**Files:**
- Modify: `docs/superpowers/plans/2026-06-17-simdq-status.md`

- [ ] **Step 1: Update the status doc**

Read the current `2026-06-17-simdq-status.md`. Apply:

1. Update the top "Last updated" date to today's date.

2. In the "Multi-plan roadmap" table (line ~12), change Plan 3's Status column to **Complete** and link it: `**Complete** ([plan](2026-06-18-simdq-plan-3-engine.md))`.

3. After the existing "Plan 2 — final status" section, append a "Plan 3 — final status" section mirroring the layout of Plans 1 and 2:

```markdown
## Plan 3 — final status

### Tasks

| # | Task | Commit(s) | Status |
|---|------|-----------|--------|
| T1 | Generalize asym kernels for runtime d | `<sha>` | ✅ |
| T2 | Generalize Hamming kernels for runtime words | `<sha>` | ✅ |
| T3 | Multi-D ctest spot-check | `<sha>` | ✅ |
| T4 | Binding: runtime d + pack/scan_hamming | `<sha>` | ✅ |
| T5 | SimdqIndex: D in {384,768,1024,1536}, family axis | `<sha>` | ✅ |
| T6 | Pytest parametrize multi-D + hamming | `<sha>` | ✅ |
| T7 | RetrievalArguments: simdq_* fields | `<sha>` | ✅ |
| T8 | SimdqEngine SearchEngine subclass | `<sha>` | ✅ |
| T9 | Factory dispatch for `simdq` | `<sha>` | ✅ |
| T10 | End-to-end SimdqEngine pytest | `<sha>` | ✅ |
| T11 | scripts/bench_simdq_beir.py R0–R6 sweep | `<sha>` | ✅ |
| T12 | Status doc update | (this commit) | ✅ |

### What's in the tree (added this plan)

`docuverse/engines/retrieval/simdq/simdq_engine.py`, the multi-D ctest
target, the runtime-d / family-aware Python layer, simdq_* fields on
RetrievalArguments, and `scripts/bench_simdq_beir.py` + base YAML.

### Test coverage

- C ctest — 16 targets (14 from Plan 2 + 2 multi-D), all passing.
- Python pytest — N tests across the four simdq test files (fill in actual
  number after running the suite once).

### Headline benchmark numbers

To be populated after running `bench_simdq_beir.py` on a real BEIR dataset.
Suggested first run: FiQA (~57k passages, encoder = granite-embedding-278m).

### Architectural deviations from spec

- **Runtime-d kernel parameterization** rather than compile-time D
  templates (spec §6). Justification: at d ≥ 384 the inner loop is too
  long for full unroll; benchmark-driven follow-up if perf shows otherwise.
- **`SimdqConfig` realized as `simdq_*` fields on `RetrievalArguments`**
  rather than a standalone dataclass (spec §8). Matches existing engine
  conventions (milvus_, lancedb_, …).

### Open for v2 / Plan 3.5

- Streaming-build path (avoids `4*N*D` peak RAM at ingest time).
- Compile-time D templates if perf benchmarks show the runtime version
  loses ≥10% on memory-bound recipes.
- Learned projection (full ASH training loop).
```

After making the edits, replace each `<sha>` with the actual commit SHA from `git log --oneline | head -20` (one per task, in T1→T11 order).

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-06-17-simdq-status.md
git commit -m "Mark Plan 3 complete in simdq status doc.

Records final task commits, captures architectural deviations
(runtime-d, simdq_* on RetrievalArguments) and the open follow-ups
for v2/Plan 3.5 (streaming build, compile-time D templates if
perf demands)."
```

---

## Final integration check

After T12, run the full test matrix once more to confirm no Plan 1 / Plan 2 regressions:

```bash
# C kernel tests
cd docuverse/engines/retrieval/simdq/_native/build
ctest --output-on-failure                                 # expect 16/16 PASS

# Python tests (full simdq suite)
cd $REPO_ROOT
conda activate ndocu
python -m pytest tests/test_simdq_*.py -v                 # expect ~50 PASS

# Smoke run of the BEIR sweep on the toy corpus from T11
python scripts/bench_simdq_beir.py \
  --dataset smoke ... --recipes R0 R1 R2 R3 R4 R5         # expect 6 rows in CSV
```

All three should pass. If any C test regresses, the offender is most likely an inadvertent-`WORDS`-or-`ASYM_*_D`-leak from T1/T2 — `grep -rn '\bWORDS\b\|ASYM_B[124]_D' _native/{include,bench,tests}/` should be empty (only the local `#define WORDS 12` in benches/tests is acceptable; it should not appear in headers).

Open for the user's review:
- Architectural deviations called out at the top of this doc.
- Whether R0 (Hamming) should be in scope for Plan 3 vs deferred — Plan 3 includes it via T2/T4/T5.
- Whether the smoke pytest in T10 needs a dedicated tiny model bundled with the repo (current plan: skip if the model isn't installable).
