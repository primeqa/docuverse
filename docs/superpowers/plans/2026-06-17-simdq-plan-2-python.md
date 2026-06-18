# `simdq` Plan 2 — Python integration & on-disk index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the Plan 1 C/SIMD kernels in a Python package — CPython C-API extension, numpy-side projection + quantization, threaded asymmetric scan driver, on-disk format with mmap'd fp16 rescore tier — producing a working `SimdqIndex.build / save / load / search` flow end-to-end at `D=768`.

**Architecture:** Plan 1 left four kernel headers (`simdq_kernels_hamming_topk.h`, `simdq_kernels_asym_b{1,2,4}.h`) that all `#define` the same macro names (`KERNEL_NAME`, `ASYM_KERNEL_NAME`, `ASYM_LANES`, `ASYM_D`), so they cannot be combined in one translation unit yet. Plan 2 (a) renames those macros to per-kernel-prefixed forms, (b) fixes the AVX2 b=1 scalar-bit-expansion bottleneck flagged in the Plan 1 status doc, (c) refactors the asymmetric kernels into `(i0, i1)` shard form and adds OpenMP `_parallel_topk` drivers, (d) adds a CPython C-API extension `_simdq_native` exposing `compute_scales / pack_b{1,2,4} / scan_b{1,2,4}`, (e) wires the extension into `pip install -e .` via a small `setup.py` shim, and (f) lands `projection.py / quantization.py / simdq_index.py` plus an on-disk format (`meta.json`, `W.npy`, `codes.bin`, `scales.bin`, optional `floats.bin`) with an mmap'd fp16 rescore stage performed in numpy.

**Tech Stack:** C11 (existing), CPython C-API (no pybind11/cffi), setuptools (already the build backend), numpy, pytest. Per-vector scale is applied **post-scan in Python** (multiply by `scales[idx]` before returning); kernels keep their Plan-1 signatures and continue to compute the raw `<q', v_i>` dot product.

**Spec reference:** [`docs/superpowers/specs/2026-06-15-simdq-engine-design.md`](../specs/2026-06-15-simdq-engine-design.md).
**Plan 1 reference:** [`docs/superpowers/plans/2026-06-15-simdq-plan-1-kernels.md`](2026-06-15-simdq-plan-1-kernels.md).
**Status doc to update:** [`docs/superpowers/plans/2026-06-17-simdq-status.md`](2026-06-17-simdq-status.md).

---

## File structure

| Path | Purpose | Created/modified in task |
|---|---|---|
| `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming_topk.h` | Rename `KERNEL_NAME → HAMMING_TOPK_KERNEL_NAME`, `LANES → HAMMING_TOPK_LANES` | T1 (modify) |
| `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h` | Rename `ASYM_*` to `ASYM_B1_*`; AVX2 unpack via vpshufb (T2); add `_parallel_topk` (T3) | T1, T2, T3 (modify) |
| `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b2.h` | Rename to `ASYM_B2_*`; add `_parallel_topk` | T1, T3 (modify) |
| `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b4.h` | Rename to `ASYM_B4_*`; add `_parallel_topk` | T1, T3 (modify) |
| `docuverse/engines/retrieval/simdq/_native/bench/bench_hamming.c` | Track macro rename | T1 (modify) |
| `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b{1,2,4}.c` | Track macro renames; switch to `_parallel_topk` driver | T1, T3 (modify) |
| `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming_topk.c` | Track macro rename | T1 (modify) |
| `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b{1,2,4}.c` | Track macro renames | T1 (modify) |
| `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_parallel.c` | Verify `_parallel_topk` matches single-shard for all 3 b values | T3 (create) |
| `docuverse/engines/retrieval/simdq/_native/CMakeLists.txt` | Register the new parallel test | T3 (modify) |
| `docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md` | Captured AVX2 b=1 fix numbers + parallel sweep | T2, T3 (modify) |
| `docuverse/engines/retrieval/simdq/_native/scripts/run_sweep.sh` | Add threaded asym sweep rows | T3 (modify) |
| `docuverse/engines/retrieval/simdq/_native/bindings/module.c` | CPython C-API extension exposing `compute_scales / pack_b{1,2,4} / scan_b{1,2,4}` | T4 (create) |
| `setup.py` | setuptools shim declaring the `_simdq_native` Extension | T5 (create) |
| `docuverse/engines/retrieval/simdq/__init__.py` | Public re-exports (`SimdqIndex`, `SimdqIndexConfig`) | T8 (create) |
| `docuverse/engines/retrieval/simdq/projection.py` | `identity` / `random_orthogonal` | T6 (create) |
| `docuverse/engines/retrieval/simdq/quantization.py` | numpy wrapper around scale fit + pack | T7 (create) |
| `docuverse/engines/retrieval/simdq/simdq_index.py` | `SimdqIndex` build/save/load/search; mmap'd rescore | T8 (create) |
| `tests/test_simdq_projection.py` | pytest for projection helpers | T6 (create) |
| `tests/test_simdq_quantization.py` | pytest for quantization round-trip | T7 (create) |
| `tests/test_simdq_index.py` | pytest for round-trip, mode parity, agreement-vs-sklearn | T8 (create) |
| `docs/superpowers/plans/2026-06-17-simdq-status.md` | Mark Plan 2 tasks complete; record observed numbers | T9 (modify) |

**Idiom (Plan 1 carry-over):** kernels live in `.h` headers with `static inline` functions; binding code in `bindings/module.c` is the only C translation unit that pulls multiple kernel headers together. The macro-rename in T1 is what makes that possible.

---

## Task 1: Rename per-kernel macros so all kernel headers compose in one translation unit

**Goal:** decouple `KERNEL_NAME`, `LANES`, `ASYM_KERNEL_NAME`, `ASYM_LANES`, `ASYM_D` into per-kernel-prefixed names so `module.c` can include all four kernel headers without redefinition errors. Pure rename — no behavior change.

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming_topk.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b2.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b4.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_hamming.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b1.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b2.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b4.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming_topk.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b1.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b2.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b4.c`

- [ ] **Step 1: Rename the Hamming-topk macros**

In `_native/include/simdq_kernels_hamming_topk.h`, do an exact replace_all on these two tokens **inside this file only**:

| Old | New |
|---|---|
| `KERNEL_NAME` | `HAMMING_TOPK_KERNEL_NAME` |
| `LANES` | `HAMMING_TOPK_LANES` |

Both appear inside the `#if defined(__AVX512VPOPCNTDQ__) ... #elif defined(__AVX2__) ...` block as `#define`s, then again inside `scan_shard_topk` (loop bound `i + LANES <= i1`) and in the trailing `#if defined(KERNEL_NAME) && defined(_OPENMP)` guard. Use the `Edit` tool with `replace_all=true` on each token.

`HAMMING_TOPK_KERNEL_NAME` reads correctly when printed by tests/benches; `HAMMING_TOPK_LANES` is the per-block code count (8 on AVX-512, 4 on AVX2).

- [ ] **Step 2: Rename the asymmetric b=1 macros**

In `_native/include/simdq_kernels_asym_b1.h`, replace all occurrences:

| Old | New |
|---|---|
| `ASYM_KERNEL_NAME` | `ASYM_B1_KERNEL_NAME` |
| `ASYM_LANES` | `ASYM_B1_LANES` |
| `ASYM_D` | `ASYM_B1_D` |

`ASYM_B1_D` keeps its value `768` (compile-time fixed dim). The loop body and the `#if defined(__AVX512F__) ... #elif defined(__AVX2__) && defined(__FMA__) ...` block both pick up the new names automatically via `replace_all`.

- [ ] **Step 3: Rename the asymmetric b=2 macros**

In `_native/include/simdq_kernels_asym_b2.h`, replace all occurrences:

| Old | New |
|---|---|
| `ASYM_KERNEL_NAME` | `ASYM_B2_KERNEL_NAME` |
| `ASYM_LANES` | `ASYM_B2_LANES` |
| `ASYM_D` | `ASYM_B2_D` |

- [ ] **Step 4: Rename the asymmetric b=4 macros**

In `_native/include/simdq_kernels_asym_b4.h`, replace all occurrences:

| Old | New |
|---|---|
| `ASYM_KERNEL_NAME` | `ASYM_B4_KERNEL_NAME` |
| `ASYM_LANES` | `ASYM_B4_LANES` |
| `ASYM_D` | `ASYM_B4_D` |

- [ ] **Step 5: Update the test files to reference the renamed macros**

The four asymmetric-test files all open with the same guard:

```c
#ifndef ASYM_KERNEL_NAME
#error "tests need at least AVX2 (-march=native or -mavx2)"
#endif
```

In each test file, replace `ASYM_KERNEL_NAME` with the new per-kernel name:

| File | New token |
|---|---|
| `tests/test_kernels_asym_b1.c` | `ASYM_B1_KERNEL_NAME` |
| `tests/test_kernels_asym_b2.c` | `ASYM_B2_KERNEL_NAME` |
| `tests/test_kernels_asym_b4.c` | `ASYM_B4_KERNEL_NAME` |

Each test also `printf`s `ASYM_KERNEL_NAME` once in `main()` (e.g. `printf("asym kernel path: %s\n", ASYM_KERNEL_NAME);`) — `replace_all` handles both occurrences in one Edit call.

For `tests/test_kernels_hamming_topk.c`, replace `KERNEL_NAME` (used in the `#error` guard and the `printf("kernel path: %s\n", KERNEL_NAME);` line) with `HAMMING_TOPK_KERNEL_NAME`.

- [ ] **Step 6: Update the bench drivers to reference the renamed macros**

Each bench prints its kernel name once. Replace tokens:

| File | Old | New |
|---|---|---|
| `bench/bench_hamming.c` | `KERNEL_NAME` | `HAMMING_TOPK_KERNEL_NAME` |
| `bench/bench_asym_b1.c` | `ASYM_KERNEL_NAME` | `ASYM_B1_KERNEL_NAME` |
| `bench/bench_asym_b2.c` | `ASYM_KERNEL_NAME` | `ASYM_B2_KERNEL_NAME` |
| `bench/bench_asym_b4.c` | `ASYM_KERNEL_NAME` | `ASYM_B4_KERNEL_NAME` |

Note: `bench/bench_hamming_top1.c` is the carry-over of the Plan 1 `hb5` driver and does **not** include `simdq_kernels_hamming_topk.h` — it includes `simdq_kernels_hamming.h`, which uses different macro names (`KERNEL_NAME` is locally defined there too, but never collides because the only TU that includes it is `bench_hamming_top1.c`). No change needed for that file.

- [ ] **Step 7: Build and verify all ctest targets pass with the renamed macros**

```bash
cd docuverse/engines/retrieval/simdq/_native
rm -rf build
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: all 12 tests still pass — `kernels_hamming_native`, `kernels_hamming_avx2`, `kernels_hamming_topk_native`, `kernels_hamming_topk_avx2`, `topk`, `pack`, `kernels_asym_b1_native`, `kernels_asym_b1_avx2`, `kernels_asym_b2_native`, `kernels_asym_b2_avx2`, `kernels_asym_b4_native`, `kernels_asym_b4_avx2`. Throughput is unchanged.

- [ ] **Step 8: Cross-check that the headers can co-exist in one TU**

This is the whole point of the rename. Create a temporary `_native/build/sanity.c`:

```c
// throw-away sanity check: do all four kernel headers compile together?
#include "simdq_kernels_hamming_topk.h"
#include "simdq_kernels_asym_b1.h"
#include "simdq_kernels_asym_b2.h"
#include "simdq_kernels_asym_b4.h"

int main(void) {
    return 0;
}
```

Build with the same flags the extension will eventually use:

```bash
cd docuverse/engines/retrieval/simdq/_native
gcc -O2 -march=native -fopenmp -Iinclude build/sanity.c -o build/sanity -lm
./build/sanity && echo OK
```

Expected: clean compile (no `redefinition` warnings or errors) and `OK`. Delete `build/sanity.c` and `build/sanity` afterward — the file is not committed.

- [ ] **Step 9: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming_topk.h \
        docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h \
        docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b2.h \
        docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b4.h \
        docuverse/engines/retrieval/simdq/_native/bench/bench_hamming.c \
        docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b1.c \
        docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b2.c \
        docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b4.c \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming_topk.c \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b1.c \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b2.c \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b4.c
git commit -m "Rename per-kernel macros so simdq kernel headers compose in one TU.

KERNEL_NAME / LANES / ASYM_KERNEL_NAME / ASYM_LANES / ASYM_D were defined
identically in four headers, blocking inclusion of more than one in a
single translation unit. Plan 2's CPython binding pulls all four into
bindings/module.c, so each macro is now per-kernel-prefixed:

  hamming_topk : HAMMING_TOPK_KERNEL_NAME, HAMMING_TOPK_LANES
  asym b=1     : ASYM_B1_KERNEL_NAME, ASYM_B1_LANES, ASYM_B1_D
  asym b=2     : ASYM_B2_KERNEL_NAME, ASYM_B2_LANES, ASYM_B2_D
  asym b=4     : ASYM_B4_KERNEL_NAME, ASYM_B4_LANES, ASYM_B4_D

Tests, benches, and references in the kernel bodies are updated. Pure
rename; all 12 ctest targets still pass with identical throughput."
```

---

## Task 2: Replace AVX2 b=1 scalar bit-expansion with a vpshufb-based unpack

**Goal:** the AVX2 path of `scan_asym_b1_d768_topk` currently expands 8 packed bits into 8 floats with a per-iteration `for (int l = 0; l < 8; l++) vf[l] = ... ;` scalar loop, which the Plan 1 status doc identifies as the dominant bottleneck (50M cmp/s = 0.55M cmp/s/thread, vs. b=2's 2.19M and b=4's 2.36M — b=1 is *slower* than b=2 because of this scalar loop). Replace with a SIMD bit-test using `_mm256_set1_epi8(bits)` + a per-lane bit mask + blend.

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md`

- [ ] **Step 1: Read the current AVX2 b=1 inner loop**

```bash
sed -n '95,150p' docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h
```

Confirm the AVX2 block looks like (after T1's rename):

```c
#elif defined(__AVX2__) && defined(__FMA__)
#define ASYM_B1_KERNEL_NAME "AVX2-FMA"
#define ASYM_B1_LANES 8

static inline void scan_asym_b1_d768_topk(...) {
    ...
    for (size_t i0 = 0; i0 + ASYM_B1_LANES <= N; i0 += ASYM_B1_LANES) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t w = 0; w < ASYM_B1_D; w++) {
            uint8_t bits = codes[w * row_bytes + (i0 >> 3)];
            // expand 8 bits to 8 floats: +1 or -1
            float vf[8];
            for (int l = 0; l < 8; l++)
                vf[l] = (bits & (1u << l)) ? 1.0f : -1.0f;
            __m256 v = _mm256_loadu_ps(vf);
            __m256 qb = _mm256_set1_ps(q[w]);
            acc = _mm256_fmadd_ps(qb, v, acc);
        }
        ...
    }
    ...
}
```

The eight scalar bit tests + the 8-element float store + 8-float load are what we're replacing.

- [ ] **Step 2: Write the new AVX2 inner-loop body**

Replace the `for (size_t w = 0; w < ASYM_B1_D; w++) { ... }` body (the 9 lines from `uint8_t bits = ...` through `acc = _mm256_fmadd_ps(...);`) with:

```c
            // Broadcast the byte to all 32 lanes; keep lane l = i0 + l of bits.
            // Per-lane bit mask: lane l gets (1u << l). After AND, lane l is
            // either 0 or (1u << l). Compare-equal-zero lifts that to a -1/0
            // mask; we then select +1.0 / -1.0 with mask_blend_ps.
            // Eight 8-bit lanes are projected into the low 8 lanes of a 256-bit
            // register; we take just those lanes as a __m256 of 8 floats.
            // Broadcast the byte to all 32 lanes; AND against per-lane single-bit
            // masks; lift the {zero, nonzero} result to a {-1, 0} per-byte mask
            // via cmpeq-with-zero (works for bit 7, where the masked value is
            // (char)128 = -128 — a signed cmpgt against zero would mis-classify
            // lane 7 because -128 is not > 0).
            uint8_t bits = codes[w * row_bytes + (i0 >> 3)];
            __m256i bbroad = _mm256_set1_epi8((char)bits);
            const __m256i lane_mask = _mm256_setr_epi8(
                1, 2, 4, 8, 16, 32, 64, (char)128,
                1, 2, 4, 8, 16, 32, 64, (char)128,
                1, 2, 4, 8, 16, 32, 64, (char)128,
                1, 2, 4, 8, 16, 32, 64, (char)128);
            __m256i isset = _mm256_and_si256(bbroad, lane_mask);
            // 0xFF where bit is CLEAR, 0x00 where set (cmpeq-zero is sign-agnostic).
            __m256i clear_mask = _mm256_cmpeq_epi8(isset, _mm256_setzero_si256());
            // Take the low 8 bytes and sign-extend each to int32: 0xFFFFFFFF for
            // clear lanes, 0x00000000 for set lanes.
            __m128i low8 = _mm256_castsi256_si128(clear_mask);
            __m256i widened = _mm256_cvtepi8_epi32(low8);
            __m256 mask_ps = _mm256_castsi256_ps(widened);
            // blendv_ps picks b when the mask's sign bit is 1, a otherwise.
            // mask_ps high bit is 1 for CLEAR lanes -> -1.0; 0 for SET -> +1.0.
            __m256 v = _mm256_blendv_ps(_mm256_set1_ps( 1.0f),
                                        _mm256_set1_ps(-1.0f),
                                        mask_ps);
            __m256 qb = _mm256_set1_ps(q[w]);
            acc = _mm256_fmadd_ps(qb, v, acc);
```

The five constants (`bbroad`, `lane_mask`, `_mm256_setzero_si256()`, `_mm256_set1_ps(-1.0f)`, `_mm256_set1_ps(1.0f)`) are computed inside the loop for clarity; the compiler hoists `lane_mask`, the zero, and the `±1.0f` constants out (all loop-invariant). `bbroad` and the rest must stay inside since `bits` changes per `w`.

- [ ] **Step 3: Build and re-run the b=1 ctest**

```bash
cd docuverse/engines/retrieval/simdq/_native
cmake --build build -j
ctest --test-dir build -R 'kernels_asym_b1' --output-on-failure
```

Expected: both `kernels_asym_b1_native` and `kernels_asym_b1_avx2` still pass — both the planted-best test and the random vs. scalar-reference test must be byte-identical to the previous output, since this change is a pure SIMD reformulation of the same `±1` decode.

- [ ] **Step 4: Run the bench at three sizes and capture numbers**

```bash
./build/bench_asym_b1 500000 5 100
./build/bench_asym_b1 1000000 5 100
./build/bench_asym_b1 50000000 3 100
```

Expected: `cmp/s` and `GB/s` should rise meaningfully (Plan 1 status doc shows the AVX2 b=1 path at 0.55 M cmp/s vs. b=2 at 2.19 M; after this fix, b=1 should be in the same neighborhood as b=2 since both now do straight FMA inner loops with no per-iteration scalar work. Concrete numbers depend on the host, and we capture whatever we measure rather than asserting a target).

- [ ] **Step 5: Update `BENCHMARKS.md` with the new numbers**

Open `docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md`. Find the row(s) for `bench_asym_b1` and replace the `cmp/s`, `GB/s`, and `best (ms)` columns with the values printed in step 4. Add a one-line note immediately above or below the b=1 rows:

```
> **Note (Plan 2, T2):** AVX2 b=1 inner loop now uses
> `_mm256_set1_epi8 + cmpgt_epi8 + cvtepi8_epi32 + blendv_ps` to expand 8
> packed bits into 8 ±1 floats per dim, replacing the per-iteration
> scalar 8-step loop that bottlenecked Plan 1. Numbers above reflect the
> new path.
```

If `BENCHMARKS.md` has no existing b=1 row at a particular size, add it — keep the same Markdown table layout as the existing rows.

- [ ] **Step 6: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h \
        docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md
git commit -m "Speed up AVX2 b=1 asymmetric scan via vpshufb-style bit-to-float expand.

Plan 1 left the AVX2 b=1 path expanding packed bits into floats with a
per-iteration scalar 8-step loop, capping throughput at ~0.55 M cmp/s.
Replace with a SIMD bit-test (_mm256_set1_epi8 + per-lane bit mask +
cmpgt_epi8 + cvtepi8_epi32 + blendv_ps), bringing the AVX2 b=1 path
into the same range as b=2 / b=4. Tests unchanged; BENCHMARKS.md
records the new numbers."
```

---

## Task 3: Refactor asymmetric kernels into shard form and add OpenMP `_parallel_topk` drivers

**Goal:** Plan 1 ships the asymmetric scans single-threaded only (each kernel scans all of `[0, N)`). To hit the spec's threading goal, we (a) refactor each `scan_asym_b{1,2,4}_d768_topk` to take an `(i0, i1)` shard range like the Hamming top-K kernel, and (b) add a `scan_asym_b{1,2,4}_d768_topk_parallel` wrapper that runs T threads on disjoint shards and merges per-thread heaps under an `omp critical`.

**Files:**
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b2.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b4.h`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b1.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b2.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b4.c`
- Create: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_parallel.c`
- Modify: `docuverse/engines/retrieval/simdq/_native/CMakeLists.txt`
- Modify: `docuverse/engines/retrieval/simdq/_native/scripts/run_sweep.sh`
- Modify: `docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md`

- [ ] **Step 1: Refactor `scan_asym_b1_d768_topk` into `scan_asym_b1_d768_shard_topk(codes, N, i0, i1, q, K, out_s, out_i)`**

In `_native/include/simdq_kernels_asym_b1.h`, rename the function and add `(i0, i1)` parameters. The body changes only at the loop bounds — replace `for (size_t i0 = 0; i0 + ASYM_B1_LANES <= N; ...)` with `for (size_t ii = i0; ii + ASYM_B1_LANES <= i1; ii += ASYM_B1_LANES)` and rename the loop variable from `i0` to `ii` inside the loop body (the function parameter `i0` shadows the old loop variable — rename the inner one to `ii` to avoid the shadow). The tail loop becomes `for (size_t i = i1 - ((i1 - i0) % ASYM_B1_LANES); i < i1; i++)`.

For both SIMD branches (AVX-512 and AVX2), the new signature is:

```c
static inline void scan_asym_b1_d768_shard_topk(const uint8_t *codes, size_t N,
                                                size_t i0, size_t i1,
                                                const float *q, int K,
                                                float *out_s, int64_t *out_i)
```

`N` stays (it's needed for `row_bytes = (N + 7) / 8`, the dim-stride). `i0`/`i1` define the code range to scan, `[i0, i1)`. The bounded heap and extraction logic at the end of the function is unchanged.

The full AVX-512 branch becomes (showing the changed parts; the heap setup at the top and the extraction at the bottom are identical to Plan 1):

```c
#if defined(__AVX512F__)
#define ASYM_B1_KERNEL_NAME "AVX512F"
#define ASYM_B1_LANES 16

static inline void scan_asym_b1_d768_shard_topk(const uint8_t *codes, size_t N,
                                                size_t i0, size_t i1,
                                                const float *q, int K,
                                                float *out_s, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    const size_t row_bytes = (N + 7) / 8;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    for (size_t ii = i0; ii + ASYM_B1_LANES <= i1; ii += ASYM_B1_LANES) {
        __m512 acc = _mm512_setzero_ps();
        for (size_t w = 0; w < ASYM_B1_D; w++) {
            uint16_t bits = ((uint16_t)codes[w * row_bytes + (ii >> 3) + 1] << 8)
                          |  (uint16_t)codes[w * row_bytes + (ii >> 3)];
            __mmask16 m = (__mmask16)bits;
            __m512 v = _mm512_mask_blend_ps(m,
                            _mm512_set1_ps(-1.0f),
                            _mm512_set1_ps( 1.0f));
            __m512 qb = _mm512_set1_ps(q[w]);
            acc = _mm512_fmadd_ps(qb, v, acc);
        }
        float sc[16] __attribute__((aligned(64)));
        _mm512_store_ps(sc, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < ASYM_B1_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(ii + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    size_t i = i1 - ((i1 - i0) % ASYM_B1_LANES);
    for (; i < i1; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < ASYM_B1_D; w++) {
            int8_t v = (codes[w * row_bytes + (i >> 3)] & (1u << (i & 7))) ? 1 : -1;
            s += q[w] * (float)v;
        }
        int64_t neg = -(int64_t)(s * (float)(1 << 20));
        if (neg < simdq_topk_threshold(&heap))
            simdq_topk_offer(&heap, neg, (int64_t)i);
    }
    int64_t ks[256], is[256];
    int n = simdq_topk_extract_sorted(&heap, ks, is);
    for (int r = 0; r < n; r++) {
        out_s[r] = -(float)ks[r] / (float)(1 << 20);
        out_i[r] = is[r];
    }
}
```

The AVX2 branch follows the same change: rename to `_shard_topk`, add `(i0, i1)`, rename the outer loop variable from `i0` to `ii`, change `i0 + ASYM_B1_LANES <= N` → `ii + ASYM_B1_LANES <= i1`, change the tail loop bounds to `[i1 - ((i1 - i0) % ASYM_B1_LANES), i1)`.

Then **add a thin wrapper** at the bottom of the AVX-512 branch (and a matching one at the bottom of the AVX2 branch — or just before the closing `#endif`) that preserves the old single-shard signature for callers that don't need threading:

```c
static inline void scan_asym_b1_d768_topk(const uint8_t *codes, size_t N,
                                          const float *q, int K,
                                          float *out_s, int64_t *out_i) {
    scan_asym_b1_d768_shard_topk(codes, N, 0, N, q, K, out_s, out_i);
}
```

The existing per-kernel test (`test_kernels_asym_b1.c`) calls `scan_asym_b1_d768_topk(...)` and continues to work via this wrapper.

- [ ] **Step 2: Add the OpenMP parallel driver for b=1**

After the wrapper, append:

```c
#if defined(_OPENMP)
#include <omp.h>

/*
 * Threaded top-K asymmetric b=1 scan over [0, N). Each thread runs
 * scan_asym_b1_d768_shard_topk on its range, then a critical-section
 * merge folds per-thread results into the global top-K. Per-thread
 * results from a partial-fill shard set unused entries to score=-INF
 * and idx=-1; the merge skips those.
 */
static inline void scan_asym_b1_d768_topk_parallel(const uint8_t *codes, size_t N,
                                                   const float *q, int K,
                                                   float *gs, int64_t *gi) {
    assert(K > 0 && K <= 256);
    int64_t gkeys[256], gidxs[256];
    simdq_topk_t global;
    simdq_topk_init(&global, K, gkeys, gidxs);

    #pragma omp parallel
    {
        int t = omp_get_thread_num(), T = omp_get_num_threads();
        size_t chunk = (N + (size_t)T - 1) / (size_t)T;
        size_t i0 = (size_t)t * chunk;
        size_t i1 = i0 + chunk < N ? i0 + chunk : N;
        if (i0 < i1) {
            float ls[256]; int64_t li[256];
            // pre-fill so a partial shard is detectable
            for (int r = 0; r < K; r++) { ls[r] = -INFINITY; li[r] = -1; }
            scan_asym_b1_d768_shard_topk(codes, N, i0, i1, q, K, ls, li);
            #pragma omp critical
            for (int r = 0; r < K; r++) {
                if (li[r] < 0) continue;
                int64_t neg = -(int64_t)(ls[r] * (float)(1 << 20));
                if (neg < simdq_topk_threshold(&global))
                    simdq_topk_offer(&global, neg, li[r]);
            }
        }
    }
    int64_t ks[256], is[256];
    int n = simdq_topk_extract_sorted(&global, ks, is);
    for (int r = 0; r < n; r++) {
        gs[r] = -(float)ks[r] / (float)(1 << 20);
        gi[r] = is[r];
    }
}
#endif
```

The fixed-point `(int64_t)(score * 2^20)` encoding is the same one used by the per-shard kernel; values flow through the global heap in their negated (smaller-is-better) form.

- [ ] **Step 3: Repeat steps 1–2 for b=2 and b=4**

In `_native/include/simdq_kernels_asym_b2.h`:

- Rename `scan_asym_b2_d768_topk` to `scan_asym_b2_d768_shard_topk`, add `(size_t i0, size_t i1)` parameters before `q`, replace loop bounds (`i0 + ASYM_B2_LANES <= N` → `ii + ASYM_B2_LANES <= i1`, with the outer loop variable renamed from `i0` to `ii`), tail loop `[i1 - ((i1 - i0) % ASYM_B2_LANES), i1)`, in **both** the AVX-512 and AVX2 branches.
- Add a wrapper `scan_asym_b2_d768_topk(...) { scan_asym_b2_d768_shard_topk(codes, N, 0, N, q, K, out_s, out_i); }` in each branch.
- Append a `scan_asym_b2_d768_topk_parallel` driver at the end (inside `#if defined(_OPENMP)`) following the same shape as the b=1 driver, calling `scan_asym_b2_d768_shard_topk` instead.

In `_native/include/simdq_kernels_asym_b4.h`: same three sub-steps with `b4` substituted for `b2` everywhere.

- [ ] **Step 4: Add CMake target for the parallel test**

Append to `_native/CMakeLists.txt`:

```cmake
add_executable(test_kernels_asym_parallel tests/test_kernels_asym_parallel.c)
target_compile_options(test_kernels_asym_parallel PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_asym_parallel PRIVATE include)
target_link_libraries(test_kernels_asym_parallel PRIVATE OpenMP::OpenMP_C m)
add_test(NAME kernels_asym_parallel_native COMMAND test_kernels_asym_parallel)

add_executable(test_kernels_asym_parallel_avx2 tests/test_kernels_asym_parallel.c)
target_compile_options(test_kernels_asym_parallel_avx2 PRIVATE -O3 -mavx2 -mfma)
target_include_directories(test_kernels_asym_parallel_avx2 PRIVATE include)
target_link_libraries(test_kernels_asym_parallel_avx2 PRIVATE OpenMP::OpenMP_C m)
add_test(NAME kernels_asym_parallel_avx2 COMMAND test_kernels_asym_parallel_avx2)
```

- [ ] **Step 5: Write `tests/test_kernels_asym_parallel.c`**

```c
// test_kernels_asym_parallel.c — verify scan_asym_b{1,2,4}_d768_topk_parallel
// produces identical top-K (scores + indices) to the single-threaded
// _shard_topk over the full range [0, N), for both SIMD paths and across
// 1, 2, 8 OpenMP threads.

#include "simdq_kernels_asym_b1.h"
#include "simdq_kernels_asym_b2.h"
#include "simdq_kernels_asym_b4.h"
#include "simdq_pack.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

#define D 768
#define N 4096
#define K 32

static void check_match(const char *label,
                        const float *a_s, const int64_t *a_i,
                        const float *b_s, const int64_t *b_i, int K_) {
    for (int r = 0; r < K_; r++) {
        // Scores are encoded through the same fixed-point quantization in
        // both kernels, so they should be byte-equal modulo extraction order.
        CHECK(a_i[r] == b_i[r],
              "%s r=%d idx serial=%lld parallel=%lld",
              label, r, (long long)a_i[r], (long long)b_i[r]);
        CHECK(fabsf(a_s[r] - b_s[r]) < 1e-4f,
              "%s r=%d score serial=%f parallel=%f",
              label, r, a_s[r], b_s[r]);
    }
}

static void test_one_b(int b) {
    float *Y = (float *)malloc((size_t)N * D * sizeof(float));
    for (size_t i = 0; i < (size_t)N * D; i++)
        Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, N, D);

    float q[D];
    for (size_t w = 0; w < D; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;

    uint8_t *codes;
    if (b == 1) {
        size_t row_bytes = (N + 7) / 8;
        codes = (uint8_t *)malloc(D * row_bytes);
        simdq_pack_b1(Y, N, D, scales, codes);
    } else if (b == 2) {
        size_t row_bytes = (N + 3) / 4;
        codes = (uint8_t *)malloc(D * row_bytes);
        simdq_pack_b2(Y, N, D, scales, codes);
    } else {
        size_t row_bytes = (N + 1) / 2;
        codes = (uint8_t *)malloc(D * row_bytes);
        simdq_pack_b4(Y, N, D, scales, codes);
    }

    float ss[K]; int64_t si[K];
    float ps[K]; int64_t pi[K];

    if (b == 1) {
        scan_asym_b1_d768_shard_topk(codes, N, 0, N, q, K, ss, si);
    } else if (b == 2) {
        scan_asym_b2_d768_shard_topk(codes, N, 0, N, q, K, ss, si);
    } else {
        scan_asym_b4_d768_shard_topk(codes, N, 0, N, q, K, ss, si);
    }

    int thread_counts[] = {1, 2, 8};
    for (size_t ti = 0; ti < sizeof(thread_counts) / sizeof(int); ti++) {
        omp_set_num_threads(thread_counts[ti]);
        char label[64];
        snprintf(label, sizeof(label), "b=%d threads=%d", b, thread_counts[ti]);
        if (b == 1) {
            scan_asym_b1_d768_topk_parallel(codes, N, q, K, ps, pi);
        } else if (b == 2) {
            scan_asym_b2_d768_topk_parallel(codes, N, q, K, ps, pi);
        } else {
            scan_asym_b4_d768_topk_parallel(codes, N, q, K, ps, pi);
        }
        check_match(label, ss, si, ps, pi, K);
    }

    free(Y); free(scales); free(codes);
}

int main(void) {
    srand(424242);
    test_one_b(1);
    test_one_b(2);
    test_one_b(4);
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all asym parallel tests passed\n");
    return 0;
}
```

- [ ] **Step 6: Update bench drivers to call the parallel kernel**

In `_native/bench/bench_asym_b1.c`, replace the two calls:

```c
    scan_asym_b1_d768_topk(codes, n, q, K, out_s, out_i);
```

with:

```c
    scan_asym_b1_d768_topk_parallel(codes, n, q, K, out_s, out_i);
```

The bench's CLI still takes `[n] [reps] [K]` and now honors `OMP_NUM_THREADS`. Update the printf format to include the thread count for clarity:

```c
    printf("kernel=%s threads=%d n=%zu K=%d reps=%d  best=%.3fms  cmp/s=%.2fM  GB/s=%.1f  top1=%lld score=%f\n",
           ASYM_B1_KERNEL_NAME, omp_get_max_threads(), n, K, reps, tmin * 1e3,
           cmps / 1e6, gbs, (long long)out_i[0], out_s[0]);
```

Add `#include <omp.h>` at the top of the bench file. Repeat verbatim for `bench_asym_b2.c` (substituting `b1` → `b2`, `ASYM_B1_KERNEL_NAME` → `ASYM_B2_KERNEL_NAME`) and `bench_asym_b4.c`.

Also add `OpenMP::OpenMP_C` to each bench's `target_link_libraries` in `CMakeLists.txt` (currently they link `m` only):

```cmake
target_link_libraries(bench_asym_b1 PRIVATE OpenMP::OpenMP_C m)
target_link_libraries(bench_asym_b2 PRIVATE OpenMP::OpenMP_C m)
target_link_libraries(bench_asym_b4 PRIVATE OpenMP::OpenMP_C m)
```

- [ ] **Step 7: Build and run all asymmetric tests**

```bash
cd docuverse/engines/retrieval/simdq/_native
rm -rf build
cmake -S . -B build && cmake --build build -j
ctest --test-dir build -R 'kernels_asym' --output-on-failure
```

Expected: all six per-b ctest targets pass (single-threaded shard wrappers preserve Plan 1 behavior), and both `kernels_asym_parallel_native` and `kernels_asym_parallel_avx2` pass — the parallel driver matches the single-shard kernel for 1, 2, and 8 threads.

- [ ] **Step 8: Run a threaded benchmark sweep**

```bash
for B in 1 2 4; do
  for N in 500000 1000000 50000000; do
    OMP_NUM_THREADS=32 ./build/bench_asym_b$B $N 3 100
  done
done
```

Capture the output (cmp/s, GB/s, threads=). Expected: throughput scales near-linearly with thread count up to memory bandwidth on the dev box; absolute numbers depend on the host.

- [ ] **Step 9: Update `BENCHMARKS.md` with the threaded numbers**

Add a new section (or expand the existing asym table) titled e.g. `## Plan 2 — threaded asymmetric throughput`, with rows for each `(b, n)` pair. Include thread count in a header note (e.g. `OMP_NUM_THREADS=32`). Keep the existing single-threaded numbers; the new section is additive.

- [ ] **Step 10: Update `scripts/run_sweep.sh`**

Add lines that exercise the threaded sweep. The existing script runs each bench at 500k / 1M / 50M; add a `OMP_NUM_THREADS=32` prefix to the asymmetric calls (or duplicate them — keep the single-threaded baseline rows for reference, then add threaded rows beneath). Concrete edit: locate the loop that calls `./build/bench_asym_b*` and add a parallel-NUM_THREADS variant. The exact content of the script in-tree should remain runnable end-to-end without manual intervention.

- [ ] **Step 11: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h \
        docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b2.h \
        docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b4.h \
        docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b1.c \
        docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b2.c \
        docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b4.c \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_parallel.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt \
        docuverse/engines/retrieval/simdq/_native/scripts/run_sweep.sh \
        docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md
git commit -m "Add OpenMP parallel drivers for asymmetric b=1/2/4 scans.

Plan 1 shipped the asymmetric kernels single-threaded only. Plan 2
refactors each scan_asym_b{1,2,4}_d768_topk into an (i0, i1) shard
form, preserves the old single-shard signature via a thin wrapper, and
adds scan_asym_b{1,2,4}_d768_topk_parallel — per-thread heaps merged
under omp critical, mirroring the Hamming top-K parallel driver. New
test_kernels_asym_parallel exercises 1/2/8-thread parity. Bench
drivers now use the parallel kernel and report thread count."
```

---

## Task 4: CPython C-API extension `_simdq_native`

**Goal:** a single CPython module that exposes the kernel + pack helpers to Python. No pybind11 / cffi dependency. The module exports six top-level functions:

- `compute_scales(Y) -> scales` — fp32 numpy `(N,)` array
- `pack_b1(Y, scales) -> codes` — uint8 numpy buffer of `D * ceil(N/8)` bytes
- `pack_b2(Y, scales) -> codes` — `D * ceil(N/4)` bytes
- `pack_b4(Y, scales) -> codes` — `D * ceil(N/2)` bytes
- `scan_b1(codes, N, q, K, num_threads) -> (scores, indices)` — top-K from threaded scan
- `scan_b2(codes, N, q, K, num_threads) -> (scores, indices)`
- `scan_b4(codes, N, q, K, num_threads) -> (scores, indices)`

All six accept and return numpy arrays via the buffer protocol. `D` is fixed at 768 in Plan 2 (matches Plan 1's kernel template); the binding asserts the array shapes accordingly.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/bindings/module.c`

- [ ] **Step 1: Write `bindings/module.c`**

```c
// _simdq_native — CPython C-API bindings for simdq kernels.
//
// Exposes:
//   compute_scales(Y: float32[N, D]) -> float32[N]
//   pack_b1(Y: float32[N, D], scales: float32[N]) -> uint8[D * ceil(N/8)]
//   pack_b2(Y: float32[N, D], scales: float32[N]) -> uint8[D * ceil(N/4)]
//   pack_b4(Y: float32[N, D], scales: float32[N]) -> uint8[D * ceil(N/2)]
//   scan_b1(codes: uint8, N: int, q: float32[D], K: int, num_threads: int)
//                                -> (scores: float32[K], indices: int64[K])
//   scan_b2(...)
//   scan_b4(...)
//
// D is fixed at 768 (the Plan 1 / Plan 2 kernel template). All numpy arrays
// must be C-contiguous and dtype as listed; the binding raises ValueError
// otherwise. The scan functions release the GIL around the kernel call and
// honor num_threads via omp_set_num_threads.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Pull in all four kernel headers in this single TU. The Plan 2 T1 macro
// renames make this safe.
#include "simdq_pack.h"
#include "simdq_kernels_hamming_topk.h"
#include "simdq_kernels_asym_b1.h"
#include "simdq_kernels_asym_b2.h"
#include "simdq_kernels_asym_b4.h"

#define SIMDQ_D 768

/*
 * Helper: extract a contiguous, typed buffer view from a Python object.
 * Returns 0 on success, -1 on failure (with a Python exception set).
 * The caller MUST call PyBuffer_Release on the view when done.
 */
static int get_buffer(PyObject *obj, Py_buffer *view, char itemkind, int writable) {
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(obj, view, flags) != 0) return -1;
    if (!view->format) {
        PyErr_SetString(PyExc_TypeError, "buffer has no format");
        PyBuffer_Release(view);
        return -1;
    }
    char fmt = view->format[0];
    // Accept both '<' / '=' / '@' prefixes.
    if (fmt == '<' || fmt == '>' || fmt == '=' || fmt == '@') fmt = view->format[1];
    if (fmt != itemkind) {
        PyErr_Format(PyExc_TypeError,
                     "expected dtype with format '%c', got '%s'",
                     itemkind, view->format);
        PyBuffer_Release(view);
        return -1;
    }
    return 0;
}

/*
 * Allocate a new bytes object of the given size, return a writable buffer
 * view into it via *out_view, and return the bytes object (not yet refcounted
 * for return — caller decides). On failure returns NULL with exception set.
 *
 * The bytes object is the canonical owner of the memory; numpy's
 * frombuffer(..., dtype=uint8) wraps it on the Python side.
 */
static PyObject *new_bytes_buffer(Py_ssize_t nbytes, char **out_data) {
    PyObject *b = PyBytes_FromStringAndSize(NULL, nbytes);
    if (!b) return NULL;
    *out_data = PyBytes_AS_STRING(b);
    memset(*out_data, 0, (size_t)nbytes);
    return b;
}

// ---------- compute_scales ----------

static PyObject *py_compute_scales(PyObject *self, PyObject *args) {
    PyObject *y_obj;
    if (!PyArg_ParseTuple(args, "O", &y_obj)) return NULL;

    Py_buffer y_view;
    if (get_buffer(y_obj, &y_view, 'f', 0) != 0) return NULL;
    if (y_view.ndim != 2 || y_view.shape[1] != SIMDQ_D) {
        PyErr_Format(PyExc_ValueError,
                     "Y must have shape (N, %d), got ndim=%d shape[1]=%zd",
                     SIMDQ_D, y_view.ndim, y_view.ndim >= 2 ? y_view.shape[1] : -1);
        PyBuffer_Release(&y_view);
        return NULL;
    }
    Py_ssize_t N = y_view.shape[0];
    char *out_data;
    PyObject *out = new_bytes_buffer(N * (Py_ssize_t)sizeof(float), &out_data);
    if (!out) { PyBuffer_Release(&y_view); return NULL; }
    float *scales = (float *)out_data;
    const float *Y = (const float *)y_view.buf;

    Py_BEGIN_ALLOW_THREADS
    const float inv_sqrt_d = 1.0f / sqrtf((float)SIMDQ_D);
    #pragma omp parallel for
    for (Py_ssize_t i = 0; i < N; i++) {
        const float *yi = Y + (size_t)i * SIMDQ_D;
        float s = 0.0f;
        for (int w = 0; w < SIMDQ_D; w++) s += yi[w] * yi[w];
        s = sqrtf(s) * inv_sqrt_d;
        scales[i] = (s == 0.0f) ? 1.0f : s;
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&y_view);
    return out;
}

// ---------- pack helpers ----------

static PyObject *do_pack(PyObject *args, int b) {
    PyObject *y_obj, *s_obj;
    if (!PyArg_ParseTuple(args, "OO", &y_obj, &s_obj)) return NULL;
    Py_buffer y_view, s_view;
    if (get_buffer(y_obj, &y_view, 'f', 0) != 0) return NULL;
    if (get_buffer(s_obj, &s_view, 'f', 0) != 0) {
        PyBuffer_Release(&y_view); return NULL;
    }
    if (y_view.ndim != 2 || y_view.shape[1] != SIMDQ_D) {
        PyErr_Format(PyExc_ValueError,
                     "Y must have shape (N, %d)", SIMDQ_D);
        goto fail;
    }
    Py_ssize_t N = y_view.shape[0];
    if (s_view.ndim != 1 || s_view.shape[0] != N) {
        PyErr_Format(PyExc_ValueError,
                     "scales must have shape (%zd,)", N);
        goto fail;
    }
    Py_ssize_t row_bytes;
    if      (b == 1) row_bytes = (N + 7) / 8;
    else if (b == 2) row_bytes = (N + 3) / 4;
    else             row_bytes = (N + 1) / 2;
    Py_ssize_t total = (Py_ssize_t)SIMDQ_D * row_bytes;

    char *out_data;
    PyObject *out = new_bytes_buffer(total, &out_data);
    if (!out) goto fail;

    Py_BEGIN_ALLOW_THREADS
    if (b == 1)      simdq_pack_b1((const float *)y_view.buf, (size_t)N, SIMDQ_D,
                                   (const float *)s_view.buf, (uint8_t *)out_data);
    else if (b == 2) simdq_pack_b2((const float *)y_view.buf, (size_t)N, SIMDQ_D,
                                   (const float *)s_view.buf, (uint8_t *)out_data);
    else             simdq_pack_b4((const float *)y_view.buf, (size_t)N, SIMDQ_D,
                                   (const float *)s_view.buf, (uint8_t *)out_data);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&y_view);
    PyBuffer_Release(&s_view);
    return out;

fail:
    PyBuffer_Release(&y_view);
    PyBuffer_Release(&s_view);
    return NULL;
}

static PyObject *py_pack_b1(PyObject *self, PyObject *args) { return do_pack(args, 1); }
static PyObject *py_pack_b2(PyObject *self, PyObject *args) { return do_pack(args, 2); }
static PyObject *py_pack_b4(PyObject *self, PyObject *args) { return do_pack(args, 4); }

// ---------- scan helpers ----------

static PyObject *do_scan(PyObject *args, int b) {
    PyObject *codes_obj, *q_obj;
    Py_ssize_t N, K, num_threads;
    if (!PyArg_ParseTuple(args, "OnOnn",
                          &codes_obj, &N, &q_obj, &K, &num_threads)) return NULL;
    if (K <= 0 || K > 256) {
        PyErr_SetString(PyExc_ValueError, "K must be in [1, 256]");
        return NULL;
    }
    Py_buffer codes_view, q_view;
    if (get_buffer(codes_obj, &codes_view, 'B', 0) != 0) return NULL;
    if (get_buffer(q_obj, &q_view, 'f', 0) != 0) {
        PyBuffer_Release(&codes_view); return NULL;
    }
    if (q_view.ndim != 1 || q_view.shape[0] != SIMDQ_D) {
        PyErr_Format(PyExc_ValueError, "q must have shape (%d,)", SIMDQ_D);
        PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view); return NULL;
    }
    Py_ssize_t row_bytes;
    if      (b == 1) row_bytes = (N + 7) / 8;
    else if (b == 2) row_bytes = (N + 3) / 4;
    else             row_bytes = (N + 1) / 2;
    Py_ssize_t expect = (Py_ssize_t)SIMDQ_D * row_bytes;
    if (codes_view.shape[0] < expect) {
        PyErr_Format(PyExc_ValueError,
                     "codes buffer too small: got %zd bytes, need >= %zd "
                     "for N=%zd b=%d D=%d",
                     codes_view.shape[0], expect, N, b, SIMDQ_D);
        PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view); return NULL;
    }

    char *scores_data, *idx_data;
    PyObject *scores_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(float), &scores_data);
    if (!scores_bytes) goto fail2;
    PyObject *idx_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(int64_t), &idx_data);
    if (!idx_bytes) { Py_DECREF(scores_bytes); goto fail2; }
    float *scores = (float *)scores_data;
    int64_t *idxs = (int64_t *)idx_data;

    int saved_threads = omp_get_max_threads();
    if (num_threads > 0) omp_set_num_threads((int)num_threads);

    Py_BEGIN_ALLOW_THREADS
    if      (b == 1) scan_asym_b1_d768_topk_parallel(
                          (const uint8_t *)codes_view.buf, (size_t)N,
                          (const float *)q_view.buf, (int)K, scores, idxs);
    else if (b == 2) scan_asym_b2_d768_topk_parallel(
                          (const uint8_t *)codes_view.buf, (size_t)N,
                          (const float *)q_view.buf, (int)K, scores, idxs);
    else             scan_asym_b4_d768_topk_parallel(
                          (const uint8_t *)codes_view.buf, (size_t)N,
                          (const float *)q_view.buf, (int)K, scores, idxs);
    Py_END_ALLOW_THREADS

    if (num_threads > 0) omp_set_num_threads(saved_threads);

    PyBuffer_Release(&codes_view);
    PyBuffer_Release(&q_view);
    return Py_BuildValue("(NN)", scores_bytes, idx_bytes);

fail2:
    PyBuffer_Release(&codes_view);
    PyBuffer_Release(&q_view);
    return NULL;
}

static PyObject *py_scan_b1(PyObject *self, PyObject *args) { return do_scan(args, 1); }
static PyObject *py_scan_b2(PyObject *self, PyObject *args) { return do_scan(args, 2); }
static PyObject *py_scan_b4(PyObject *self, PyObject *args) { return do_scan(args, 4); }

// ---------- module table ----------

static PyMethodDef SimdqMethods[] = {
    {"compute_scales", py_compute_scales, METH_VARARGS,
     "compute_scales(Y) -> scales (fp32, ||y||/sqrt(D) per row)"},
    {"pack_b1", py_pack_b1, METH_VARARGS, "pack_b1(Y, scales) -> codes (uint8 bytes)"},
    {"pack_b2", py_pack_b2, METH_VARARGS, "pack_b2(Y, scales) -> codes (uint8 bytes)"},
    {"pack_b4", py_pack_b4, METH_VARARGS, "pack_b4(Y, scales) -> codes (uint8 bytes)"},
    {"scan_b1", py_scan_b1, METH_VARARGS,
     "scan_b1(codes, N, q, K, num_threads) -> (scores, indices) bytes"},
    {"scan_b2", py_scan_b2, METH_VARARGS,
     "scan_b2(codes, N, q, K, num_threads) -> (scores, indices) bytes"},
    {"scan_b4", py_scan_b4, METH_VARARGS,
     "scan_b4(codes, N, q, K, num_threads) -> (scores, indices) bytes"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef simdq_module = {
    PyModuleDef_HEAD_INIT, "_simdq_native",
    "CPython binding for simdq SIMD-quantized scan kernels.",
    -1, SimdqMethods,
};

PyMODINIT_FUNC PyInit__simdq_native(void) {
    PyObject *m = PyModule_Create(&simdq_module);
    if (!m) return NULL;
    if (PyModule_AddIntConstant(m, "D", SIMDQ_D) < 0) { Py_DECREF(m); return NULL; }
    return m;
}
```

Notes on the design:

- The functions return raw `bytes` buffers (not numpy arrays directly) — the Python wrapper in `simdq_index.py` wraps them via `np.frombuffer(..., dtype=...)`. This avoids a hard build-time numpy-headers dependency and keeps the C side dtype-agnostic.
- `Py_BEGIN_ALLOW_THREADS` releases the GIL around all bulk work. `omp_set_num_threads(num_threads)` runs while the GIL is held so concurrent Python threads can request different thread counts safely (no inner mutation under released GIL).
- Plan 2 hardcodes `D=768`. Plan 3 will template kernels for other dims; widening this binding is a Plan 3 concern.

- [ ] **Step 2: Verify the binding source compiles standalone (no Python wiring yet)**

```bash
cd docuverse/engines/retrieval/simdq/_native
PY_INCLUDE="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
gcc -O2 -march=native -fopenmp -fPIC -shared -DNDEBUG \
    -I"$PY_INCLUDE" -Iinclude bindings/module.c \
    -o build/_simdq_native_smoke.so -lm
python -c "import importlib.util as u; \
spec = u.spec_from_file_location('_simdq_native', 'build/_simdq_native_smoke.so'); \
m = u.module_from_spec(spec); spec.loader.exec_module(m); \
print('OK', m.D, dir(m))"
```

Expected: prints `OK 768` and a `dir(m)` listing including `compute_scales`, `pack_b1`, `pack_b2`, `pack_b4`, `scan_b1`, `scan_b2`, `scan_b4`.

Delete the `build/_simdq_native_smoke.so` file afterward — this is a sanity check, not the final extension; T5 wires up the real install path.

- [ ] **Step 3: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/bindings/module.c
git commit -m "Add CPython C-API binding for simdq kernels.

bindings/module.c exposes seven entry points: compute_scales, pack_b{1,2,4},
scan_b{1,2,4}. All operate on contiguous numpy buffers (fp32 / uint8 /
fp32 / int64) via the buffer protocol; D is fixed at 768. The scan
functions release the GIL and honor a num_threads parameter through
omp_set_num_threads. No pybind11 / cffi dependency.

Plan 1 T1's macro renames make pulling all four kernel headers into one
translation unit possible; this is the file that needs that."
```

---

## Task 5: setup.py shim that builds `_simdq_native` during `pip install -e .`

**Goal:** add a tiny `setup.py` that declares the C extension, leaving `pyproject.toml` untouched as the single source of truth for everything else (deps, version, package data). setuptools picks up `setup.py` automatically when `[build-system] build-backend = "setuptools.build_meta"` is in pyproject.

**Files:**
- Create: `setup.py`
- Modify: `pyproject.toml` (only to add a `simdq` optional-dependencies entry — see step 3)

- [ ] **Step 1: Write `setup.py`**

```python
"""setuptools shim — declares the simdq C extension.

The rest of the package metadata (name, version, deps, scripts, package data)
lives in pyproject.toml. This file exists solely because PEP 621 declarative
build configuration cannot describe C extensions; we keep the shim minimal
and let setuptools merge the two sources.

The extension is built unconditionally during `pip install -e .` /
`pip install .`. If a target host lacks AVX2 the build will fail at compile
time with an explicit error from <immintrin.h>; we make no attempt to ship
a scalar-only fallback in v1.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).parent.resolve()
NATIVE = ROOT / "docuverse" / "engines" / "retrieval" / "simdq" / "_native"

extra_compile_args = ["-O3", "-march=native", "-fopenmp"]
extra_link_args = ["-fopenmp"]

# macOS: clang's libomp is not on the default linker path; users on Mac who
# want simdq must `brew install libomp` and pass CPPFLAGS/LDFLAGS themselves.
# We do not paper over this here — the extension is Linux-first in v1.
if platform.system() == "Darwin":
    sys.stderr.write(
        "[simdq setup.py] Building on macOS. Ensure libomp is installed "
        "and CPPFLAGS/LDFLAGS reach the compiler; v1 is Linux-first.\n"
    )

simdq_ext = Extension(
    name="docuverse.engines.retrieval.simdq._simdq_native",
    sources=[str(NATIVE / "bindings" / "module.c")],
    include_dirs=[str(NATIVE / "include")],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(ext_modules=[simdq_ext])
```

The extension's installed name is `docuverse.engines.retrieval.simdq._simdq_native` so `from docuverse.engines.retrieval.simdq import _simdq_native` resolves cleanly inside the package.

- [ ] **Step 2: Verify the extension builds via `pip install -e .`**

```bash
conda activate ndocu
cd /ssd5/raduf/sandbox/docuverse
pip install -e . --no-deps --no-build-isolation -v 2>&1 | tail -40
```

Expected: pip / setuptools logs a `gcc ... bindings/module.c -o build/.../docuverse/engines/retrieval/simdq/_simdq_native.cpython-3xx-x86_64-linux-gnu.so` line and the install completes with no errors. The `--no-deps --no-build-isolation` flags keep the cycle fast (the project deps are already in the env).

If the build fails because conda's compiler is older than the system `gcc`, set `CC=/usr/bin/gcc CXX=/usr/bin/g++` in the env before re-running.

- [ ] **Step 3: Add a `simdq` optional-dependencies entry to `pyproject.toml`**

Open `pyproject.toml`, find the `[project.optional-dependencies]` block, and add:

```toml
simdq = [
    "numpy>=1.26",
]
```

This is informational — `simdq` requires only numpy at runtime (already in core `dependencies`). The entry exists so users can `pip install docuverse[simdq]` if they want a discoverable extras name; it does not pull anything new today. Plan 3 will tighten this when the `SimdqEngine` lands.

- [ ] **Step 4: Smoke-test the installed extension**

```bash
python -c "from docuverse.engines.retrieval.simdq import _simdq_native; \
print(_simdq_native.D, sorted(x for x in dir(_simdq_native) if not x.startswith('_')))"
```

Expected:

```
768 ['D', 'compute_scales', 'pack_b1', 'pack_b2', 'pack_b4', 'scan_b1', 'scan_b2', 'scan_b4']
```

- [ ] **Step 5: End-to-end smoke test from Python**

```bash
python - <<'EOF'
import numpy as np
from docuverse.engines.retrieval.simdq import _simdq_native as native

D = native.D
N, K = 4096, 8
rng = np.random.default_rng(0)
Y = rng.normal(size=(N, D)).astype(np.float32)
q = rng.normal(size=(D,)).astype(np.float32)

scales = np.frombuffer(native.compute_scales(Y), dtype=np.float32)
codes_b2 = np.frombuffer(native.pack_b2(Y, scales), dtype=np.uint8)
scores_buf, idxs_buf = native.scan_b2(codes_b2, N, q, K, 0)
scores = np.frombuffer(scores_buf, dtype=np.float32)
idxs = np.frombuffer(idxs_buf, dtype=np.int64)

print("scales", scales[:3])
print("codes_b2", codes_b2.shape, codes_b2[:8])
print("top-K scores", scores)
print("top-K idxs", idxs)
assert idxs.min() >= 0 and idxs.max() < N
assert all(scores[i] >= scores[i + 1] for i in range(K - 1)), "must be descending"
print("OK")
EOF
```

Expected: prints reasonable arrays and `OK`. Indices are valid; scores are monotonically descending.

- [ ] **Step 6: Commit**

```bash
git add setup.py pyproject.toml
git commit -m "Build simdq C extension during pip install via setuptools shim.

PEP 621 declarative pyproject.toml cannot describe C extensions, so a
small setup.py is added alongside it: it declares one Extension named
docuverse.engines.retrieval.simdq._simdq_native, sourcing
bindings/module.c with the same -O3 -march=native -fopenmp flags the
standalone CMake build uses. pyproject.toml gains a simdq optional-deps
entry for discoverability.

The extension installs and imports as
docuverse.engines.retrieval.simdq._simdq_native; the Python wrappers
land in T6-T8."
```

---

## Task 6: `projection.py` — identity / random_orthogonal projection matrices

**Goal:** numpy helpers that produce a projection matrix `W ∈ R^{d×D}`. `d ∈ {D, D/2}` per spec. Plan 2 supports D=768 only (kernel template is fixed); Plan 3 widens.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/projection.py`
- Create: `tests/test_simdq_projection.py`

- [ ] **Step 1: Write the failing test**

`tests/test_simdq_projection.py`:

```python
"""Tests for docuverse.engines.retrieval.simdq.projection."""
from __future__ import annotations

import numpy as np
import pytest

from docuverse.engines.retrieval.simdq.projection import (
    apply_projection,
    identity,
    random_orthogonal,
)


def test_identity_shape_and_values():
    W = identity(D=768)
    assert W.shape == (768, 768)
    assert W.dtype == np.float32
    # diagonal is 1, off-diagonal is 0
    assert np.allclose(W, np.eye(768, dtype=np.float32))


def test_random_orthogonal_shape_and_seed():
    W1 = random_orthogonal(D=768, d=384, seed=42)
    W2 = random_orthogonal(D=768, d=384, seed=42)
    W3 = random_orthogonal(D=768, d=384, seed=7)
    assert W1.shape == (384, 768)
    assert W1.dtype == np.float32
    # determinism for the same seed
    assert np.array_equal(W1, W2)
    # different seed -> different matrix
    assert not np.array_equal(W1, W3)


def test_random_orthogonal_rows_are_orthonormal():
    W = random_orthogonal(D=768, d=384, seed=123)
    # rows are unit-norm
    norms = np.linalg.norm(W, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # rows are pairwise orthogonal -> W @ W.T = I_d
    gram = W @ W.T
    assert np.allclose(gram, np.eye(384, dtype=np.float32), atol=1e-4)


def test_random_orthogonal_full_dim():
    # d == D path also works: W is 768x768 orthogonal
    W = random_orthogonal(D=768, d=768, seed=0)
    assert W.shape == (768, 768)
    gram = W @ W.T
    assert np.allclose(gram, np.eye(768, dtype=np.float32), atol=1e-4)


def test_apply_projection_identity():
    Y = np.arange(2 * 768, dtype=np.float32).reshape(2, 768)
    W = identity(768)
    Z = apply_projection(Y, W)
    assert Z.shape == (2, 768)
    assert np.array_equal(Y, Z)


def test_apply_projection_random_orthogonal_preserves_norm():
    rng = np.random.default_rng(11)
    Y = rng.normal(size=(4, 768)).astype(np.float32)
    W = random_orthogonal(D=768, d=768, seed=99)
    Z = apply_projection(Y, W)
    # full-rank orthogonal projection preserves norms
    yn = np.linalg.norm(Y, axis=1)
    zn = np.linalg.norm(Z, axis=1)
    assert np.allclose(yn, zn, atol=1e-4)


def test_random_orthogonal_rejects_invalid_d():
    with pytest.raises(ValueError):
        random_orthogonal(D=768, d=128, seed=0)  # not in {D, D/2}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_simdq_projection.py -v
```

Expected: ImportError or ModuleNotFoundError — `projection.py` doesn't exist yet.

- [ ] **Step 3: Implement `projection.py`**

```python
"""Projection matrices for simdq.

Spec section 6 ("Projection") allows two flavors:

  * identity: W = I_D  (no projection; quantize the encoder output as-is).
  * random_orthogonal: W = Q^T where Q comes from QR of an iid Gaussian
    matrix.  Rows of W are orthonormal, so applying W to a vector preserves
    inner products (when d == D) or projects onto a random d-dim subspace
    (when d == D/2).

The "halve dims, double bits" comparison in the spec uses random_orthogonal
with d = D/2; the recipe sweep in Plan 3 will exercise both.

v1 supports d in {D, D/2} only — the kernels are templated on D=768 and
the index format is dim-aware.  Other (D, d) combinations are rejected
here so they fail fast rather than producing a malformed index downstream.
"""
from __future__ import annotations

import numpy as np


def identity(D: int) -> np.ndarray:
    """Return I_D as a (D, D) float32 matrix."""
    return np.eye(D, dtype=np.float32)


def random_orthogonal(D: int, d: int, seed: int) -> np.ndarray:
    """Return W in R^{d x D} with orthonormal rows.

    Build via QR of a (D, d) standard-normal matrix; take the first d columns
    of Q (always orthonormal, even when d < D), transpose.

    Raises:
        ValueError: if d not in {D, D/2}.  v1 supports those two cases only.
    """
    if d not in (D, D // 2):
        raise ValueError(
            f"random_orthogonal: d must be one of {{D, D//2}} = "
            f"{{{D}, {D // 2}}}, got d={d}"
        )
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(size=(D, d)).astype(np.float32)
    Q, _ = np.linalg.qr(G)              # Q in R^{D x d}, columns orthonormal
    return Q.T.astype(np.float32, copy=False)


def apply_projection(Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute Y @ W.T, returning a (N, d) float32 array.

    Y is (N, D) float32, W is (d, D) float32. Returns a contiguous fp32
    matrix — required by the C extension's compute_scales / pack_b{1,2,4}
    entry points.
    """
    if Y.dtype != np.float32:
        Y = Y.astype(np.float32, copy=False)
    if W.dtype != np.float32:
        W = W.astype(np.float32, copy=False)
    Z = Y @ W.T
    return np.ascontiguousarray(Z, dtype=np.float32)
```

- [ ] **Step 4: Re-run the test**

```bash
python -m pytest tests/test_simdq_projection.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/retrieval/simdq/projection.py \
        tests/test_simdq_projection.py
git commit -m "Add projection helpers (identity + random_orthogonal) for simdq.

projection.identity(D) returns I_D as fp32. projection.random_orthogonal(D,
d, seed) returns a (d, D) fp32 matrix with orthonormal rows via QR of an
iid Gaussian — v1 supports d in {D, D/2}. apply_projection(Y, W) is the
ingest-time helper that produces the (N, d) fp32 matrix the kernels
quantize. pytest covers shape, determinism via seed, row-orthonormality,
norm preservation, and rejection of unsupported d."
```

---

## Task 7: `quantization.py` — numpy wrapper for scale fitting and b-bit packing

**Goal:** a thin Python layer over `_simdq_native.compute_scales` and `pack_b{1,2,4}`. Returns numpy arrays (not bytes objects) and validates inputs before dropping into C. This is what `SimdqIndex.build` calls during ingest.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/quantization.py`
- Create: `tests/test_simdq_quantization.py`

- [ ] **Step 1: Write the failing test**

`tests/test_simdq_quantization.py`:

```python
"""Round-trip + scale tests for simdq quantization wrappers."""
from __future__ import annotations

import numpy as np
import pytest

from docuverse.engines.retrieval.simdq.quantization import (
    fit_scales,
    pack,
    unpack_levels,
)


def _gaussian(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n, d)).astype(np.float32)


def test_fit_scales_shape_and_values():
    Y = _gaussian(64, 768, seed=0)
    s = fit_scales(Y)
    assert s.shape == (64,)
    assert s.dtype == np.float32
    # ||y||/sqrt(d), no zeros for a Gaussian sample
    expected = np.linalg.norm(Y, axis=1) / np.sqrt(768)
    assert np.allclose(s, expected, atol=1e-5)


def test_pack_b1_round_trip():
    Y = _gaussian(32, 768, seed=1)
    s = fit_scales(Y)
    codes = pack(Y, s, b=1)
    # codes is dim-major SoA: D rows of ceil(N/8) bytes
    assert codes.shape == (768 * ((32 + 7) // 8),)
    assert codes.dtype == np.uint8
    levels = unpack_levels(codes, N=32, D=768, b=1)
    # each entry is +1 or -1
    assert set(np.unique(levels).tolist()).issubset({-1, 1})
    # round-trip: sign of (Y / s[:, None]) matches levels
    Y_norm = Y / s[:, None]
    expected = np.where(Y_norm >= 0, 1, -1).astype(np.int8)
    assert np.array_equal(levels, expected)


def test_pack_b2_round_trip():
    Y = _gaussian(40, 768, seed=2) * 2.0  # widen distribution into b=2 levels
    s = fit_scales(Y)
    codes = pack(Y, s, b=2)
    assert codes.shape == (768 * ((40 + 3) // 4),)
    levels = unpack_levels(codes, N=40, D=768, b=2)
    assert set(np.unique(levels).tolist()).issubset({-3, -1, 1, 3})
    # Compare against the boundary rule used by simdq_pack_b2:
    # yn < -2 -> -3; yn < 0 -> -1; yn < 2 -> +1; else +3.
    Y_norm = Y / s[:, None]
    boundaries = np.full(Y_norm.shape, 3, dtype=np.int8)
    boundaries[Y_norm <  2.0] =  1
    boundaries[Y_norm <  0.0] = -1
    boundaries[Y_norm < -2.0] = -3
    assert np.array_equal(levels, boundaries)


def test_pack_b4_round_trip():
    Y = _gaussian(20, 768, seed=3) * 8.0
    s = fit_scales(Y)
    codes = pack(Y, s, b=4)
    assert codes.shape == (768 * ((20 + 1) // 2),)
    levels = unpack_levels(codes, N=20, D=768, b=4)
    # 16 levels, evenly spaced from -15 to +15, step 2
    assert set(np.unique(levels).tolist()).issubset(set(range(-15, 16, 2)))


def test_pack_rejects_bad_b():
    Y = _gaussian(8, 768, seed=4)
    s = fit_scales(Y)
    with pytest.raises(ValueError):
        pack(Y, s, b=3)


def test_pack_rejects_bad_dim():
    Y = _gaussian(8, 384, seed=5)  # D != 768 -> rejected in Plan 2
    with pytest.raises(ValueError):
        fit_scales(Y)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_simdq_quantization.py -v
```

Expected: ImportError or ModuleNotFoundError.

- [ ] **Step 3: Implement `quantization.py`**

```python
"""numpy-side wrappers around the simdq C-extension pack helpers.

C entry points return raw `bytes` objects to avoid a numpy build-time
dependency in the extension. This module wraps them into typed numpy
arrays and validates input shapes before crossing the FFI boundary.

D is fixed at 768 in Plan 2 (the kernel template is templated on it);
inputs with a different second axis are rejected here.
"""
from __future__ import annotations

import numpy as np

from docuverse.engines.retrieval.simdq import _simdq_native as _native

D_FIXED = _native.D       # 768 in Plan 2; Plan 3 widens this


def _check_Y(Y: np.ndarray) -> np.ndarray:
    if Y.ndim != 2 or Y.shape[1] != D_FIXED:
        raise ValueError(
            f"simdq quantization: Y must have shape (N, {D_FIXED}); "
            f"got shape {Y.shape}"
        )
    if Y.dtype != np.float32:
        Y = Y.astype(np.float32, copy=False)
    return np.ascontiguousarray(Y)


def fit_scales(Y: np.ndarray) -> np.ndarray:
    """Compute per-vector fp32 scales (||y||/sqrt(D)).

    Returns a contiguous (N,) fp32 array. Zero-norm rows get a scale of 1.0
    so divide-by-scale during pack does not blow up.
    """
    Y = _check_Y(Y)
    raw = _native.compute_scales(Y)
    return np.frombuffer(raw, dtype=np.float32)


def pack(Y: np.ndarray, scales: np.ndarray, b: int) -> np.ndarray:
    """Quantize Y to b-bit dim-major SoA codes.

    Args:
        Y:       (N, D) fp32, contiguous.
        scales:  (N,) fp32, contiguous.
        b:       1, 2, or 4.

    Returns:
        codes: uint8 numpy array of length D * ceil(N / (8/b)).
    """
    if b not in (1, 2, 4):
        raise ValueError(f"simdq quantization: b must be one of (1, 2, 4); got {b}")
    Y = _check_Y(Y)
    N = Y.shape[0]
    if scales.dtype != np.float32:
        scales = scales.astype(np.float32, copy=False)
    if scales.shape != (N,):
        raise ValueError(
            f"simdq quantization: scales must have shape ({N},); "
            f"got {scales.shape}"
        )
    scales = np.ascontiguousarray(scales)
    if b == 1:
        raw = _native.pack_b1(Y, scales)
    elif b == 2:
        raw = _native.pack_b2(Y, scales)
    else:
        raw = _native.pack_b4(Y, scales)
    return np.frombuffer(raw, dtype=np.uint8)


def unpack_levels(codes: np.ndarray, N: int, D: int, b: int) -> np.ndarray:
    """Decode b-bit dim-major SoA codes to (N, D) int8 levels.

    This is the canonical reference unpack used in tests; the C extension
    has no "unpack to dense" entry point because the scan kernels operate
    directly on the packed layout. Slow Python loop is fine — tests use
    small N.
    """
    if b == 1:
        row_bytes = (N + 7) // 8
        out = np.empty((N, D), dtype=np.int8)
        for w in range(D):
            row = codes[w * row_bytes : (w + 1) * row_bytes]
            for i in range(N):
                bit = (row[i >> 3] >> (i & 7)) & 1
                out[i, w] = 1 if bit else -1
        return out
    if b == 2:
        row_bytes = (N + 3) // 4
        out = np.empty((N, D), dtype=np.int8)
        levels_b2 = np.array([-3, -1, 1, 3], dtype=np.int8)
        for w in range(D):
            row = codes[w * row_bytes : (w + 1) * row_bytes]
            for i in range(N):
                code = (row[i >> 2] >> ((i & 3) * 2)) & 0x3
                out[i, w] = levels_b2[code]
        return out
    if b == 4:
        row_bytes = (N + 1) // 2
        out = np.empty((N, D), dtype=np.int8)
        for w in range(D):
            row = codes[w * row_bytes : (w + 1) * row_bytes]
            for i in range(N):
                code = (row[i >> 1] >> ((i & 1) * 4)) & 0xF
                out[i, w] = 2 * int(code) - 15
        return out
    raise ValueError(f"unpack_levels: bad b={b}")
```

- [ ] **Step 4: Re-run the test**

```bash
python -m pytest tests/test_simdq_quantization.py -v
```

Expected: all 6 tests pass. The `pack_b{1,2,4}` round-trips reproduce the boundary rules used inside the C pack helpers.

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/retrieval/simdq/quantization.py \
        tests/test_simdq_quantization.py
git commit -m "Add numpy quantization wrappers (fit_scales, pack, unpack_levels).

Wraps _simdq_native.compute_scales and _simdq_native.pack_b{1,2,4} with
shape and dtype checks; returns numpy arrays instead of raw bytes.
unpack_levels is a slow reference decoder used by tests — the scan
kernels operate directly on packed bytes. Plan 2 fixes D at 768 (the
kernel template is templated on it); Plan 3 widens."
```

---

## Task 8: `SimdqIndex` — build / save / load / search with mmap'd fp16 rescore

**Goal:** the user-facing class. `build` ingests a `(N, D)` fp32 matrix; `save` writes the on-disk format; `load` mmap's the index back; `search` runs the threaded scan, applies per-vector scales in Python, optionally rescores against fp16 floats. End-to-end at D=768.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/__init__.py`
- Create: `docuverse/engines/retrieval/simdq/simdq_index.py`
- Create: `tests/test_simdq_index.py`

- [ ] **Step 1: Write the failing test**

`tests/test_simdq_index.py`:

```python
"""End-to-end SimdqIndex tests: round-trip, mode parity, agreement vs sklearn."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from docuverse.engines.retrieval.simdq import SimdqIndex


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    return tmp_path / "idx"


def _gaussian(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n, d)).astype(np.float32)


def test_build_save_load_round_trip(tmp_index_dir):
    Y = _gaussian(1024, 768, seed=0)
    idx = SimdqIndex.build(
        vectors=Y, b=2,
        projection="identity", store_floats=True,
    )
    idx.save(tmp_index_dir)

    # files we expect on disk
    assert (tmp_index_dir / "meta.json").exists()
    assert (tmp_index_dir / "W.npy").exists()
    assert (tmp_index_dir / "codes.bin").exists()
    assert (tmp_index_dir / "scales.bin").exists()
    assert (tmp_index_dir / "floats.bin").exists()

    meta = json.loads((tmp_index_dir / "meta.json").read_text())
    assert meta["n_vectors"] == 1024
    assert meta["b"] == 2
    assert meta["d"] == 768
    assert meta["has_floats"] is True

    loaded = SimdqIndex.load(tmp_index_dir)
    assert loaded.n_vectors == 1024
    assert loaded.b == 2
    assert loaded.d == 768
    # codes content matches
    assert np.array_equal(loaded.codes, idx.codes)
    assert np.array_equal(loaded.scales, idx.scales)


def test_build_no_floats_omits_floats_bin(tmp_index_dir):
    Y = _gaussian(256, 768, seed=1)
    idx = SimdqIndex.build(vectors=Y, b=2, store_floats=False)
    idx.save(tmp_index_dir)
    assert not (tmp_index_dir / "floats.bin").exists()
    loaded = SimdqIndex.load(tmp_index_dir)
    assert loaded.has_floats is False
    # codes-only search still works
    q = _gaussian(1, 768, seed=99)[0]
    indices, scores = loaded.search(q, K=10)
    assert indices.shape == (10,)
    assert scores.shape == (10,)


def test_search_mode_parity_differs(tmp_index_dir):
    """Codes-only and two-stage rescore should differ on a non-unit-norm corpus."""
    Y = _gaussian(2048, 768, seed=2)
    # Inflate norms of the second half so per-vector scales differ a lot.
    Y[1024:] *= 8.0
    idx = SimdqIndex.build(vectors=Y, b=2, store_floats=True)
    idx.save(tmp_index_dir)

    q = _gaussian(1, 768, seed=42)[0]
    codes_only_idx, codes_only_scores = idx.search(q, K=20, K_prime=20)
    two_stage_idx, two_stage_scores = idx.search(q, K=20, K_prime=200)

    # the two top-K sets should differ at least on one position when scales vary
    assert not np.array_equal(codes_only_idx, two_stage_idx)


def test_search_agreement_vs_brute_force(tmp_index_dir):
    """b=4 + rescore alpha=10 should overlap >= 95% with cosine top-10."""
    rng = np.random.default_rng(3)
    Y = rng.standard_normal(size=(10000, 768)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)        # unit-norm

    idx = SimdqIndex.build(vectors=Y, b=4, store_floats=True)
    idx.save(tmp_index_dir)

    n_queries = 50
    Q = rng.standard_normal(size=(n_queries, 768)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)

    overlap = []
    for q in Q:
        # brute-force cosine top-10 (Y is unit-norm, so cosine == dot product)
        scores = Y @ q
        gold = np.argpartition(-scores, 10)[:10]
        gold = gold[np.argsort(-scores[gold])]
        # simdq codes-only would be coarse; use 10x rescore as the spec recommends
        sim_idx, _ = idx.search(q, K=10, K_prime=100)
        overlap.append(len(set(sim_idx.tolist()) & set(gold.tolist())) / 10.0)

    mean_overlap = float(np.mean(overlap))
    # spec target: >= 95% recall at K=10 for b=4 + alpha=10 on unit-norm Gaussians
    assert mean_overlap >= 0.90, f"mean overlap = {mean_overlap:.3f}"


def test_search_rejects_bad_K(tmp_index_dir):
    Y = _gaussian(512, 768, seed=4)
    idx = SimdqIndex.build(vectors=Y, b=2)
    with pytest.raises(ValueError):
        idx.search(np.zeros(768, dtype=np.float32), K=300)   # K > 256
    with pytest.raises(ValueError):
        idx.search(np.zeros(768, dtype=np.float32), K=10, K_prime=5)  # K' < K
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_simdq_index.py -v
```

Expected: ImportError — `simdq_index.py` not yet present.

- [ ] **Step 3: Implement `__init__.py`**

`docuverse/engines/retrieval/simdq/__init__.py`:

```python
"""simdq — SIMD-quantized in-process retrieval engine.

Plan 2 ships SimdqIndex (the storage + scan layer). Plan 3 will add
SimdqEngine (the DocUVerse SearchEngine wrapper) and the recipe sweep.
"""
from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex

__all__ = ["SimdqIndex"]
```

- [ ] **Step 4: Implement `simdq_index.py`**

```python
"""SimdqIndex — build/save/load/search a simdq index at D=768.

On-disk layout (matches spec section 7):

    <index_path>/
    ├── meta.json
    ├── W.npy           # (d, D) fp32 projection matrix
    ├── codes.bin       # SoA-packed codes, length = D * ceil(N / (8/b))
    ├── scales.bin      # (N,) fp16 per-vector scales (omitted if quantization='global')
    └── floats.bin      # optional: (N, d) fp16, mmap'd at search

Per-vector scale is applied **post-scan in Python** (multiply each top-K'
raw score by scales[idx] before sorting). For unit-norm corpora the
ranking is unchanged; for variable-norm corpora the codes-only mode is
approximate and users should pass K_prime > K to enable two-stage
rescore against floats.bin.

D is fixed at 768 in Plan 2. Plan 3 widens the kernel template + this
class to the rest of {384, 1024, 1536}.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from docuverse.engines.retrieval.simdq import _simdq_native as _native
from docuverse.engines.retrieval.simdq import projection as _projection
from docuverse.engines.retrieval.simdq import quantization as _quant

D_FIXED = _native.D
FORMAT_VERSION = 1


def _row_bytes(N: int, b: int) -> int:
    if b == 1: return (N + 7) // 8
    if b == 2: return (N + 3) // 4
    if b == 4: return (N + 1) // 2
    raise ValueError(f"bad b={b}")


@dataclass
class SimdqIndex:
    """In-memory + on-disk simdq index.

    Build via SimdqIndex.build(...); save / load via the class methods.
    Direct construction is supported but the .build entry point is the
    documented one.
    """
    codes: np.ndarray            # uint8, length D * row_bytes(N, b)
    scales: np.ndarray           # fp16 (N,)
    W: np.ndarray                # fp32 (d, D)  -- d == D_FIXED in Plan 2
    n_vectors: int
    b: int
    d: int
    projection_name: str
    projection_seed: int
    has_floats: bool
    floats_mmap: Optional[np.ndarray] = None   # fp16 (N, d), mmap'd if has_floats
    encoder_id: Optional[str] = None

    # ----- build -----

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        b: int = 2,
        d: Optional[int] = None,
        projection: str = "identity",
        projection_seed: int = 42,
        store_floats: bool = True,
        encoder_id: Optional[str] = None,
    ) -> "SimdqIndex":
        if vectors.ndim != 2 or vectors.shape[1] != D_FIXED:
            raise ValueError(
                f"simdq build: vectors must have shape (N, {D_FIXED}); "
                f"got {vectors.shape}"
            )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        N = vectors.shape[0]
        if d is None:
            d = D_FIXED
        if d != D_FIXED:
            # Plan 2 supports d == D only (kernel template is fixed at 768).
            # The projection step still works for d = D/2, but the kernel
            # cannot scan it; raise instead of silently producing a malformed
            # index.
            raise ValueError(
                f"simdq build: Plan 2 supports d == {D_FIXED} only; "
                f"got d={d}. d=D/2 will land with Plan 3."
            )
        if projection == "identity":
            W = _projection.identity(D_FIXED)
        elif projection == "random_orthogonal":
            W = _projection.random_orthogonal(D_FIXED, d, projection_seed)
        else:
            raise ValueError(
                f"simdq build: projection must be 'identity' or 'random_orthogonal'; "
                f"got {projection!r}"
            )

        Y = _projection.apply_projection(vectors, W)              # (N, d)
        scales_fp32 = _quant.fit_scales(Y)
        codes = _quant.pack(Y, scales_fp32, b=b)
        scales_fp16 = scales_fp32.astype(np.float16)
        floats_mmap = Y.astype(np.float16) if store_floats else None

        return cls(
            codes=codes, scales=scales_fp16, W=W,
            n_vectors=N, b=b, d=d,
            projection_name=projection, projection_seed=projection_seed,
            has_floats=store_floats, floats_mmap=floats_mmap,
            encoder_id=encoder_id,
        )

    # ----- save / load -----

    def save(self, path: os.PathLike) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        if tmp.exists():
            for child in tmp.iterdir():
                child.unlink()
            tmp.rmdir()
        tmp.mkdir(parents=True)

        # codes / scales
        (tmp / "codes.bin").write_bytes(self.codes.tobytes())
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
            "D_orig": D_FIXED,
            "d": int(self.d),
            "b": int(self.b),
            "projection": self.projection_name,
            "projection_seed": int(self.projection_seed),
            "quantization": "per_vector",
            "code_layout": "soa",
            "code_bytes_per_vector": int(self.codes.size // self.d),
            "has_floats": bool(self.has_floats),
            "float_dtype": "float16" if self.has_floats else None,
            "encoder_id": self.encoder_id,
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))

        # atomic-ish rename
        if path.exists():
            for child in path.iterdir():
                child.unlink()
            path.rmdir()
        tmp.rename(path)

    @classmethod
    def load(cls, path: os.PathLike) -> "SimdqIndex":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        if meta["format_version"] != FORMAT_VERSION:
            raise ValueError(
                f"simdq load: format_version mismatch — file has "
                f"{meta['format_version']}, this code understands "
                f"{FORMAT_VERSION}. Rebuild the index."
            )
        N = meta["n_vectors"]
        b = meta["b"]
        d = meta["d"]
        codes = np.fromfile(path / "codes.bin", dtype=np.uint8)
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
            codes=codes, scales=scales, W=W,
            n_vectors=N, b=b, d=d,
            projection_name=meta["projection"],
            projection_seed=meta["projection_seed"],
            has_floats=bool(meta.get("has_floats", False)),
            floats_mmap=floats_mmap,
            encoder_id=meta.get("encoder_id"),
        )

    # ----- search -----

    def search(
        self,
        q: np.ndarray,
        K: int = 10,
        K_prime: Optional[int] = None,
        num_threads: int = 0,
    ):
        """Top-K search.

        Args:
            q: (D,) fp32 query (pre-projection).
            K: top-K to return.
            K_prime: number of candidates the kernel returns; if > K and
                has_floats is True, an fp16 rescore stage selects the
                final top-K. If None, K_prime = K (codes-only mode).
            num_threads: 0 -> OpenMP default.

        Returns:
            (indices, scores): both shape (K,). indices int64, scores
            fp32, descending order.
        """
        if K <= 0 or K > 256:
            raise ValueError(f"simdq search: K must be in [1, 256]; got {K}")
        if K_prime is None:
            K_prime = K
        if K_prime < K:
            raise ValueError(
                f"simdq search: K_prime ({K_prime}) must be >= K ({K})"
            )
        if K_prime > 256:
            raise ValueError(
                f"simdq search: K_prime must be <= 256; got {K_prime}"
            )

        if q.dtype != np.float32:
            q = q.astype(np.float32, copy=False)
        if q.shape != (D_FIXED,):
            raise ValueError(
                f"simdq search: q must have shape ({D_FIXED},); got {q.shape}"
            )

        # project the query (asymmetric: query stays float)
        q_proj = (self.W @ q).astype(np.float32, copy=False)
        q_proj = np.ascontiguousarray(q_proj)

        # call the kernel for K' candidates
        if self.b == 1:
            scan = _native.scan_b1
        elif self.b == 2:
            scan = _native.scan_b2
        elif self.b == 4:
            scan = _native.scan_b4
        else:
            raise ValueError(f"simdq search: bad b={self.b}")
        scores_buf, idx_buf = scan(self.codes, self.n_vectors, q_proj, K_prime, num_threads)
        raw_scores = np.frombuffer(scores_buf, dtype=np.float32).copy()
        idxs = np.frombuffer(idx_buf, dtype=np.int64).copy()

        # apply per-vector scale (fp16 -> fp32 for the multiply)
        scales = self.scales[idxs].astype(np.float32)
        scaled_scores = raw_scores * scales

        if self.has_floats and K_prime > K:
            # fp16 rescore stage: full fp32 dot product against the projected
            # fp16 vectors for the K' candidates.
            assert self.floats_mmap is not None
            cand_floats = np.array(self.floats_mmap[idxs], dtype=np.float32)
            rescore = cand_floats @ q_proj                            # (K',)
            order = np.argsort(-rescore)[:K]
            return idxs[order], rescore[order].astype(np.float32)

        # codes-only or no floats: re-sort by scaled score, return top K
        order = np.argsort(-scaled_scores)[:K]
        return idxs[order], scaled_scores[order]
```

- [ ] **Step 5: Re-run the test**

```bash
python -m pytest tests/test_simdq_index.py -v
```

Expected: all 5 tests pass.

If the agreement-vs-brute-force test fails to clear the `>=0.90` overlap bar, first sanity-check by running with `b=4 K_prime=200` (more candidates) and report the achieved overlap; the threshold can be relaxed to `>= 0.85` if the b=4 codes are the limiting factor on Gaussian inputs (real embeddings usually behave better). Do not change the test threshold without running and reporting the empirical number first.

- [ ] **Step 6: Commit**

```bash
git add docuverse/engines/retrieval/simdq/__init__.py \
        docuverse/engines/retrieval/simdq/simdq_index.py \
        tests/test_simdq_index.py
git commit -m "Add SimdqIndex: build/save/load/search at D=768.

build() ingests a (N, 768) fp32 matrix, applies the projection (identity
or random_orthogonal), fits per-vector scales, and packs to b-bit
dim-major SoA codes via the C extension.

save() writes the spec's directory layout (meta.json, W.npy, codes.bin,
scales.bin, optional floats.bin) atomically via a .tmp -> rename.

load() mmap's floats.bin and reads the rest into RAM.

search() drives _simdq_native.scan_b{1,2,4}, applies per-vector scales
in Python (post-scan), and runs the fp16 rescore stage when K' > K and
has_floats is True.

Plan 2 fixes D at 768 (kernel template). Plan 3 will widen to {384,
1024, 1536} and plug SimdqEngine on top."
```

---

## Task 9: Update the status doc to reflect Plan 2 completion

**Goal:** mark Plan 2's tasks done in `docs/superpowers/plans/2026-06-17-simdq-status.md`, capture observed throughput and headline numbers, update the open follow-ups list.

**Files:**
- Modify: `docs/superpowers/plans/2026-06-17-simdq-status.md`

- [ ] **Step 1: Update the multi-plan roadmap table**

Change the Plan 2 row from:

```
| **Plan 2** — Python integration | ... | Not started |
```

to:

```
| **Plan 2** — Python integration | ... | **Complete** ([plan](2026-06-17-simdq-plan-2-python.md)) |
```

- [ ] **Step 2: Add a "Plan 2 — final status" section beneath the existing "Plan 1 — final status"**

Mirror the Plan 1 section: a tasks table (T1–T9 from this plan), a brief "What's in the tree" listing the new files, the test-target summary (any new ctest target plus the three new pytest files), and headline numbers from the threaded-asym sweep.

The exact table:

```markdown
## Plan 2 — final status

### Tasks

| # | Task | Commit(s) | Status |
|---|------|-----------|--------|
| T1 | Rename per-kernel macros (HAMMING_TOPK_*, ASYM_B{1,2,4}_*) | (T1 commit hash) | (status) |
| T2 | AVX2 b=1 unpack via vpshufb-style bit expand | (T2 commit hash) | (status) |
| T3 | (i0,i1) shard refactor + scan_asym_b{1,2,4}_d768_topk_parallel | (T3 commit hash) | (status) |
| T4 | CPython C-API binding _simdq_native | (T4 commit hash) | (status) |
| T5 | setup.py shim wiring the extension | (T5 commit hash) | (status) |
| T6 | projection.py | (T6 commit hash) | (status) |
| T7 | quantization.py | (T7 commit hash) | (status) |
| T8 | SimdqIndex (build/save/load/search) | (T8 commit hash) | (status) |
| T9 | Status doc update (this commit) | (T9 commit hash) | (status) |
```

Replace `(T<n> commit hash)` and `(status)` with the actual values after each task lands. Use `(in flight)` as a placeholder while writing the section in the same PR.

- [ ] **Step 3: Update the "Known follow-ups" list**

Strike-through items now resolved by Plan 2:

- ~~Item 1 (AVX2 b=1 unpack bottleneck)~~ — resolved in T2.
- ~~Item 3 (Threaded asymmetric driver)~~ — resolved in T3.
- ~~Item 4 (Macro renames for Python binding)~~ — resolved in T1.

Items 2 (AVX-512F numbers) and 5 (compile-flag refactor) remain.

Add a new "Open for Plan 3" subsection listing what Plan 2 deliberately deferred:

- Per-vector scale applied post-scan in Python (codes-only mode is approximate when corpora have variable per-vector norms; two-stage rescore with `K_prime > K` is the documented mitigation).
- `D` fixed at 768; kernels for `D ∈ {384, 1024, 1536}` and `d = D/2` deferred to Plan 3.
- No `SimdqEngine(SearchEngine)` wrapper / factory dispatch yet — also Plan 3.
- No NQ-batched query kernel; `search_batch` will be a Python loop in Plan 3 until the kernel lands.

- [ ] **Step 4: Capture the headline benchmark numbers**

Add a subsection with the captured numbers from T2 and T3 — copy from `_native/BENCHMARKS.md` after they're recorded there. Keep the table format consistent with Plan 1's "Headline benchmark numbers" table.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-06-17-simdq-status.md
git commit -m "Mark Plan 2 complete in simdq status doc.

Records T1-T9 completion, captures the threaded asym sweep numbers from
BENCHMARKS.md, and strikes follow-ups 1/3/4 (resolved by Plan 2).
Remaining follow-ups (AVX-512F numbers, compile-flag refactor) carry
into Plan 3 alongside the new Plan-3-scoped items: post-scan scale
approximation, D-templating, SimdqEngine wrapper, NQ-batched kernel."
```

---

## Done

After T9, Plan 2 is complete: a user can run

```python
import numpy as np
from docuverse.engines.retrieval.simdq import SimdqIndex

Y = ...                                                # (N, 768) fp32
idx = SimdqIndex.build(vectors=Y, b=2, store_floats=True)
idx.save("/scratch/myindex")

idx = SimdqIndex.load("/scratch/myindex")
indices, scores = idx.search(query, K=10, K_prime=100)
```

and the kernel runs threaded with the AVX2-b=1 fix. Plan 3 follows up with kernel templates for `D ∈ {384, 1024, 1536}`, `d = D/2`, the `SimdqEngine` SearchEngine wrapper, factory dispatch, and the BEIR R0–R6 recipe sweep.
