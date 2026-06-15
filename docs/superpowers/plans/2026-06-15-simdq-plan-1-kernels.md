# `simdq` Plan 1 — Kernels & standalone benchmarks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the 1-bit Hamming kernel work in-tree under DocUVerse, generalize it from top-1 to top-K, and add asymmetric scan kernels at `b ∈ {1, 2, 4}` for `D=768` — all standalone C, no Python yet.

**Architecture:** Port the `embedding_hashes` C/SIMD code into `docuverse/engines/retrieval/simdq/_native/`, extend the SoA-layout shard/merge pattern with a per-thread bounded max-heap top-K, and add three new asymmetric kernels (`b=1, 2, 4`) doing float×int FMA against packed DB codes. Both AVX-512F and AVX2 fallback paths. CMake builds standalone benchmark binaries + a `ctest` suite, with **no Python dependency in this plan**.

**Tech Stack:** C11, AVX-512F (preferred) + AVX2 fallback, OpenMP, CMake.

**Source of truth for the port:** `/home/raduf/sandbox2/embedding_hashes/`. We copy and rename, not git-import. The existing `embedding_hashes` repo stays untouched.

---

## File structure

| Path under `docuverse/engines/retrieval/simdq/_native/` | Purpose | Created in task |
|---|---|---|
| `CMakeLists.txt` | Standalone build + ctest registry | T1 |
| `include/simdq_common.h` | RNG, alloc, `fill_soa_parallel`, `WORDS=12` | T1 (port) |
| `include/simdq_kernels_hamming.h` | Hamming `scan_shard`, `scan_batch_parallel` (top-1) | T1 (port) |
| `bench/bench_hamming_top1.c` | Existing `hb5` shape, renamed | T1 (port) |
| `tests/test_kernels_hamming.c` | Existing `test_kernels.c`, renamed | T1 (port) |
| `include/simdq_topk.h` | Bounded max-heap data structure | T2 |
| `tests/test_topk.c` | Heap unit tests | T2 |
| `include/simdq_kernels_hamming_topk.h` | Hamming top-K kernels (both SIMD paths) | T3 |
| `tests/test_kernels_hamming_topk.c` | Top-K Hamming correctness | T3 |
| `bench/bench_hamming.c` | Top-K Hamming benchmark | T4 |
| `include/simdq_pack.h` | b=1/b=2/b=4 SoA pack/unpack helpers | T5, T8, T11 |
| `include/simdq_kernels_asym_b1.h` | b=1 asymmetric, both SIMD paths | T6 |
| `tests/test_kernels_asym_b1.c` | b=1 correctness | T6 |
| `bench/bench_asym_b1.c` | b=1 benchmark | T7 |
| `include/simdq_kernels_asym_b2.h` | b=2 asymmetric, both SIMD paths | T9 |
| `tests/test_kernels_asym_b2.c` | b=2 correctness | T9 |
| `bench/bench_asym_b2.c` | b=2 benchmark | T10 |
| `include/simdq_kernels_asym_b4.h` | b=4 asymmetric, both SIMD paths | T12 |
| `tests/test_kernels_asym_b4.c` | b=4 correctness | T12 |
| `bench/bench_asym_b4.c` | b=4 benchmark | T13 |
| `scripts/run_sweep.sh` | Sweep all benches over canned sizes | T14 |
| `BENCHMARKS.md` | Captured throughput numbers from a sweep run | T14 |

**Idiom:** following the existing `embedding_hashes` style, kernels live in `.h` headers with `static inline` functions; `.c` files in `bench/` and `tests/` are thin drivers. No separate `.c` translation units for the kernels.

---

## Task 1: Port the existing Hamming kernel into `_native/` and verify it still works

**Goal:** establish the new directory, port the code with renamed include guards/symbols, and confirm the existing tests + benches pass with no behavior change.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/CMakeLists.txt`
- Create: `docuverse/engines/retrieval/simdq/_native/include/simdq_common.h` (from `hamming_common.h`)
- Create: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming.h` (from `hamming_kernels.h`)
- Create: `docuverse/engines/retrieval/simdq/_native/bench/bench_hamming_top1.c` (from `src/hamming_bench_v5.c`)
- Create: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming.c` (from `tests/test_kernels.c`)

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p docuverse/engines/retrieval/simdq/_native/{include,bench,tests,scripts}
```

- [ ] **Step 2: Copy and rename `hamming_common.h` → `include/simdq_common.h`**

Copy `/home/raduf/sandbox2/embedding_hashes/src/hamming_common.h` to `docuverse/engines/retrieval/simdq/_native/include/simdq_common.h` verbatim, then update the file header to:

```c
// simdq_common.h — shared helpers for SIMD-quantized scan kernels.
//
// Each binary code is 768 bits = WORDS x uint64_t. Two layouts appear across
// the kernels:
//   AoS: db[i*WORDS + w] = word w of code i  — cache-friendly per code
//   SoA: dbT[w*n + i]    = word w of code i  — cache-friendly across codes;
//        one wide SIMD load grabs word w of several consecutive codes.
```

No other changes — `WORDS=12`, `NQ`, all helpers stay identical.

- [ ] **Step 3: Copy and rename `hamming_kernels.h` → `include/simdq_kernels_hamming.h`**

Copy `/home/raduf/sandbox2/embedding_hashes/src/hamming_kernels.h` verbatim, then change the include line to refer to the new header:

```c
#include "simdq_common.h"
```

(was: `#include "hamming_common.h"`)

No other changes.

- [ ] **Step 4: Copy `hamming_bench_v5.c` → `bench/bench_hamming_top1.c`**

Copy `/home/raduf/sandbox2/embedding_hashes/src/hamming_bench_v5.c` to the new path. Update the includes:

```c
#include "simdq_kernels_hamming.h"
```

(was: `#include "hamming_kernels.h"`)

No other changes — this is the existing `hb5` binary, just renamed.

- [ ] **Step 5: Copy `tests/test_kernels.c` → `tests/test_kernels_hamming.c`**

Copy verbatim, update the include:

```c
#include "simdq_kernels_hamming.h"
```

No other changes.

- [ ] **Step 6: Write `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.16)
project(simdq_native C)

set(CMAKE_C_STANDARD 11)
find_package(OpenMP REQUIRED)

set(COMMON_FLAGS -O3 -march=native)
set(UNROLL_FLAGS ${COMMON_FLAGS} -funroll-loops)

# ---- benchmark binaries ----

add_executable(bench_hamming_top1 bench/bench_hamming_top1.c)
target_compile_options(bench_hamming_top1 PRIVATE ${COMMON_FLAGS})
target_include_directories(bench_hamming_top1 PRIVATE include)
target_link_libraries(bench_hamming_top1 PRIVATE OpenMP::OpenMP_C)

# ---- tests (ctest) ----
enable_testing()

# native path: AVX-512 VPOPCNTDQ kernels on capable hardware
add_executable(test_kernels_hamming tests/test_kernels_hamming.c)
target_compile_options(test_kernels_hamming PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_hamming PRIVATE include)
target_link_libraries(test_kernels_hamming PRIVATE OpenMP::OpenMP_C)
add_test(NAME kernels_hamming_native COMMAND test_kernels_hamming)

# AVX2 nibble-LUT path, forced even on AVX-512 hardware
add_executable(test_kernels_hamming_avx2 tests/test_kernels_hamming.c)
target_compile_options(test_kernels_hamming_avx2 PRIVATE -O3 -mavx2 -mpopcnt)
target_include_directories(test_kernels_hamming_avx2 PRIVATE include)
target_link_libraries(test_kernels_hamming_avx2 PRIVATE OpenMP::OpenMP_C)
add_test(NAME kernels_hamming_avx2 COMMAND test_kernels_hamming_avx2)
```

- [ ] **Step 7: Build and run the tests + benchmark**

```bash
cd docuverse/engines/retrieval/simdq/_native
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
OMP_NUM_THREADS=8 ./build/bench_hamming_top1 1000000 5
```

Expected:
- Both ctest targets pass: `kernels_hamming_native` and `kernels_hamming_avx2`.
- `bench_hamming_top1` produces output similar to the existing `hb5 1000000 5` run (cmp/s ≥ 2 GB at 8 threads on the dev box).

- [ ] **Step 8: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/
git commit -m "Port hb5 Hamming kernel work into docuverse/.../simdq/_native/.

Copies hamming_common.h / hamming_kernels.h / hb5 / test_kernels.c
into the new in-tree location with renamed include guards. Behavior is
unchanged; the existing top-1 tests and benchmark still pass.

Foundation for the simdq retrieval engine; subsequent commits add
top-K, asymmetric multi-bit scans, and Python bindings."
```

---

## Task 2: Bounded max-heap top-K data structure

**Goal:** a small, branch-light top-K data structure to plug into the scan kernels. Holds the K *smallest* (distance, index) pairs seen so far for symmetric scans, and the K *largest* (score, index) pairs for asymmetric scans. We use the "bounded max-heap" pattern: heap of size K, root = worst kept entry; new entry replaces the root iff it beats it.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/include/simdq_topk.h`
- Create: `docuverse/engines/retrieval/simdq/_native/tests/test_topk.c`

- [ ] **Step 1: Write the failing test**

`tests/test_topk.c`:

```c
// test_topk.c — bounded max-heap of (key, index) pairs.
//
// We test both the "min top-K" usage (keep K smallest keys; root holds the
// largest of the K kept) and the "max top-K" usage (keep K largest keys;
// root holds the smallest of the K kept). The data structure is a single
// bounded max-heap; "max top-K" is implemented by negating keys at the
// caller, so only one heap implementation is tested here.

#include "simdq_topk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static void test_init_and_threshold(void) {
    simdq_topk_t h;
    int64_t keys[8]; int64_t idxs[8];
    simdq_topk_init(&h, 8, keys, idxs);
    CHECK(h.k == 8, "k=%d", h.k);
    CHECK(h.size == 0, "size=%d", h.size);
    CHECK(simdq_topk_threshold(&h) == INT64_MAX, "empty threshold");
}

static void test_fill_to_capacity(void) {
    simdq_topk_t h;
    int64_t keys[4]; int64_t idxs[4];
    simdq_topk_init(&h, 4, keys, idxs);

    int64_t input_keys[] = {50, 10, 30, 70};
    for (int i = 0; i < 4; i++) simdq_topk_offer(&h, input_keys[i], i);
    CHECK(h.size == 4, "size=%d", h.size);
    CHECK(simdq_topk_threshold(&h) == 70, "root key=%lld",
          (long long)simdq_topk_threshold(&h));
}

static void test_evict_when_full(void) {
    simdq_topk_t h;
    int64_t keys[4]; int64_t idxs[4];
    simdq_topk_init(&h, 4, keys, idxs);
    int64_t input_keys[] = {50, 10, 30, 70};
    for (int i = 0; i < 4; i++) simdq_topk_offer(&h, input_keys[i], i);

    // 5 is smaller than root (70): should evict 70 and keep 5
    simdq_topk_offer(&h, 5, 100);
    CHECK(h.size == 4, "size after evict=%d", h.size);
    CHECK(simdq_topk_threshold(&h) == 50, "new root=%lld",
          (long long)simdq_topk_threshold(&h));

    // 90 is bigger than root (50): should be ignored
    simdq_topk_offer(&h, 90, 200);
    CHECK(simdq_topk_threshold(&h) == 50, "root unchanged after rejected offer");
}

static void test_extract_sorted(void) {
    simdq_topk_t h;
    int64_t keys[4]; int64_t idxs[4];
    simdq_topk_init(&h, 4, keys, idxs);
    int64_t input_keys[] = {50, 10, 30, 70};
    for (int i = 0; i < 4; i++) simdq_topk_offer(&h, input_keys[i], i);

    int64_t out_keys[4]; int64_t out_idxs[4];
    int n = simdq_topk_extract_sorted(&h, out_keys, out_idxs);
    CHECK(n == 4, "extract returned n=%d", n);

    // ascending by key (smallest first)
    int64_t want_keys[] = {10, 30, 50, 70};
    int64_t want_idxs[] = {1, 2, 0, 3};
    for (int i = 0; i < 4; i++) {
        CHECK(out_keys[i] == want_keys[i] && out_idxs[i] == want_idxs[i],
              "rank %d got (%lld,%lld) want (%lld,%lld)", i,
              (long long)out_keys[i], (long long)out_idxs[i],
              (long long)want_keys[i], (long long)want_idxs[i]);
    }
}

static void test_merge_two_heaps(void) {
    // mimic the per-thread-merge pattern used by the parallel driver
    simdq_topk_t a, b;
    int64_t ka[4], ia[4], kb[4], ib[4];
    simdq_topk_init(&a, 4, ka, ia);
    simdq_topk_init(&b, 4, kb, ib);

    int64_t aks[] = {10, 20, 30, 40};
    int64_t bks[] = {15, 25, 35, 45};
    for (int i = 0; i < 4; i++) simdq_topk_offer(&a, aks[i], i);
    for (int i = 0; i < 4; i++) simdq_topk_offer(&b, bks[i], 100 + i);

    // merge: drain b into a
    int64_t bk_out[4], bi_out[4];
    int nb = simdq_topk_extract_sorted(&b, bk_out, bi_out);
    for (int i = 0; i < nb; i++) simdq_topk_offer(&a, bk_out[i], bi_out[i]);

    int64_t ok[4], oi[4];
    int n = simdq_topk_extract_sorted(&a, ok, oi);
    CHECK(n == 4, "merged size=%d", n);
    int64_t want_keys[] = {10, 15, 20, 25};
    int64_t want_idxs[] = {0, 100, 1, 101};
    for (int i = 0; i < 4; i++)
        CHECK(ok[i] == want_keys[i] && oi[i] == want_idxs[i],
              "merge rank %d got (%lld,%lld) want (%lld,%lld)", i,
              (long long)ok[i], (long long)oi[i],
              (long long)want_keys[i], (long long)want_idxs[i]);
}

int main(void) {
    test_init_and_threshold();
    test_fill_to_capacity();
    test_evict_when_full();
    test_extract_sorted();
    test_merge_two_heaps();
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all topk tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Add the test target to `CMakeLists.txt`**

Append to `_native/CMakeLists.txt`:

```cmake
add_executable(test_topk tests/test_topk.c)
target_compile_options(test_topk PRIVATE ${COMMON_FLAGS})
target_include_directories(test_topk PRIVATE include)
add_test(NAME topk COMMAND test_topk)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd docuverse/engines/retrieval/simdq/_native/build
cmake .. && cmake --build . -j
```

Expected: build error, `simdq_topk.h` not found.

- [ ] **Step 4: Implement `simdq_topk.h`**

```c
// simdq_topk.h — bounded max-heap of (key, index) pairs for top-K scans.
//
// Caller-allocated storage: `keys` and `idxs` arrays of length k, owned by
// the caller. The struct itself is small (4 pointers + 2 ints) and lives
// on the stack in scan kernels.
//
// The heap holds the K *smallest* keys seen so far; the root (index 0) is
// the *largest* of the K kept — i.e. the eviction candidate. simdq_topk_offer
// replaces the root iff the new key is strictly less.
//
// For top-K-largest workflows (asymmetric scans where higher score = better),
// negate keys at the caller and convert back at extraction.
//
// Hot-path contract: simdq_topk_threshold(h) returns the root key and is
// safe to call when the heap is full. While the heap is not yet full,
// the threshold is INT64_MAX. Inner scan loops should compare candidate
// keys against this threshold before going through simdq_topk_offer.

#pragma once

#include <stdint.h>

typedef struct {
    int64_t *keys;   // heap-ordered; keys[0] is the largest of the K kept
    int64_t *idxs;
    int k;           // capacity
    int size;        // 0..k
} simdq_topk_t;

static inline void simdq_topk_init(simdq_topk_t *h, int k,
                                   int64_t *keys, int64_t *idxs) {
    h->keys = keys;
    h->idxs = idxs;
    h->k = k;
    h->size = 0;
}

static inline int64_t simdq_topk_threshold(const simdq_topk_t *h) {
    return (h->size < h->k) ? INT64_MAX : h->keys[0];
}

static inline void simdq_topk__sift_down(simdq_topk_t *h, int i) {
    while (1) {
        int l = 2*i + 1, r = 2*i + 2, m = i;
        if (l < h->size && h->keys[l] > h->keys[m]) m = l;
        if (r < h->size && h->keys[r] > h->keys[m]) m = r;
        if (m == i) break;
        int64_t tk = h->keys[i]; h->keys[i] = h->keys[m]; h->keys[m] = tk;
        int64_t ti = h->idxs[i]; h->idxs[i] = h->idxs[m]; h->idxs[m] = ti;
        i = m;
    }
}

static inline void simdq_topk__sift_up(simdq_topk_t *h, int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->keys[p] >= h->keys[i]) break;
        int64_t tk = h->keys[i]; h->keys[i] = h->keys[p]; h->keys[p] = tk;
        int64_t ti = h->idxs[i]; h->idxs[i] = h->idxs[p]; h->idxs[p] = ti;
        i = p;
    }
}

static inline void simdq_topk_offer(simdq_topk_t *h, int64_t key, int64_t idx) {
    if (h->size < h->k) {
        h->keys[h->size] = key;
        h->idxs[h->size] = idx;
        h->size++;
        simdq_topk__sift_up(h, h->size - 1);
    } else if (key < h->keys[0]) {
        h->keys[0] = key;
        h->idxs[0] = idx;
        simdq_topk__sift_down(h, 0);
    }
}

// Extracts heap contents in ASCENDING key order. Destroys the heap.
// Returns the number of elements written (= size at entry).
static inline int simdq_topk_extract_sorted(simdq_topk_t *h,
                                            int64_t *out_keys,
                                            int64_t *out_idxs) {
    int n = h->size;
    // pop the max repeatedly; results emerge largest-first, so write in reverse
    while (h->size > 0) {
        int64_t k = h->keys[0], i = h->idxs[0];
        h->size--;
        if (h->size > 0) {
            h->keys[0] = h->keys[h->size];
            h->idxs[0] = h->idxs[h->size];
            simdq_topk__sift_down(h, 0);
        }
        out_keys[h->size] = k;
        out_idxs[h->size] = i;
    }
    return n;
}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd docuverse/engines/retrieval/simdq/_native/build
cmake --build . -j && ctest -R '^topk$' --output-on-failure
```

Expected: `topk` test passes.

- [ ] **Step 6: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_topk.h \
        docuverse/engines/retrieval/simdq/_native/tests/test_topk.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add bounded max-heap top-K data structure for simdq scans.

Caller-allocated, hot-path threshold check, ascending-sorted extract.
Foundation for top-K Hamming and asymmetric scan kernels."
```

---

## Task 3: Hamming top-K scan kernel (both SIMD paths)

**Goal:** generalize `scan_shard` from top-1 to top-K, for both AVX-512 and AVX2 paths.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming_topk.h`
- Create: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming_topk.c`

- [ ] **Step 1: Write the failing test**

`tests/test_kernels_hamming_topk.c`:

```c
// test_kernels_hamming_topk.c — correctness for top-K Hamming scans on
// both SIMD paths. Same shape as test_kernels_hamming.c (planted-best
// cases at boundaries, randomized + scalar reference) but exercising
// scan_shard_topk and scan_batch_parallel_topk over K > 1.

#include "simdq_kernels_hamming_topk.h"
#include <stdio.h>
#include <string.h>

#ifndef KERNEL_NAME
#error "tests need at least AVX2"
#endif

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

// independent scalar top-K reference, ascending order
static void ref_topk(const uint64_t *dbT, size_t n, const uint64_t *q,
                     int K, int64_t *out_d, int64_t *out_i) {
    // brute O(n*K) — fine for n <= a few thousand in tests
    for (int r = 0; r < K; r++) { out_d[r] = INT64_MAX; out_i[r] = -1; }
    for (size_t k = 0; k < n; k++) {
        int h = 0;
        for (int w = 0; w < WORDS; w++)
            h += __builtin_popcountll(q[w] ^ dbT[(size_t)w * n + k]);
        // insertion into sorted top-K
        if (h < out_d[K - 1]) {
            int r = K - 1;
            while (r > 0 && out_d[r - 1] > h) {
                out_d[r] = out_d[r - 1]; out_i[r] = out_i[r - 1]; r--;
            }
            out_d[r] = h; out_i[r] = (int64_t)k;
        }
    }
}

// plant `count` distinct unique-best codes for q at given indices, with
// strictly increasing distances 0, 1, 2, ... (count <= 6 for safety).
static void plant_ascending(uint64_t *dbT, size_t n, const uint64_t *q,
                            const size_t *positions, int count) {
    for (int p = 0; p < count; p++) {
        uint64_t code[WORDS];
        memcpy(code, q, WORDS * 8);
        for (int f = 0; f < p; f++) {
            // bits unique across plants: shift by p*64 + f
            unsigned bit = (unsigned)((p * 64 + f) % (WORDS * 64));
            code[bit / 64] ^= 1ull << (bit % 64);
        }
        for (int w = 0; w < WORDS; w++)
            dbT[(size_t)w * n + positions[p]] = code[w];
    }
}

static void test_topk_planted(void) {
    size_t n = 1000;
    int K = 5;
    uint64_t *dbT = alloc_codes(n);
    uint64_t q[WORDS];
    fill_rnd64(q, WORDS);
    fill_rnd64(dbT, n * WORDS);
    size_t pos[5] = {0, 250, 500, 750, n - 1};
    plant_ascending(dbT, n, q, pos, 5);

    int64_t out_d[5], out_i[5];
    scan_shard_topk(dbT, n, 0, n, q, K, out_d, out_i);

    int64_t ref_d[5], ref_i[5];
    ref_topk(dbT, n, q, K, ref_d, ref_i);

    for (int r = 0; r < K; r++)
        CHECK(out_d[r] == ref_d[r] && out_i[r] == ref_i[r],
              "K=%d r=%d got (%lld,%lld) ref (%lld,%lld)", K, r,
              (long long)out_d[r], (long long)out_i[r],
              (long long)ref_d[r], (long long)ref_i[r]);
    free(dbT);
}

static void test_topk_random(void) {
    size_t n = 2000;
    int K = 10;
    uint64_t *dbT = alloc_codes(n);
    uint64_t q[WORDS];
    fill_rnd64(q, WORDS);
    fill_rnd64(dbT, n * WORDS);

    int64_t out_d[10], out_i[10];
    scan_shard_topk(dbT, n, 0, n, q, K, out_d, out_i);

    int64_t ref_d[10], ref_i[10];
    ref_topk(dbT, n, q, K, ref_d, ref_i);

    // distances must match exactly; indices may differ on ties
    for (int r = 0; r < K; r++) {
        CHECK(out_d[r] == ref_d[r], "K=%d r=%d d=%lld ref=%lld",
              K, r, (long long)out_d[r], (long long)ref_d[r]);
        // verify index attains the claimed distance
        int h = 0;
        size_t k = (size_t)out_i[r];
        for (int w = 0; w < WORDS; w++)
            h += __builtin_popcountll(q[w] ^ dbT[(size_t)w * n + k]);
        CHECK(h == out_d[r], "K=%d r=%d idx=%lld dist mismatch", K, r,
              (long long)out_i[r]);
    }
    free(dbT);
}

static void test_topk_K_eq_1(void) {
    // K=1 must agree with the existing top-1 kernel
    size_t n = 500;
    uint64_t *dbT = alloc_codes(n);
    uint64_t q[WORDS];
    fill_rnd64(q, WORDS);
    fill_rnd64(dbT, n * WORDS);

    int64_t out_d, out_i;
    scan_shard_topk(dbT, n, 0, n, q, 1, &out_d, &out_i);

    int rd; size_t ri = ref_scan_soa_top1(dbT, n, 0, n, q, &rd);
    CHECK(out_d == rd, "d=%lld ref=%d", (long long)out_d, rd);
    int h = 0;
    for (int w = 0; w < WORDS; w++)
        h += __builtin_popcountll(q[w] ^ dbT[(size_t)w * n + (size_t)out_i]);
    CHECK(h == rd, "K=1 idx=%lld dist mismatch", (long long)out_i);
    free(dbT);
}

static void test_topk_threaded(void) {
    size_t n = 10000;
    int K = 8;
    uint64_t *dbT = alloc_codes(n);
    uint64_t q[WORDS];
    fill_rnd64(q, WORDS);
    fill_rnd64(dbT, n * WORDS);

    // plant 8 strictly-better codes spread across shards
    size_t pos[8];
    for (int p = 0; p < 8; p++) pos[p] = (size_t)p * n / 8 + 17;
    plant_ascending(dbT, n, q, pos, 8);

    int64_t gd[8], gi[8];
    scan_batch_parallel_topk(dbT, n, q, K, gd, gi);
    int64_t rd[8], ri[8];
    ref_topk(dbT, n, q, K, rd, ri);

    for (int r = 0; r < K; r++)
        CHECK(gd[r] == rd[r] && gi[r] == ri[r],
              "threaded K=%d r=%d got (%lld,%lld) ref (%lld,%lld)", K, r,
              (long long)gd[r], (long long)gi[r],
              (long long)rd[r], (long long)ri[r]);
    free(dbT);
}

int main(void) {
    srand(98765);
    printf("kernel path: %s\n", KERNEL_NAME);
    test_topk_K_eq_1();
    test_topk_planted();
    test_topk_random();
#ifdef _OPENMP
    test_topk_threaded();
#endif
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all topk kernel tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Add test target + run to verify failure**

Append to `CMakeLists.txt`:

```cmake
add_executable(test_kernels_hamming_topk tests/test_kernels_hamming_topk.c)
target_compile_options(test_kernels_hamming_topk PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_hamming_topk PRIVATE include)
target_link_libraries(test_kernels_hamming_topk PRIVATE OpenMP::OpenMP_C)
add_test(NAME kernels_hamming_topk_native COMMAND test_kernels_hamming_topk)

add_executable(test_kernels_hamming_topk_avx2 tests/test_kernels_hamming_topk.c)
target_compile_options(test_kernels_hamming_topk_avx2 PRIVATE -O3 -mavx2 -mpopcnt)
target_include_directories(test_kernels_hamming_topk_avx2 PRIVATE include)
target_link_libraries(test_kernels_hamming_topk_avx2 PRIVATE OpenMP::OpenMP_C)
add_test(NAME kernels_hamming_topk_avx2 COMMAND test_kernels_hamming_topk_avx2)
```

```bash
cd build && cmake .. && cmake --build . -j
```

Expected: build error, `simdq_kernels_hamming_topk.h` not found.

- [ ] **Step 3: Implement `simdq_kernels_hamming_topk.h`**

```c
// simdq_kernels_hamming_topk.h — top-K Hamming SoA scan kernels.
//
// Pattern: the inner SIMD loop computes per-lane distances exactly as in
// the top-1 path. After each LANES-block, distances are stored to a small
// scratch array and offered to a per-shard simdq_topk_t heap. The hot
// loop is unchanged from the top-1 kernel; the only added work is a
// threshold compare + heap offer per LANES-block (amortized over LANES
// distances). For K << n this overhead is negligible.

#pragma once

#include "simdq_common.h"
#include "simdq_topk.h"
#include <immintrin.h>
#include <string.h>

// Reference top-1 helper used by the K=1 test (independent of SIMD).
static inline size_t ref_scan_soa_top1(const uint64_t *dbT, size_t n,
                                       size_t i0, size_t i1,
                                       const uint64_t *q, int *out_d) {
    int best = INT32_MAX; size_t bi = i0;
    for (size_t k = i0; k < i1; k++) {
        int h = hamming_soa(dbT, n, k, q);
        if (h < best) { best = h; bi = k; }
    }
    *out_d = best;
    return bi;
}

#if defined(__AVX512VPOPCNTDQ__)
#define KERNEL_NAME "AVX512-VPOPCNTDQ"
#define LANES 8

/*
 * Single-query top-K Hamming SoA scan over codes [i0, i1). Maintains a
 * bounded max-heap of size K; after each 8-code block we read out the 8
 * per-lane distances, compare against the heap threshold, and offer the
 * survivors. Returns the K smallest distances (ascending) and their
 * indices in out_d / out_i.
 */
static inline void scan_shard_topk(const uint64_t *dbT, size_t n,
                                   size_t i0, size_t i1,
                                   const uint64_t *q, int K,
                                   int64_t *out_d, int64_t *out_i) {
    int64_t hkeys[256], hidxs[256];     // K up to 256 supported on stack
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    size_t i = i0;
    for (; i + LANES <= i1; i += LANES) {
        __m512i acc = _mm512_setzero_si512();
        for (int w = 0; w < WORDS; w++) {
            __m512i d = _mm512_loadu_si512(dbT + (size_t)w * n + i);
            acc = _mm512_add_epi64(acc,
                  _mm512_popcnt_epi64(_mm512_xor_si512(d, _mm512_set1_epi64(q[w]))));
        }
        int64_t d8[8];
        _mm512_storeu_si512(d8, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < 8; l++)
            if (d8[l] < thr) {
                simdq_topk_offer(&heap, d8[l], (int64_t)(i + l));
                thr = simdq_topk_threshold(&heap);
            }
    }
    for (size_t k = i; k < i1; k++) {
        int h = hamming_soa(dbT, n, k, q);
        if ((int64_t)h < simdq_topk_threshold(&heap))
            simdq_topk_offer(&heap, (int64_t)h, (int64_t)k);
    }
    simdq_topk_extract_sorted(&heap, out_d, out_i);
}

#elif defined(__AVX2__)
#define KERNEL_NAME "AVX2-nibbleLUT"
#define LANES 4

static inline __m256i popcnt_bytes_avx2(__m256i x, __m256i lut, __m256i m0f) {
    __m256i lo = _mm256_and_si256(x, m0f);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), m0f);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                           _mm256_shuffle_epi8(lut, hi));
}

static inline void scan_shard_topk(const uint64_t *dbT, size_t n,
                                   size_t i0, size_t i1,
                                   const uint64_t *q, int K,
                                   int64_t *out_d, int64_t *out_i) {
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();

    size_t i = i0;
    for (; i + LANES <= i1; i += LANES) {
        __m256i accb = zero;
        for (int w = 0; w < WORDS; w++) {
            __m256i d = _mm256_loadu_si256(
                (const __m256i *)(dbT + (size_t)w * n + i));
            __m256i x = _mm256_xor_si256(d, _mm256_set1_epi64x(q[w]));
            accb = _mm256_add_epi8(accb, popcnt_bytes_avx2(x, lut, m0f));
        }
        __m256i sums = _mm256_sad_epu8(accb, zero);
        int64_t d4[4];
        _mm256_storeu_si256((__m256i *)d4, sums);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < LANES; l++)
            if (d4[l] < thr) {
                simdq_topk_offer(&heap, d4[l], (int64_t)(i + l));
                thr = simdq_topk_threshold(&heap);
            }
    }
    for (size_t k = i; k < i1; k++) {
        int h = hamming_soa(dbT, n, k, q);
        if ((int64_t)h < simdq_topk_threshold(&heap))
            simdq_topk_offer(&heap, (int64_t)h, (int64_t)k);
    }
    simdq_topk_extract_sorted(&heap, out_d, out_i);
}
#endif

#if defined(KERNEL_NAME) && defined(_OPENMP)
#include <omp.h>

/*
 * Threaded top-K Hamming scan. Each thread runs scan_shard_topk on its
 * range, then a critical-section merge folds per-thread heaps into the
 * global top-K. For T threads merging K entries each, merge cost is
 * O(T*K*log K) — negligible vs the scan.
 */
static inline void scan_batch_parallel_topk(const uint64_t *dbT, size_t n,
                                            const uint64_t *q, int K,
                                            int64_t *gd, int64_t *gi) {
    int64_t gkeys[256], gidxs[256];
    simdq_topk_t global;
    simdq_topk_init(&global, K, gkeys, gidxs);

    #pragma omp parallel
    {
        int t = omp_get_thread_num(), T = omp_get_num_threads();
        size_t chunk = (n + T - 1) / T;
        size_t i0 = (size_t)t * chunk;
        size_t i1 = i0 + chunk < n ? i0 + chunk : n;
        if (i0 < i1) {
            int64_t ld[256], li[256];
            scan_shard_topk(dbT, n, i0, i1, q, K, ld, li);
            #pragma omp critical
            for (int r = 0; r < K; r++)
                if (li[r] >= 0 && ld[r] < simdq_topk_threshold(&global))
                    simdq_topk_offer(&global, ld[r], li[r]);
        }
    }
    simdq_topk_extract_sorted(&global, gd, gi);
}
#endif
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd build && cmake --build . -j
ctest -R 'kernels_hamming_topk' --output-on-failure
```

Expected: both `kernels_hamming_topk_native` and `kernels_hamming_topk_avx2` pass.

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_hamming_topk.h \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_hamming_topk.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add top-K Hamming SoA scan kernels (AVX-512 + AVX2 + threaded driver).

Generalizes scan_shard from top-1 to top-K via a bounded max-heap fed
after each LANES-block. Threshold check before offer keeps the hot
loop branch-light. Threaded variant merges per-thread heaps under an
omp critical."
```

---

## Task 4: Top-K Hamming benchmark binary

**Goal:** a sweep-friendly benchmark in the same shape as the existing `hb5`, but reporting top-K throughput. Lets us check that top-K overhead is negligible (which we expect for K << n).

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/bench/bench_hamming.c`

- [ ] **Step 1: Write the benchmark driver**

`bench/bench_hamming.c`:

```c
// bench_hamming.c — top-K Hamming SoA scan throughput.
//
// CLI: bench_hamming [n] [reps] [K]
//   n     number of database codes (default 1,000,000)
//   reps  timed repetitions; reports the minimum (default 5)
//   K     top-K depth (default 100)
// Threaded; honors OMP_NUM_THREADS.

#include "simdq_kernels_hamming_topk.h"
#include <stdio.h>

int main(int argc, char **argv) {
    size_t n;
    int reps;
    parse_args(argc, argv, &n, &reps);
    int K = (argc > 3) ? atoi(argv[3]) : 100;
    if (K < 1 || K > 256) { fprintf(stderr, "K must be in [1,256]\n"); return 2; }
    srand(42);

    uint64_t *dbT = alloc_codes(n);
    uint64_t q[WORDS];
    if (!dbT) { fprintf(stderr, "alloc failed\n"); return 1; }
    fill_soa_parallel(dbT, n);
    fill_rnd64(q, WORDS);

    int64_t out_d[256], out_i[256];
    // warmup
    scan_batch_parallel_topk(dbT, n, q, K, out_d, out_i);

    double tmin = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        scan_batch_parallel_topk(dbT, n, q, K, out_d, out_i);
        double dt = now() - t0;
        if (dt < tmin) tmin = dt;
    }
    double cmps = (double)n / tmin;
    double bytes = (double)n * WORDS * 8;
    double gbs = bytes / tmin / 1e9;
    printf("kernel=%s n=%zu K=%d reps=%d  best=%.3fms  cmp/s=%.2fM  GB/s=%.1f  top1=%lld\n",
           KERNEL_NAME, n, K, reps, tmin * 1e3, cmps / 1e6, gbs,
           (long long)out_i[0]);
    free(dbT);
    return 0;
}
```

- [ ] **Step 2: Add target to `CMakeLists.txt`**

```cmake
add_executable(bench_hamming bench/bench_hamming.c)
target_compile_options(bench_hamming PRIVATE ${COMMON_FLAGS})
target_include_directories(bench_hamming PRIVATE include)
target_link_libraries(bench_hamming PRIVATE OpenMP::OpenMP_C)
```

- [ ] **Step 3: Build and run a smoke test**

```bash
cd build && cmake .. && cmake --build . -j
OMP_NUM_THREADS=8 ./bench_hamming 1000000 5 100
OMP_NUM_THREADS=32 ./bench_hamming 50000000 3 100
```

Expected: numbers similar to `hb5` for top-1 (within 5%), since top-K overhead at K=100 against random codes is tiny.

- [ ] **Step 4: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/bench/bench_hamming.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add bench_hamming top-K throughput driver."
```

---

## Task 5: b=1 asymmetric — pack/unpack helpers

**Goal:** SoA pack helper for 1-bit codes — sign-bit per dim, 8 dims per byte. This is the simplest of the three b values; it warms us up for b=2 and b=4.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/include/simdq_pack.h`

The packing layout for asymmetric scans is **dim-major SoA**: for each dim `w` we store a contiguous run of `N` packed values across codes. With b=1, that's 1 bit per code per dim → `ceil(N/8)` bytes per dim. To load 64 codes' dim-`w` value, we read 8 bytes.

This differs from the *Hamming* SoA layout (which packed all 768 bits of each code in groups, transposed by 64-bit word). The asymmetric layout transposes by *dim*, not by word — so the inner kernel can broadcast `q[w]` and FMA against a vector of unpacked codes.

- [ ] **Step 1: Write `simdq_pack.h`**

```c
// simdq_pack.h — pack/unpack helpers for b-bit SoA codes used by
// asymmetric scan kernels.
//
// Layout (dim-major SoA, fixed across all b values):
//   For each dim w in [0, d), store a contiguous run of N codes' dim-w
//   value, packed at b bits per value. The buffer layout is:
//
//     b=1:  codes_b1[w * ceil(N/8) + (i>>3)]   bit (i & 7)
//     b=2:  codes_b2[w * ceil(N/4) + (i>>2)]   bits 2*(i & 3) .. 2*(i & 3)+1
//     b=4:  codes_b4[w * ceil(N/2) + (i>>1)]   nibble (i & 1)
//
// Quantization levels are V_b = {2c - (2^b - 1) | c=0..2^b-1}:
//   b=1:  {-1, +1}
//   b=2:  {-3, -1, +1, +3}
//   b=4:  {-15, -13, ..., +13, +15}
//
// Per-vector scale: each code i has a fp32 scale s_i stored separately.
// During pack, the i-th vector y_i is normalized by its own scale before
// being mapped to the nearest level. The ASH-style recovery factor used
// at scan time is left to the kernel (see kernel comments).

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

// Compute per-vector scale: ||y_i|| / sqrt(d).  Used by all b values.
// Returns a buffer of N fp32 scales the caller must free().
static inline float *simdq_pack_scales(const float *Y, size_t N, size_t d) {
    float *scales = (float *)malloc(N * sizeof(float));
    if (!scales) return NULL;
    const float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (size_t i = 0; i < N; i++) {
        float s = 0.0f;
        const float *yi = Y + i * d;
        for (size_t w = 0; w < d; w++) s += yi[w] * yi[w];
        scales[i] = sqrtf(s) * inv_sqrt_d;
        if (scales[i] == 0.0f) scales[i] = 1.0f;     // guard against zero
    }
    return scales;
}

// Quantize and pack into b=1 dim-major SoA. Codes buffer must be at least
// d * ceil(N/8) bytes, zeroed before this call (we OR into it).
static inline void simdq_pack_b1(const float *Y, size_t N, size_t d,
                                 const float *scales, uint8_t *codes_b1) {
    const size_t row_bytes = (N + 7) / 8;
    memset(codes_b1, 0, d * row_bytes);
    for (size_t i = 0; i < N; i++) {
        const float *yi = Y + i * d;
        const float si = scales[i];
        const size_t byte = i >> 3;
        const uint8_t bit = (uint8_t)(1u << (i & 7));
        for (size_t w = 0; w < d; w++) {
            float yn = yi[w] / si;
            // 1-bit: sign(yn) -> +1 or -1; pack as bit=1 for +1, bit=0 for -1
            if (yn >= 0.0f) codes_b1[w * row_bytes + byte] |= bit;
        }
    }
}

// Quantize and pack into b=2 dim-major SoA. Levels {-3,-1,+1,+3}, encoded
// as 2-bit unsigned indices {0,1,2,3} mapping to {-3,-1,+1,+3}.
// Codes buffer must be at least d * ceil(N/4) bytes.
static inline void simdq_pack_b2(const float *Y, size_t N, size_t d,
                                 const float *scales, uint8_t *codes_b2) {
    const size_t row_bytes = (N + 3) / 4;
    memset(codes_b2, 0, d * row_bytes);
    for (size_t i = 0; i < N; i++) {
        const float *yi = Y + i * d;
        const float si = scales[i];
        const size_t byte = i >> 2;
        const unsigned shift = (unsigned)((i & 3) * 2);
        for (size_t w = 0; w < d; w++) {
            float yn = yi[w] / si;
            // map to nearest level in {-3,-1,+1,+3} via scaled threshold
            // boundaries at -2, 0, +2
            uint8_t code;
            if      (yn < -2.0f) code = 0;        // -3
            else if (yn <  0.0f) code = 1;        // -1
            else if (yn <  2.0f) code = 2;        // +1
            else                 code = 3;        // +3
            codes_b2[w * row_bytes + byte] |= (uint8_t)(code << shift);
        }
    }
}

// Quantize and pack into b=4 dim-major SoA. Levels {-15,-13,...,+13,+15},
// encoded as 4-bit unsigned indices {0..15} mapping to (2c - 15).
// Codes buffer must be at least d * ceil(N/2) bytes.
static inline void simdq_pack_b4(const float *Y, size_t N, size_t d,
                                 const float *scales, uint8_t *codes_b4) {
    const size_t row_bytes = (N + 1) / 2;
    memset(codes_b4, 0, d * row_bytes);
    for (size_t i = 0; i < N; i++) {
        const float *yi = Y + i * d;
        const float si = scales[i];
        const size_t byte = i >> 1;
        const unsigned shift = (unsigned)((i & 1) * 4);
        for (size_t w = 0; w < d; w++) {
            float yn = yi[w] / si;
            // 16 evenly-spaced levels from -15 to +15, step 2.
            // Boundaries at ..., -12, -10, -8, ... +12, +14
            int level = (int)floorf((yn + 15.0f) * 0.5f + 0.5f);
            if (level < 0) level = 0;
            if (level > 15) level = 15;
            codes_b4[w * row_bytes + byte] |= (uint8_t)((unsigned)level << shift);
        }
    }
}

// Decode a single b=1 code value to its level (-1 or +1).
static inline int8_t simdq_unpack_b1(const uint8_t *codes_b1, size_t row_bytes,
                                     size_t w, size_t i) {
    uint8_t byte = codes_b1[w * row_bytes + (i >> 3)];
    return (byte & (1u << (i & 7))) ? (int8_t)1 : (int8_t)(-1);
}

static inline int8_t simdq_unpack_b2(const uint8_t *codes_b2, size_t row_bytes,
                                     size_t w, size_t i) {
    uint8_t byte = codes_b2[w * row_bytes + (i >> 2)];
    uint8_t code = (byte >> ((i & 3) * 2)) & 0x3;
    static const int8_t levels[4] = {-3, -1, 1, 3};
    return levels[code];
}

static inline int8_t simdq_unpack_b4(const uint8_t *codes_b4, size_t row_bytes,
                                     size_t w, size_t i) {
    uint8_t byte = codes_b4[w * row_bytes + (i >> 1)];
    uint8_t code = (byte >> ((i & 1) * 4)) & 0xF;
    return (int8_t)(2 * (int)code - 15);
}
```

- [ ] **Step 2: Write a pack/unpack round-trip test**

`tests/test_pack.c`:

```c
// test_pack.c — verify pack + unpack reconstructs the quantized levels for
// each b in {1, 2, 4}, and that scales handle the per-vector normalization.

#include "simdq_pack.h"
#include <stdio.h>
#include <stdlib.h>

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static void test_b1_roundtrip(void) {
    size_t N = 17, d = 32;
    float Y[17 * 32];
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, N, d);
    uint8_t codes[32 * 3];   // d * ceil(17/8) = 32 * 3
    simdq_pack_b1(Y, N, d, scales, codes);
    size_t row_bytes = (N + 7) / 8;
    for (size_t i = 0; i < N; i++) {
        for (size_t w = 0; w < d; w++) {
            float yn = Y[i * d + w] / scales[i];
            int8_t want = (yn >= 0.0f) ? 1 : -1;
            int8_t got = simdq_unpack_b1(codes, row_bytes, w, i);
            CHECK(got == want, "i=%zu w=%zu yn=%f got=%d want=%d",
                  i, w, yn, got, want);
        }
    }
    free(scales);
}

static void test_b2_roundtrip(void) {
    size_t N = 9, d = 16;
    float Y[9 * 16];
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;
    float *scales = simdq_pack_scales(Y, N, d);
    uint8_t codes[16 * 3];   // d * ceil(9/4) = 16 * 3
    simdq_pack_b2(Y, N, d, scales, codes);
    size_t row_bytes = (N + 3) / 4;
    for (size_t i = 0; i < N; i++) {
        for (size_t w = 0; w < d; w++) {
            float yn = Y[i * d + w] / scales[i];
            int8_t want;
            if      (yn < -2.0f) want = -3;
            else if (yn <  0.0f) want = -1;
            else if (yn <  2.0f) want = 1;
            else                 want = 3;
            int8_t got = simdq_unpack_b2(codes, row_bytes, w, i);
            CHECK(got == want, "i=%zu w=%zu yn=%f got=%d want=%d",
                  i, w, yn, got, want);
        }
    }
    free(scales);
}

static void test_b4_roundtrip(void) {
    size_t N = 13, d = 16;
    float Y[13 * 16];
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) * 30.0f - 15.0f;
    float *scales = simdq_pack_scales(Y, N, d);
    uint8_t codes[16 * 7];   // d * ceil(13/2) = 16 * 7
    simdq_pack_b4(Y, N, d, scales, codes);
    size_t row_bytes = (N + 1) / 2;
    for (size_t i = 0; i < N; i++) {
        for (size_t w = 0; w < d; w++) {
            float yn = Y[i * d + w] / scales[i];
            int level = (int)floorf((yn + 15.0f) * 0.5f + 0.5f);
            if (level < 0) level = 0; if (level > 15) level = 15;
            int8_t want = (int8_t)(2 * level - 15);
            int8_t got = simdq_unpack_b4(codes, row_bytes, w, i);
            CHECK(got == want, "i=%zu w=%zu yn=%f got=%d want=%d",
                  i, w, yn, got, want);
        }
    }
    free(scales);
}

int main(void) {
    srand(12345);
    test_b1_roundtrip();
    test_b2_roundtrip();
    test_b4_roundtrip();
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all pack/unpack tests passed\n");
    return 0;
}
```

Append to `CMakeLists.txt`:

```cmake
add_executable(test_pack tests/test_pack.c)
target_compile_options(test_pack PRIVATE ${COMMON_FLAGS})
target_include_directories(test_pack PRIVATE include)
target_link_libraries(test_pack PRIVATE m)
add_test(NAME pack COMMAND test_pack)
```

- [ ] **Step 3: Build and run**

```bash
cd build && cmake .. && cmake --build . -j
ctest -R '^pack$' --output-on-failure
```

Expected: pack test passes.

- [ ] **Step 4: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_pack.h \
        docuverse/engines/retrieval/simdq/_native/tests/test_pack.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add b=1/2/4 SoA pack helpers with per-vector scales.

Dim-major SoA layout: contiguous run of N codes' dim-w value, packed at
b bits. Round-trip tests cover all three b values and the per-vector
||y||/sqrt(d) scale path."
```

---

## Task 6: b=1 asymmetric scan kernel (both SIMD paths)

**Goal:** the simplest of the three asymmetric scans — float query × 1-bit DB code, ASH-style asymmetric. The DB code value is just `±1`, so the kernel is `acc += q[w] * (codes[w][i] ? 1 : -1)`. This validates the framework before tackling b=2 and b=4.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h`
- Create: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b1.c`

- [ ] **Step 1: Write the failing test**

`tests/test_kernels_asym_b1.c`:

```c
// test_kernels_asym_b1.c — correctness for asymmetric b=1 scan, both SIMD
// paths. The DB has ±1 codes; we compute q' . v_i  (float dot product)
// and confirm the kernel returns the top-K largest scores.

#include "simdq_kernels_asym_b1.h"
#include "simdq_pack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ASYM_KERNEL_NAME
#error "tests need at least AVX2 (-march=native or -mavx2)"
#endif

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

// scalar reference: returns the K largest scores in descending order
static void ref_topk_asym_b1(const uint8_t *codes, size_t N, size_t d,
                             const float *q, int K,
                             float *out_s, int64_t *out_i) {
    size_t row_bytes = (N + 7) / 8;
    for (int r = 0; r < K; r++) { out_s[r] = -INFINITY; out_i[r] = -1; }
    for (size_t i = 0; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
            int8_t v = simdq_unpack_b1(codes, row_bytes, w, i);
            s += q[w] * (float)v;
        }
        if (s > out_s[K - 1]) {
            int r = K - 1;
            while (r > 0 && out_s[r - 1] < s) {
                out_s[r] = out_s[r - 1]; out_i[r] = out_i[r - 1]; r--;
            }
            out_s[r] = s; out_i[r] = (int64_t)i;
        }
    }
}

static void test_random_b1(void) {
    size_t N = 1024, d = 768;
    float *Y = malloc(N * d * sizeof(float));
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, N, d);
    size_t row_bytes = (N + 7) / 8;
    uint8_t *codes = malloc(d * row_bytes);
    simdq_pack_b1(Y, N, d, scales, codes);

    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;

    int K = 10;
    float ks[10]; int64_t kis[10];
    scan_asym_b1_d768_topk(codes, N, q, K, ks, kis);

    float rs[10]; int64_t ris[10];
    ref_topk_asym_b1(codes, N, d, q, K, rs, ris);

    for (int r = 0; r < K; r++) {
        // scores must agree to within float roundoff
        CHECK(fabsf(ks[r] - rs[r]) < 1e-3f,
              "r=%d got=%f ref=%f diff=%f", r, ks[r], rs[r], fabsf(ks[r] - rs[r]));
        CHECK(kis[r] == ris[r] || fabsf(ks[r] - rs[r]) < 1e-4f,
              "r=%d idx got=%lld ref=%lld", r,
              (long long)kis[r], (long long)ris[r]);
    }
    free(Y); free(scales); free(codes);
}

static void test_planted_b1(void) {
    // plant a vector that exactly matches q's signs at index 42 -> top score
    size_t N = 256, d = 768;
    float *Y = malloc(N * d * sizeof(float));
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;
    // override row 42 with sign-aligned values
    for (size_t w = 0; w < 768; w++) Y[42 * d + w] = (q[w] >= 0) ? 1.0f : -1.0f;

    float *scales = simdq_pack_scales(Y, N, d);
    size_t row_bytes = (N + 7) / 8;
    uint8_t *codes = malloc(d * row_bytes);
    simdq_pack_b1(Y, N, d, scales, codes);

    float ks[3]; int64_t kis[3];
    scan_asym_b1_d768_topk(codes, N, q, 3, ks, kis);
    CHECK(kis[0] == 42, "planted top idx=%lld (want 42)", (long long)kis[0]);
    free(Y); free(scales); free(codes);
}

int main(void) {
    srand(7777);
    printf("asym kernel path: %s\n", ASYM_KERNEL_NAME);
    test_random_b1();
    test_planted_b1();
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all asym b1 tests passed\n");
    return 0;
}
```

Append to `CMakeLists.txt`:

```cmake
add_executable(test_kernels_asym_b1 tests/test_kernels_asym_b1.c)
target_compile_options(test_kernels_asym_b1 PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_asym_b1 PRIVATE include)
target_link_libraries(test_kernels_asym_b1 PRIVATE m)
add_test(NAME kernels_asym_b1_native COMMAND test_kernels_asym_b1)

add_executable(test_kernels_asym_b1_avx2 tests/test_kernels_asym_b1.c)
target_compile_options(test_kernels_asym_b1_avx2 PRIVATE -O3 -mavx2 -mfma)
target_include_directories(test_kernels_asym_b1_avx2 PRIVATE include)
target_link_libraries(test_kernels_asym_b1_avx2 PRIVATE m)
add_test(NAME kernels_asym_b1_avx2 COMMAND test_kernels_asym_b1_avx2)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd build && cmake .. && cmake --build . -j
```

Expected: build error, `simdq_kernels_asym_b1.h` not found.

- [ ] **Step 3: Implement `simdq_kernels_asym_b1.h`**

```c
// simdq_kernels_asym_b1.h — asymmetric float×1-bit scan, top-K, D=768.
//
// Layout: codes is dim-major SoA, b=1 packing (8 codes per byte).
//   codes[w * row_bytes + (i >> 3)]  bit (i & 7)  = sign(y_i[w]/s_i)
// Score: s_i = sum_w q'[w] * v_i[w]   where v_i[w] in {-1, +1}.
//
// Per-dim inner loop expands 8 packed bits into 8 ±1 floats and FMAs them
// against q'[w] broadcast. Per-LANES-codes block, 8/16 codes' partial
// scores are accumulated; per-block we offer them to a top-K heap.

#pragma once

#include "simdq_topk.h"
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define ASYM_D 768

#if defined(__AVX512F__)
#define ASYM_KERNEL_NAME "AVX512F"
#define ASYM_LANES 16     // floats per AVX-512 register; 16 codes per block

/*
 * Single-shard top-K asymmetric b=1 scan over codes [0, N). Inner loop:
 * for each dim w, broadcast q'[w], unpack 16 codes' bit (one byte holds
 * 8 codes; 16 codes = 2 bytes via mask -> int8 -> float conversion),
 * FMA into per-lane accumulator. After d dims, store the 16 partial
 * scores and offer to top-K heap.
 *
 * For d=768 and N <= a few million this is memory-bound on the codes
 * buffer at 1 byte per 8 codes per dim = N/8 * 768 bytes.
 */
static inline void scan_asym_b1_d768_topk(const uint8_t *codes, size_t N,
                                          const float *q, int K,
                                          float *out_s, int64_t *out_i) {
    const size_t row_bytes = (N + 7) / 8;
    // bounded max-heap of NEGATIVE scores, so larger -> "smaller" -> top
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    // Process 16 codes at a time; we need 2 bytes per dim (16 / 8).
    for (size_t i0 = 0; i0 + ASYM_LANES <= N; i0 += ASYM_LANES) {
        __m512 acc = _mm512_setzero_ps();
        for (size_t w = 0; w < ASYM_D; w++) {
            // 2 bytes from this dim's row, holding codes [i0, i0+16)
            uint16_t bits = ((uint16_t)codes[w * row_bytes + (i0 >> 3) + 1] << 8)
                          |  (uint16_t)codes[w * row_bytes + (i0 >> 3)];
            // expand 16 bits to a 16-lane mask register
            __mmask16 m = (__mmask16)bits;
            // +1 where bit set, -1 elsewhere
            __m512 v = _mm512_mask_blend_ps(m,
                            _mm512_set1_ps(-1.0f),
                            _mm512_set1_ps( 1.0f));
            __m512 qb = _mm512_set1_ps(q[w]);
            acc = _mm512_fmadd_ps(qb, v, acc);
        }
        // store 16 partial scores; offer each as -score (heap is min-of-largest)
        float sc[16] __attribute__((aligned(64)));
        _mm512_store_ps(sc, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < ASYM_LANES; l++) {
            // negate to convert top-largest into top-smallest
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));   // fixed-point key
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(i0 + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    // tail: scalar
    size_t i = N - (N % ASYM_LANES);
    for (; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < ASYM_D; w++) {
            int8_t v = (codes[w * row_bytes + (i >> 3)] & (1u << (i & 7))) ? 1 : -1;
            s += q[w] * (float)v;
        }
        int64_t neg = -(int64_t)(s * (float)(1 << 20));
        if (neg < simdq_topk_threshold(&heap))
            simdq_topk_offer(&heap, neg, (int64_t)i);
    }
    // extract sorted ascending in -score -> descending in score
    int64_t ks[256], is[256];
    int n = simdq_topk_extract_sorted(&heap, ks, is);
    for (int r = 0; r < n; r++) {
        out_s[r] = -(float)ks[r] / (float)(1 << 20);
        out_i[r] = is[r];
    }
}

#elif defined(__AVX2__) && defined(__FMA__)
#define ASYM_KERNEL_NAME "AVX2-FMA"
#define ASYM_LANES 8

static inline void scan_asym_b1_d768_topk(const uint8_t *codes, size_t N,
                                          const float *q, int K,
                                          float *out_s, int64_t *out_i) {
    const size_t row_bytes = (N + 7) / 8;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    for (size_t i0 = 0; i0 + ASYM_LANES <= N; i0 += ASYM_LANES) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t w = 0; w < ASYM_D; w++) {
            uint8_t bits = codes[w * row_bytes + (i0 >> 3)];
            // expand 8 bits to 8 floats: +1 or -1
            float vf[8];
            for (int l = 0; l < 8; l++)
                vf[l] = (bits & (1u << l)) ? 1.0f : -1.0f;
            __m256 v = _mm256_loadu_ps(vf);
            __m256 qb = _mm256_set1_ps(q[w]);
            acc = _mm256_fmadd_ps(qb, v, acc);
        }
        float sc[8] __attribute__((aligned(32)));
        _mm256_store_ps(sc, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < ASYM_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(i0 + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    size_t i = N - (N % ASYM_LANES);
    for (; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < ASYM_D; w++) {
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
#endif
```

> **Note on the fixed-point heap key:** the heap stores `int64_t` keys, so we encode each float score as a fixed-point negation `(int64_t)(-score * 2^20)`. This preserves ordering for any score within ±2^43, which covers any d=768 scan with q in fp32. The `2^20` factor leaves headroom.

- [ ] **Step 4: Run tests**

```bash
cd build && cmake --build . -j
ctest -R 'kernels_asym_b1' --output-on-failure
```

Expected: both `kernels_asym_b1_native` and `kernels_asym_b1_avx2` pass.

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b1.h \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b1.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add b=1 asymmetric scan kernel (AVX-512F + AVX2-FMA, top-K).

Float query x 1-bit DB code, dim-major SoA. AVX-512 path uses
mask_blend to expand 16 packed bits to ±1 floats; AVX2-FMA path
expands 8 bits via a small per-iteration scalar table."
```

---

## Task 7: b=1 asymmetric benchmark binary

**Goal:** standalone microbench reporting cmp/s and GB/s for the b=1 asymmetric scan.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b1.c`

- [ ] **Step 1: Write the benchmark**

```c
// bench_asym_b1.c — asymmetric b=1 (float query x 1-bit DB) top-K scan
// throughput.
//
// CLI: bench_asym_b1 [n] [reps] [K]

#include "simdq_kernels_asym_b1.h"
#include "simdq_pack.h"
#include "simdq_common.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    size_t n;
    int reps;
    parse_args(argc, argv, &n, &reps);
    int K = (argc > 3) ? atoi(argv[3]) : 100;
    if (K < 1 || K > 256) { fprintf(stderr, "K must be in [1,256]\n"); return 2; }
    srand(42);
    const size_t d = 768;

    // generate Y on the heap; for very large N this is the dominant alloc
    float *Y = malloc(n * d * sizeof(float));
    if (!Y) { fprintf(stderr, "alloc Y failed\n"); return 1; }
    for (size_t i = 0; i < n * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, n, d);
    size_t row_bytes = (n + 7) / 8;
    uint8_t *codes = aligned_alloc(64, d * row_bytes);
    simdq_pack_b1(Y, n, d, scales, codes);
    free(Y);   // we no longer need the floats

    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;

    float out_s[256]; int64_t out_i[256];
    // warmup
    scan_asym_b1_d768_topk(codes, n, q, K, out_s, out_i);

    double tmin = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        scan_asym_b1_d768_topk(codes, n, q, K, out_s, out_i);
        double dt = now() - t0;
        if (dt < tmin) tmin = dt;
    }
    double cmps = (double)n / tmin;
    double bytes = (double)n * d / 8;
    double gbs = bytes / tmin / 1e9;
    printf("kernel=%s n=%zu K=%d reps=%d  best=%.3fms  cmp/s=%.2fM  GB/s=%.1f  top1=%lld score=%f\n",
           ASYM_KERNEL_NAME, n, K, reps, tmin * 1e3,
           cmps / 1e6, gbs, (long long)out_i[0], out_s[0]);
    free(codes); free(scales);
    return 0;
}
```

Note: this benchmark is **single-threaded**. Threading the asymmetric scans (per-thread top-K heaps + merge) is a follow-up — Plan 1 establishes the kernel; threading lands in Plan 2 alongside the Python integration.

Append to `CMakeLists.txt`:

```cmake
add_executable(bench_asym_b1 bench/bench_asym_b1.c)
target_compile_options(bench_asym_b1 PRIVATE ${COMMON_FLAGS})
target_include_directories(bench_asym_b1 PRIVATE include)
target_link_libraries(bench_asym_b1 PRIVATE m)
```

- [ ] **Step 2: Build and run**

```bash
cd build && cmake .. && cmake --build . -j
./bench_asym_b1 1000000 5 100
```

Expected: working output with cmp/s and GB/s numbers. The b=1 asymmetric kernel does d=768 float-FMAs per code; expect roughly 200-500M cmp/s single-threaded, depending on cache behavior.

- [ ] **Step 3: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b1.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add bench_asym_b1 microbench (single-threaded for now)."
```

---

## Task 8: b=2 asymmetric — pack already exists; add scan kernel

The pack helpers were added in Task 5. Task 8 implements the `b=2` scan kernel.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b2.h`
- Create: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b2.c`

- [ ] **Step 1: Write the test**

`tests/test_kernels_asym_b2.c` — same shape as `test_kernels_asym_b1.c`, but:

1. Replace all `_b1` symbols with `_b2`.
2. Update `ref_topk_asym_b2` to compute the b=2 scalar reference: levels `{-3, -1, +1, +3}` via `simdq_unpack_b2`.
3. Update `simdq_pack_b1` calls to `simdq_pack_b2`.
4. Update planted-best test: instead of sign-aligned ±1 values, plant a vector whose every dim quantizes to the level *closest in sign* to `q[w]` — i.e. for `q[w] > 0`, plant `+3 * scale`; for `q[w] < 0`, plant `-3 * scale`. The planted code's score is then `sum_w |q[w]| * 3 / s = 3 * sum|q|/s`, which dominates random codes.

Full code (concrete, no placeholders):

```c
// test_kernels_asym_b2.c — correctness for asymmetric b=2 scan, both SIMD
// paths.

#include "simdq_kernels_asym_b2.h"
#include "simdq_pack.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef ASYM_KERNEL_NAME
#error "tests need at least AVX2 (-march=native or -mavx2)"
#endif

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static void ref_topk_asym_b2(const uint8_t *codes, size_t N, size_t d,
                             const float *q, int K,
                             float *out_s, int64_t *out_i) {
    size_t row_bytes = (N + 3) / 4;
    for (int r = 0; r < K; r++) { out_s[r] = -INFINITY; out_i[r] = -1; }
    for (size_t i = 0; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
            int8_t v = simdq_unpack_b2(codes, row_bytes, w, i);
            s += q[w] * (float)v;
        }
        if (s > out_s[K - 1]) {
            int r = K - 1;
            while (r > 0 && out_s[r - 1] < s) {
                out_s[r] = out_s[r - 1]; out_i[r] = out_i[r - 1]; r--;
            }
            out_s[r] = s; out_i[r] = (int64_t)i;
        }
    }
}

static void test_random_b2(void) {
    size_t N = 1024, d = 768;
    float *Y = malloc(N * d * sizeof(float));
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, N, d);
    size_t row_bytes = (N + 3) / 4;
    uint8_t *codes = malloc(d * row_bytes);
    simdq_pack_b2(Y, N, d, scales, codes);

    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;

    int K = 10;
    float ks[10]; int64_t kis[10];
    scan_asym_b2_d768_topk(codes, N, q, K, ks, kis);

    float rs[10]; int64_t ris[10];
    ref_topk_asym_b2(codes, N, d, q, K, rs, ris);

    for (int r = 0; r < K; r++) {
        CHECK(fabsf(ks[r] - rs[r]) < 1e-3f,
              "r=%d got=%f ref=%f", r, ks[r], rs[r]);
        // index may differ on near-ties
        CHECK(kis[r] == ris[r] || fabsf(ks[r] - rs[r]) < 1e-4f,
              "r=%d idx got=%lld ref=%lld", r,
              (long long)kis[r], (long long)ris[r]);
    }
    free(Y); free(scales); free(codes);
}

static void test_planted_b2(void) {
    size_t N = 256, d = 768;
    float *Y = malloc(N * d * sizeof(float));
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;
    // plant row 42 to quantize to +3 / -3 in every dim aligned with q
    for (size_t w = 0; w < 768; w++)
        Y[42 * d + w] = (q[w] >= 0) ? 100.0f : -100.0f;

    float *scales = simdq_pack_scales(Y, N, d);
    size_t row_bytes = (N + 3) / 4;
    uint8_t *codes = malloc(d * row_bytes);
    simdq_pack_b2(Y, N, d, scales, codes);

    float ks[3]; int64_t kis[3];
    scan_asym_b2_d768_topk(codes, N, q, 3, ks, kis);
    CHECK(kis[0] == 42, "planted top idx=%lld (want 42)", (long long)kis[0]);
    free(Y); free(scales); free(codes);
}

int main(void) {
    srand(8888);
    printf("asym b2 kernel path: %s\n", ASYM_KERNEL_NAME);
    test_random_b2();
    test_planted_b2();
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all asym b2 tests passed\n");
    return 0;
}
```

Append to `CMakeLists.txt`:

```cmake
add_executable(test_kernels_asym_b2 tests/test_kernels_asym_b2.c)
target_compile_options(test_kernels_asym_b2 PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_asym_b2 PRIVATE include)
target_link_libraries(test_kernels_asym_b2 PRIVATE m)
add_test(NAME kernels_asym_b2_native COMMAND test_kernels_asym_b2)

add_executable(test_kernels_asym_b2_avx2 tests/test_kernels_asym_b2.c)
target_compile_options(test_kernels_asym_b2_avx2 PRIVATE -O3 -mavx2 -mfma)
target_include_directories(test_kernels_asym_b2_avx2 PRIVATE include)
target_link_libraries(test_kernels_asym_b2_avx2 PRIVATE m)
add_test(NAME kernels_asym_b2_avx2 COMMAND test_kernels_asym_b2_avx2)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd build && cmake .. && cmake --build . -j
```

Expected: `simdq_kernels_asym_b2.h` not found.

- [ ] **Step 3: Implement `simdq_kernels_asym_b2.h`**

```c
// simdq_kernels_asym_b2.h — asymmetric float×2-bit scan, top-K, D=768.
//
// b=2 packing: 4 codes per byte, 2 bits each, levels {-3,-1,+1,+3}
// encoded as {0,1,2,3}. Decoding: code -> levels[code] in int8.

#pragma once

#include "simdq_topk.h"
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#define ASYM_D 768

#if defined(__AVX512F__)
#define ASYM_KERNEL_NAME "AVX512F"
#define ASYM_LANES 16

/*
 * Per dim w, we need ASYM_LANES=16 codes' decoded level. With b=2 and 4
 * codes per byte, that's 4 bytes (16 codes / 4 codes-per-byte). The 16
 * 2-bit fields are unpacked into 16 int8 values via a vpshufb-style
 * lookup, then converted to fp32 and FMA'd against q'[w] broadcast.
 */
static inline void scan_asym_b2_d768_topk(const uint8_t *codes, size_t N,
                                          const float *q, int K,
                                          float *out_s, int64_t *out_i) {
    const size_t row_bytes = (N + 3) / 4;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    for (size_t i0 = 0; i0 + ASYM_LANES <= N; i0 += ASYM_LANES) {
        __m512 acc = _mm512_setzero_ps();
        for (size_t w = 0; w < ASYM_D; w++) {
            // 4 bytes from this dim row, holding 16 codes' 2-bit values
            uint32_t packed = *(const uint32_t *)(codes + w * row_bytes + (i0 >> 2));
            // unpack to 16 int8 levels via small scalar table; the
            // optimizer keeps this register-resident inside the inner loop
            float vf[16];
            for (int l = 0; l < 16; l++) {
                uint8_t code = (uint8_t)((packed >> (l * 2)) & 0x3);
                static const int8_t levels[4] = {-3, -1, 1, 3};
                vf[l] = (float)levels[code];
            }
            __m512 v = _mm512_loadu_ps(vf);
            __m512 qb = _mm512_set1_ps(q[w]);
            acc = _mm512_fmadd_ps(qb, v, acc);
        }
        float sc[16] __attribute__((aligned(64)));
        _mm512_store_ps(sc, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < ASYM_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(i0 + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    // tail
    size_t i = N - (N % ASYM_LANES);
    static const int8_t levels[4] = {-3, -1, 1, 3};
    for (; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < ASYM_D; w++) {
            uint8_t byte = codes[w * row_bytes + (i >> 2)];
            int8_t v = levels[(byte >> ((i & 3) * 2)) & 0x3];
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

#elif defined(__AVX2__) && defined(__FMA__)
#define ASYM_KERNEL_NAME "AVX2-FMA"
#define ASYM_LANES 8

static inline void scan_asym_b2_d768_topk(const uint8_t *codes, size_t N,
                                          const float *q, int K,
                                          float *out_s, int64_t *out_i) {
    const size_t row_bytes = (N + 3) / 4;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);
    static const int8_t levels[4] = {-3, -1, 1, 3};

    for (size_t i0 = 0; i0 + ASYM_LANES <= N; i0 += ASYM_LANES) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t w = 0; w < ASYM_D; w++) {
            // 2 bytes hold 8 codes' 2-bit values
            uint16_t packed = *(const uint16_t *)(codes + w * row_bytes + (i0 >> 2));
            float vf[8];
            for (int l = 0; l < 8; l++)
                vf[l] = (float)levels[(packed >> (l * 2)) & 0x3];
            __m256 v = _mm256_loadu_ps(vf);
            __m256 qb = _mm256_set1_ps(q[w]);
            acc = _mm256_fmadd_ps(qb, v, acc);
        }
        float sc[8] __attribute__((aligned(32)));
        _mm256_store_ps(sc, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < ASYM_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(i0 + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    size_t i = N - (N % ASYM_LANES);
    for (; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < ASYM_D; w++) {
            uint8_t byte = codes[w * row_bytes + (i >> 2)];
            int8_t v = levels[(byte >> ((i & 3) * 2)) & 0x3];
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
#endif
```

- [ ] **Step 4: Run tests**

```bash
cd build && cmake --build . -j
ctest -R 'kernels_asym_b2' --output-on-failure
```

Expected: both `kernels_asym_b2_native` and `kernels_asym_b2_avx2` pass.

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b2.h \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b2.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add b=2 asymmetric scan kernel (AVX-512F + AVX2-FMA, top-K).

Float query x 2-bit DB code (levels {-3,-1,+1,+3}) over 768 dims.
Per-dim inner loop unpacks 16 (or 8 on AVX2) packed 2-bit values
into a fp32 vector, FMA against q[w] broadcast."
```

---

## Task 9: b=2 asymmetric benchmark binary

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b2.c`

- [ ] **Step 1: Write the benchmark**

Same shape as `bench_asym_b1.c`. Three changes:

1. Includes `simdq_kernels_asym_b2.h` instead of `_b1`.
2. Calls `simdq_pack_b2` and `scan_asym_b2_d768_topk`.
3. `row_bytes = (n + 3) / 4` and `bytes = n * d / 4`.

Concrete file:

```c
#include "simdq_kernels_asym_b2.h"
#include "simdq_pack.h"
#include "simdq_common.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    size_t n;
    int reps;
    parse_args(argc, argv, &n, &reps);
    int K = (argc > 3) ? atoi(argv[3]) : 100;
    if (K < 1 || K > 256) { fprintf(stderr, "K must be in [1,256]\n"); return 2; }
    srand(42);
    const size_t d = 768;

    float *Y = malloc(n * d * sizeof(float));
    if (!Y) { fprintf(stderr, "alloc Y failed\n"); return 1; }
    for (size_t i = 0; i < n * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, n, d);
    size_t row_bytes = (n + 3) / 4;
    uint8_t *codes = aligned_alloc(64, d * row_bytes);
    simdq_pack_b2(Y, n, d, scales, codes);
    free(Y);

    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;

    float out_s[256]; int64_t out_i[256];
    scan_asym_b2_d768_topk(codes, n, q, K, out_s, out_i);   // warmup

    double tmin = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        scan_asym_b2_d768_topk(codes, n, q, K, out_s, out_i);
        double dt = now() - t0;
        if (dt < tmin) tmin = dt;
    }
    double cmps = (double)n / tmin;
    double bytes = (double)n * d / 4;
    double gbs = bytes / tmin / 1e9;
    printf("kernel=%s n=%zu K=%d reps=%d  best=%.3fms  cmp/s=%.2fM  GB/s=%.1f  top1=%lld score=%f\n",
           ASYM_KERNEL_NAME, n, K, reps, tmin * 1e3,
           cmps / 1e6, gbs, (long long)out_i[0], out_s[0]);
    free(codes); free(scales);
    return 0;
}
```

Append to `CMakeLists.txt`:

```cmake
add_executable(bench_asym_b2 bench/bench_asym_b2.c)
target_compile_options(bench_asym_b2 PRIVATE ${COMMON_FLAGS})
target_include_directories(bench_asym_b2 PRIVATE include)
target_link_libraries(bench_asym_b2 PRIVATE m)
```

- [ ] **Step 2: Build and run**

```bash
cd build && cmake .. && cmake --build . -j
./bench_asym_b2 1000000 5 100
```

Expected: GB/s should be ~2× `bench_asym_b1` (twice the bytes per code), but cmp/s in the same order of magnitude.

- [ ] **Step 3: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b2.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add bench_asym_b2 microbench."
```

---

## Task 10: b=4 asymmetric scan kernel

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b4.h`
- Create: `docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b4.c`

- [ ] **Step 1: Write the test**

`tests/test_kernels_asym_b4.c` — same shape as the b=2 test, replacing all `_b2` with `_b4`. Specifically:

```c
// test_kernels_asym_b4.c — correctness for asymmetric b=4 scan.

#include "simdq_kernels_asym_b4.h"
#include "simdq_pack.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef ASYM_KERNEL_NAME
#error "tests need at least AVX2"
#endif

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static void ref_topk_asym_b4(const uint8_t *codes, size_t N, size_t d,
                             const float *q, int K,
                             float *out_s, int64_t *out_i) {
    size_t row_bytes = (N + 1) / 2;
    for (int r = 0; r < K; r++) { out_s[r] = -INFINITY; out_i[r] = -1; }
    for (size_t i = 0; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
            int8_t v = simdq_unpack_b4(codes, row_bytes, w, i);
            s += q[w] * (float)v;
        }
        if (s > out_s[K - 1]) {
            int r = K - 1;
            while (r > 0 && out_s[r - 1] < s) {
                out_s[r] = out_s[r - 1]; out_i[r] = out_i[r - 1]; r--;
            }
            out_s[r] = s; out_i[r] = (int64_t)i;
        }
    }
}

static void test_random_b4(void) {
    size_t N = 1024, d = 768;
    float *Y = malloc(N * d * sizeof(float));
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, N, d);
    size_t row_bytes = (N + 1) / 2;
    uint8_t *codes = malloc(d * row_bytes);
    simdq_pack_b4(Y, N, d, scales, codes);

    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;

    int K = 10;
    float ks[10]; int64_t kis[10];
    scan_asym_b4_d768_topk(codes, N, q, K, ks, kis);
    float rs[10]; int64_t ris[10];
    ref_topk_asym_b4(codes, N, d, q, K, rs, ris);
    for (int r = 0; r < K; r++) {
        CHECK(fabsf(ks[r] - rs[r]) < 1e-3f,
              "r=%d got=%f ref=%f", r, ks[r], rs[r]);
        CHECK(kis[r] == ris[r] || fabsf(ks[r] - rs[r]) < 1e-4f,
              "r=%d idx got=%lld ref=%lld", r,
              (long long)kis[r], (long long)ris[r]);
    }
    free(Y); free(scales); free(codes);
}

static void test_planted_b4(void) {
    size_t N = 256, d = 768;
    float *Y = malloc(N * d * sizeof(float));
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;
    for (size_t w = 0; w < 768; w++)
        Y[42 * d + w] = (q[w] >= 0) ? 100.0f : -100.0f;
    float *scales = simdq_pack_scales(Y, N, d);
    size_t row_bytes = (N + 1) / 2;
    uint8_t *codes = malloc(d * row_bytes);
    simdq_pack_b4(Y, N, d, scales, codes);
    float ks[3]; int64_t kis[3];
    scan_asym_b4_d768_topk(codes, N, q, 3, ks, kis);
    CHECK(kis[0] == 42, "planted top idx=%lld (want 42)", (long long)kis[0]);
    free(Y); free(scales); free(codes);
}

int main(void) {
    srand(9999);
    printf("asym b4 kernel path: %s\n", ASYM_KERNEL_NAME);
    test_random_b4();
    test_planted_b4();
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all asym b4 tests passed\n");
    return 0;
}
```

Append to `CMakeLists.txt`:

```cmake
add_executable(test_kernels_asym_b4 tests/test_kernels_asym_b4.c)
target_compile_options(test_kernels_asym_b4 PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_asym_b4 PRIVATE include)
target_link_libraries(test_kernels_asym_b4 PRIVATE m)
add_test(NAME kernels_asym_b4_native COMMAND test_kernels_asym_b4)

add_executable(test_kernels_asym_b4_avx2 tests/test_kernels_asym_b4.c)
target_compile_options(test_kernels_asym_b4_avx2 PRIVATE -O3 -mavx2 -mfma)
target_include_directories(test_kernels_asym_b4_avx2 PRIVATE include)
target_link_libraries(test_kernels_asym_b4_avx2 PRIVATE m)
add_test(NAME kernels_asym_b4_avx2 COMMAND test_kernels_asym_b4_avx2)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd build && cmake .. && cmake --build . -j
```

Expected: `simdq_kernels_asym_b4.h` not found.

- [ ] **Step 3: Implement `simdq_kernels_asym_b4.h`**

```c
// simdq_kernels_asym_b4.h — asymmetric float×4-bit scan, top-K, D=768.
//
// b=4 packing: 2 codes per byte, 4 bits each, levels {-15,-13,...,+13,+15}
// encoded as {0,1,...,15} mapping to 2*c - 15.

#pragma once

#include "simdq_topk.h"
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#define ASYM_D 768

#if defined(__AVX512F__)
#define ASYM_KERNEL_NAME "AVX512F"
#define ASYM_LANES 16

static inline void scan_asym_b4_d768_topk(const uint8_t *codes, size_t N,
                                          const float *q, int K,
                                          float *out_s, int64_t *out_i) {
    const size_t row_bytes = (N + 1) / 2;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    for (size_t i0 = 0; i0 + ASYM_LANES <= N; i0 += ASYM_LANES) {
        __m512 acc = _mm512_setzero_ps();
        for (size_t w = 0; w < ASYM_D; w++) {
            // 8 bytes hold 16 codes' 4-bit values
            uint64_t packed = *(const uint64_t *)(codes + w * row_bytes + (i0 >> 1));
            float vf[16];
            for (int l = 0; l < 16; l++) {
                uint8_t code = (uint8_t)((packed >> (l * 4)) & 0xF);
                vf[l] = (float)(2 * (int)code - 15);
            }
            __m512 v = _mm512_loadu_ps(vf);
            __m512 qb = _mm512_set1_ps(q[w]);
            acc = _mm512_fmadd_ps(qb, v, acc);
        }
        float sc[16] __attribute__((aligned(64)));
        _mm512_store_ps(sc, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < ASYM_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(i0 + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    // tail
    size_t i = N - (N % ASYM_LANES);
    for (; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < ASYM_D; w++) {
            uint8_t byte = codes[w * row_bytes + (i >> 1)];
            int8_t v = (int8_t)(2 * (int)((byte >> ((i & 1) * 4)) & 0xF) - 15);
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

#elif defined(__AVX2__) && defined(__FMA__)
#define ASYM_KERNEL_NAME "AVX2-FMA"
#define ASYM_LANES 8

static inline void scan_asym_b4_d768_topk(const uint8_t *codes, size_t N,
                                          const float *q, int K,
                                          float *out_s, int64_t *out_i) {
    const size_t row_bytes = (N + 1) / 2;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    for (size_t i0 = 0; i0 + ASYM_LANES <= N; i0 += ASYM_LANES) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t w = 0; w < ASYM_D; w++) {
            uint32_t packed = *(const uint32_t *)(codes + w * row_bytes + (i0 >> 1));
            float vf[8];
            for (int l = 0; l < 8; l++) {
                uint8_t code = (uint8_t)((packed >> (l * 4)) & 0xF);
                vf[l] = (float)(2 * (int)code - 15);
            }
            __m256 v = _mm256_loadu_ps(vf);
            __m256 qb = _mm256_set1_ps(q[w]);
            acc = _mm256_fmadd_ps(qb, v, acc);
        }
        float sc[8] __attribute__((aligned(32)));
        _mm256_store_ps(sc, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < ASYM_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(i0 + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    size_t i = N - (N % ASYM_LANES);
    for (; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < ASYM_D; w++) {
            uint8_t byte = codes[w * row_bytes + (i >> 1)];
            int8_t v = (int8_t)(2 * (int)((byte >> ((i & 1) * 4)) & 0xF) - 15);
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
#endif
```

- [ ] **Step 4: Run tests**

```bash
cd build && cmake --build . -j
ctest -R 'kernels_asym_b4' --output-on-failure
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/include/simdq_kernels_asym_b4.h \
        docuverse/engines/retrieval/simdq/_native/tests/test_kernels_asym_b4.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add b=4 asymmetric scan kernel (AVX-512F + AVX2-FMA, top-K).

Float query x 4-bit DB code (16 levels) over 768 dims. Same kernel
shape as b=2; per-dim unpacks 16/8 4-bit codes per iteration."
```

---

## Task 11: b=4 asymmetric benchmark binary

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b4.c`

- [ ] **Step 1: Write the benchmark**

Same shape as `bench_asym_b2.c`. Three changes: include `_b4`, call `simdq_pack_b4` and `scan_asym_b4_d768_topk`, `bytes = n * d / 2`.

```c
#include "simdq_kernels_asym_b4.h"
#include "simdq_pack.h"
#include "simdq_common.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    size_t n;
    int reps;
    parse_args(argc, argv, &n, &reps);
    int K = (argc > 3) ? atoi(argv[3]) : 100;
    if (K < 1 || K > 256) { fprintf(stderr, "K must be in [1,256]\n"); return 2; }
    srand(42);
    const size_t d = 768;

    float *Y = malloc(n * d * sizeof(float));
    if (!Y) { fprintf(stderr, "alloc Y failed\n"); return 1; }
    for (size_t i = 0; i < n * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, n, d);
    size_t row_bytes = (n + 1) / 2;
    uint8_t *codes = aligned_alloc(64, d * row_bytes);
    simdq_pack_b4(Y, n, d, scales, codes);
    free(Y);

    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;

    float out_s[256]; int64_t out_i[256];
    scan_asym_b4_d768_topk(codes, n, q, K, out_s, out_i);

    double tmin = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        scan_asym_b4_d768_topk(codes, n, q, K, out_s, out_i);
        double dt = now() - t0;
        if (dt < tmin) tmin = dt;
    }
    double cmps = (double)n / tmin;
    double bytes = (double)n * d / 2;
    double gbs = bytes / tmin / 1e9;
    printf("kernel=%s n=%zu K=%d reps=%d  best=%.3fms  cmp/s=%.2fM  GB/s=%.1f  top1=%lld score=%f\n",
           ASYM_KERNEL_NAME, n, K, reps, tmin * 1e3,
           cmps / 1e6, gbs, (long long)out_i[0], out_s[0]);
    free(codes); free(scales);
    return 0;
}
```

Append to `CMakeLists.txt`:

```cmake
add_executable(bench_asym_b4 bench/bench_asym_b4.c)
target_compile_options(bench_asym_b4 PRIVATE ${COMMON_FLAGS})
target_include_directories(bench_asym_b4 PRIVATE include)
target_link_libraries(bench_asym_b4 PRIVATE m)
```

- [ ] **Step 2: Build and run**

```bash
cd build && cmake .. && cmake --build . -j
./bench_asym_b4 1000000 5 100
```

Expected: GB/s should be roughly 4× `bench_asym_b1` (4× more bytes per code).

- [ ] **Step 3: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/bench/bench_asym_b4.c \
        docuverse/engines/retrieval/simdq/_native/CMakeLists.txt
git commit -m "Add bench_asym_b4 microbench."
```

---

## Task 12: Sweep script + record kernel throughput numbers

**Goal:** a one-shot script that runs every benchmark over the canned `{500k, 1M, 50M}` sizes and writes a captured-results document.

**Files:**
- Create: `docuverse/engines/retrieval/simdq/_native/scripts/run_sweep.sh`
- Create: `docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md`

- [ ] **Step 1: Write the sweep script**

```bash
#!/bin/bash
# run_sweep.sh — execute every Plan-1 benchmark over canned sizes and
# print a markdown-friendly table to stdout.
#
# Usage: bash scripts/run_sweep.sh [build_dir]
#   build_dir defaults to ./build
set -euo pipefail

BUILD="${1:-build}"
BINS=(
    bench_hamming_top1
    bench_hamming
    bench_asym_b1
    bench_asym_b2
    bench_asym_b4
)
SIZES=(500000 1000000 50000000)
REPS=(10 10 3)
K=100

if [[ ! -d "$BUILD" ]]; then
    echo "build dir '$BUILD' not found; build first" >&2
    exit 1
fi

THREADS="${OMP_NUM_THREADS:-$(nproc)}"
echo "# simdq Plan 1 kernel sweep — threads=$THREADS, K=$K"
echo
printf "| binary | n | reps | output |\n"
printf "|--------|---|------|--------|\n"
for bin in "${BINS[@]}"; do
    if [[ ! -x "$BUILD/$bin" ]]; then
        printf "| %s | — | — | (binary missing) |\n" "$bin"
        continue
    fi
    for i in "${!SIZES[@]}"; do
        n="${SIZES[$i]}"
        reps="${REPS[$i]}"
        out=$(OMP_NUM_THREADS="$THREADS" "$BUILD/$bin" "$n" "$reps" "$K" 2>&1 | tail -1)
        printf "| %s | %s | %s | %s |\n" "$bin" "$n" "$reps" "$out"
    done
done
```

```bash
chmod +x docuverse/engines/retrieval/simdq/_native/scripts/run_sweep.sh
```

- [ ] **Step 2: Run the sweep and capture output**

```bash
cd docuverse/engines/retrieval/simdq/_native
bash scripts/run_sweep.sh build > BENCHMARKS.md
```

- [ ] **Step 3: Inspect `BENCHMARKS.md` and add commentary**

Open `BENCHMARKS.md`. Below the auto-generated table, add a `## Notes` section that captures:

1. **Hamming top-1 vs top-K parity:** `bench_hamming` (top-K=100) cmp/s should be within 5% of `bench_hamming_top1`. If not, the per-shard heap update is too expensive for K=100 — investigate.
2. **Asymmetric memory bound:** `bench_asym_b{1,2,4}` GB/s should rise roughly proportionally with `b` (b=4 ~4× b=1) up to the system's effective DRAM bandwidth ceiling. If GB/s saturates at b=2, b=4 is bandwidth-bound.
3. **AVX-512 vs AVX2:** kernels build to AVX-512 by default on this hardware. To compare paths, recompile a single bench with `-mavx2 -mfma -mno-avx512f` and re-run; capture both rows in the table.

Example template (fill with actual numbers):

```markdown
## Notes

- Hardware: Cascade Lake, 32 cores, AVX-512F + AVX-512BW (no VPOPCNTDQ).
- bench_hamming top-K=100 ran at X.XX cmp/s vs bench_hamming_top1 at Y.YY cmp/s — within Z%.
- bench_asym_b{1,2,4} GB/s: A.A / B.B / C.C — b=4 ratio to b=1 is D×.
```

- [ ] **Step 4: Commit**

```bash
git add docuverse/engines/retrieval/simdq/_native/scripts/run_sweep.sh \
        docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md
git commit -m "Add Plan-1 kernel sweep script and captured results.

scripts/run_sweep.sh runs every Plan-1 benchmark over {500k, 1M, 50M}
and writes a markdown table to stdout. BENCHMARKS.md records the
sweep output on the dev box for this commit's kernels."
```

---

## Task 13: Final integration check

**Goal:** make sure the whole `_native/` tree builds cleanly from scratch, all tests pass, all benchmarks run, and nothing is left broken.

- [ ] **Step 1: Clean build from scratch**

```bash
cd docuverse/engines/retrieval/simdq/_native
rm -rf build
cmake -S . -B build && cmake --build build -j
```

Expected: clean build, no warnings other than expected unused-variable in stub paths.

- [ ] **Step 2: Run full test suite**

```bash
ctest --test-dir build --output-on-failure
```

Expected output (one line per test, all pass):

```
    Start  1: kernels_hamming_native
1/12 Test  #1: kernels_hamming_native ........... Passed
    Start  2: kernels_hamming_avx2
2/12 Test  #2: kernels_hamming_avx2 ............. Passed
    Start  3: topk
3/12 Test  #3: topk ............................. Passed
    Start  4: kernels_hamming_topk_native
4/12 Test  #4: kernels_hamming_topk_native ...... Passed
    Start  5: kernels_hamming_topk_avx2
5/12 Test  #5: kernels_hamming_topk_avx2 ........ Passed
    Start  6: pack
6/12 Test  #6: pack ............................. Passed
    Start  7: kernels_asym_b1_native
7/12 Test  #7: kernels_asym_b1_native ........... Passed
    Start  8: kernels_asym_b1_avx2
8/12 Test  #8: kernels_asym_b1_avx2 ............. Passed
    Start  9: kernels_asym_b2_native
9/12 Test  #9: kernels_asym_b2_native ........... Passed
    Start 10: kernels_asym_b2_avx2
10/12 Test #10: kernels_asym_b2_avx2 ............ Passed
    Start 11: kernels_asym_b4_native
11/12 Test #11: kernels_asym_b4_native .......... Passed
    Start 12: kernels_asym_b4_avx2
12/12 Test #12: kernels_asym_b4_avx2 ............ Passed
```

- [ ] **Step 3: Run sweep and update BENCHMARKS.md**

```bash
bash scripts/run_sweep.sh build > BENCHMARKS.md
```

If any benchmark output changed materially from Step 3 of Task 12, commit the updated `BENCHMARKS.md`:

```bash
git add docuverse/engines/retrieval/simdq/_native/BENCHMARKS.md
git diff --cached --stat   # confirm only BENCHMARKS.md changed
git commit -m "Refresh BENCHMARKS.md after final clean-build sweep."
```

- [ ] **Step 4: Verify the existing DocUVerse test suite still passes**

The new code is standalone C — no Python, no link into the existing package. But verify nothing accidentally got committed at the repo root or under `docuverse/` outside `_native/`:

```bash
cd /ssd5/raduf/sandbox/docuverse
git diff --stat HEAD~13   # or however many commits this plan added
```

Expected: every changed file is under `docuverse/engines/retrieval/simdq/_native/` or `docs/superpowers/`.

- [ ] **Step 5: Run DocUVerse's existing pytest suite to confirm nothing else broke**

```bash
cd /ssd5/raduf/sandbox/docuverse
conda activate ndocu
python -m pytest tests/ -x --timeout=120
```

Expected: same number of passes/failures as before Plan 1 began. Plan 1 adds no Python; existing tests should be untouched.

If a previously-failing test fails differently or a previously-passing test fails, investigate before declaring Plan 1 complete.

---

## Plan 1 — done

After Task 13, you have:

- `_native/` builds standalone: 5 benchmarks + 8 ctest targets.
- 1-bit symmetric Hamming generalized from top-1 to top-K, threaded.
- Asymmetric `b ∈ {1, 2, 4}` scan kernels for `D=768`, top-K, both AVX-512F and AVX2-FMA paths, single-threaded (threading is Plan 2 territory).
- `BENCHMARKS.md` captures throughput numbers for the dev box.
- No Python yet; nothing in `docuverse/` outside `_native/` has changed.

Plan 2 (next) will add: CPython C-API binding, `SimdqIndex` Python class, numpy ingest pipeline (quantization + projection), on-disk format + mmap rescore, and the threaded asymmetric scan driver.

**Known follow-up for Plan 2:** the macros `KERNEL_NAME` and `LANES` are defined in both `simdq_kernels_hamming.h` and `simdq_kernels_hamming_topk.h` (and `ASYM_KERNEL_NAME`/`ASYM_LANES` in each `simdq_kernels_asym_b*.h`). Plan 1 binaries each include only one of these headers, so there's no collision. Plan 2's Python binding will pull multiple kernel headers into a single translation unit and must rename these (e.g., `HAMMING_TOPK_LANES`, `ASYM_B2_LANES`) before the binding compiles.
