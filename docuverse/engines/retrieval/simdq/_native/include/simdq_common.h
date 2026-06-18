// simdq_common.h — shared helpers for SIMD-quantized scan kernels.
//
// Each binary code is D bits = words x uint64_t (words = D/64).
// Two layouts appear across the kernels:
//   AoS: db[i*words + w] = word w of code i  — cache-friendly per code
//   SoA: dbT[w*n + i]    = word w of code i  — cache-friendly across codes;
//        one wide SIMD load grabs word w of several consecutive codes.
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <nmmintrin.h>

#ifndef NQ
#define NQ 8      // queries per batch (batched kernels; override with -DNQ=...)
#endif

/*
 * Current time from the monotonic clock (CLOCK_MONOTONIC), immune to
 * system clock adjustments. Returns seconds as a double; the fractional
 * part carries sub-second (nanosecond) precision.
 */
static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * 64-bit pseudo-random value built from three rand() draws combined with
 * shifts and XORs so all 64 bits are populated. Non-cryptographic; the
 * stream is reproducible after srand() with a fixed seed.
 */
static inline uint64_t rnd64(void) {
    return ((uint64_t)rand() << 42) ^ ((uint64_t)rand() << 21) ^ rand();
}

/*
 * Fills p[0..count) with rnd64() values; used to generate random database
 * codes and query vectors.
 */
static inline void fill_rnd64(uint64_t *p, size_t count) {
    for (size_t i = 0; i < count; i++) p[i] = rnd64();
}

/*
 * Allocates storage for n binary codes of `words` uint64_t each
 * (n * words * 8 bytes), aligned to 64 bytes so codes and SIMD loads
 * sit on cache-line boundaries.
 * Returns NULL on failure; the caller releases it with free().
 */
static inline uint64_t *alloc_codes(size_t n, size_t words) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * words * 8)) return NULL;
    return (uint64_t *)p;
}

/*
 * Parses the command line shared by all benchmarks: prog [n] [reps].
 *   n     number of database codes (default 1,000,000)
 *   reps  timed repetitions; benchmarks report the minimum (default 5)
 */
static inline void parse_args(int argc, char **argv, size_t *n, int *reps) {
    *n    = (argc > 1) ? strtoull(argv[1], NULL, 10) : 1000000;
    *reps = (argc > 2) ? atoi(argv[2]) : 5;
}

/*
 * Scalar Hamming distance between query q and code k of an SoA database
 * (word stride n): sum over `words` words of popcnt(q[w] ^ d[w]).
 * Used by the SIMD kernels for the n % LANES tail codes.
 */
static inline int hamming_soa(const uint64_t *dbT, size_t n, size_t words,
                              size_t k, const uint64_t *q) {
    int h = 0;
    for (size_t w = 0; w < words; w++)
        h += (int)_mm_popcnt_u64(q[w] ^ dbT[w * n + k]);
    return h;
}

/*
 * Baseline kernel: scalar linear scan of an AoS database. For each code,
 * XORs the `words` word pairs with the query and sums hardware popcounts
 * (_mm_popcnt_u64, the single-cycle SSE4.2 POPCNT instruction):
 *   h = sum_w popcnt(q[w] ^ d[w])
 * Returns the index of the code with the smallest distance (top-1).
 */
static inline size_t scan_scalar_aos(const uint64_t *db, size_t n, size_t words,
                                     const uint64_t *q) {
    int best = 1 << 30; size_t bi = 0;
    for (size_t i = 0; i < n; i++) {
        const uint64_t *d = db + i * words;
        int h = 0;
        for (size_t w = 0; w < words; w++)
            h += (int)_mm_popcnt_u64(q[w] ^ d[w]);
        if (h < best) { best = h; bi = i; }
    }
    return bi;
}

#ifdef _OPENMP
/*
 * Parallel SoA fill, one word-plane per thread with a private rand_r
 * stream (seed 42 + w, so contents are deterministic regardless of thread
 * count). First-touch places pages on the touching thread's NUMA node,
 * and parallel init is also just much faster for multi-GB arrays.
 */
static inline void fill_soa_parallel(uint64_t *dbT, size_t n, size_t words) {
    #pragma omp parallel for schedule(static)
    for (size_t w = 0; w < words; w++) {
        unsigned int seed = 42 + (unsigned)w;
        for (size_t i = 0; i < n; i++) {
            uint64_t a = (uint64_t)rand_r(&seed) << 42;
            uint64_t b = (uint64_t)rand_r(&seed) << 21;
            dbT[w * n + i] = a ^ b ^ rand_r(&seed);
        }
    }
}
#endif
