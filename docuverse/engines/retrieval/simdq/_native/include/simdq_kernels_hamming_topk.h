// simdq_kernels_hamming_topk.h — top-K Hamming SoA scan kernels.
//
// Pattern: the inner SIMD loop computes per-lane distances exactly as in
// the top-1 path. After each HAMMING_TOPK_LANES-block, distances are stored to a small
// scratch array and offered to a per-shard simdq_topk_t heap. The hot
// loop is unchanged from the top-1 kernel; the only added work is a
// threshold compare + heap offer per HAMMING_TOPK_LANES-block (amortized over HAMMING_TOPK_LANES
// distances). For K << n this overhead is negligible.

#pragma once

#ifdef KERNEL_NAME
#  error "simdq_kernels_hamming.h and simdq_kernels_hamming_topk.h must not be included in the same TU"
#endif

#include "simdq_common.h"
#include "simdq_topk.h"
#include <assert.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <string.h>

// Reference top-1 helper used by the K=1 test (independent of SIMD).
static inline size_t ref_scan_soa_top1(const uint64_t *dbT, size_t n,
                                       size_t words, size_t i0, size_t i1,
                                       const uint64_t *q, int *out_d) {
    int best = INT32_MAX; size_t bi = i0;
    for (size_t k = i0; k < i1; k++) {
        int h = hamming_soa(dbT, n, words, k, q);
        if (h < best) { best = h; bi = k; }
    }
    *out_d = best;
    return bi;
}

#if defined(__AVX512VPOPCNTDQ__)
#define HAMMING_TOPK_KERNEL_NAME "AVX512-VPOPCNTDQ"
#define HAMMING_TOPK_LANES 8

/*
 * Single-query top-K Hamming SoA scan over codes [i0, i1). Maintains a
 * bounded max-heap of size K; after each 8-code block we read out the 8
 * per-lane distances, compare against the heap threshold, and offer the
 * survivors. Returns the K smallest distances (ascending) and their
 * indices in out_d / out_i.
 * words is the number of 64-bit words per code (inner-loop bound; = D / 64).
 * i0/i1 define the code range to scan.
 */
static inline void scan_hamming_shard_topk(const uint64_t *dbT, size_t n,
                                           size_t words, size_t i0, size_t i1,
                                           const uint64_t *q, int K,
                                           int64_t *out_d, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    int64_t hkeys[256], hidxs[256];     // K up to 256 supported on stack
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    size_t i = i0;
    for (; i + HAMMING_TOPK_LANES <= i1; i += HAMMING_TOPK_LANES) {
        __m512i acc = _mm512_setzero_si512();
        for (size_t w = 0; w < words; w++) {
            __m512i d = _mm512_loadu_si512(dbT + w * n + i);
            acc = _mm512_add_epi64(acc,
                  _mm512_popcnt_epi64(_mm512_xor_si512(d, _mm512_set1_epi64(q[w]))));
        }
        int64_t d8[HAMMING_TOPK_LANES];
        _mm512_storeu_si512(d8, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < HAMMING_TOPK_LANES; l++)
            if (d8[l] < thr) {
                simdq_topk_offer(&heap, d8[l], (int64_t)(i + l));
                thr = simdq_topk_threshold(&heap);
            }
    }
    for (size_t k = i; k < i1; k++) {
        int h = hamming_soa(dbT, n, words, k, q);
        if ((int64_t)h < simdq_topk_threshold(&heap))
            simdq_topk_offer(&heap, (int64_t)h, (int64_t)k);
    }
    simdq_topk_extract_sorted(&heap, out_d, out_i);
}

#elif defined(__AVX2__)
#define HAMMING_TOPK_KERNEL_NAME "AVX2-nibbleLUT"
#define HAMMING_TOPK_LANES 4

static inline __m256i popcnt_bytes_avx2(__m256i x, __m256i lut, __m256i m0f) {
    __m256i lo = _mm256_and_si256(x, m0f);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), m0f);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                           _mm256_shuffle_epi8(lut, hi));
}

/*
 * words is the number of 64-bit words per code (inner-loop bound; = D / 64).
 * Byte-count accumulators overflow at words >= 24 (words * 8 < 255);
 * callers must ensure words <= 23.
 * i0/i1 define the code range to scan.
 */
static inline void scan_hamming_shard_topk(const uint64_t *dbT, size_t n,
                                           size_t words, size_t i0, size_t i1,
                                           const uint64_t *q, int K,
                                           int64_t *out_d, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();

    size_t i = i0;
    for (; i + HAMMING_TOPK_LANES <= i1; i += HAMMING_TOPK_LANES) {
        __m256i accb = zero;
        for (size_t w = 0; w < words; w++) {
            __m256i d = _mm256_loadu_si256(
                (const __m256i *)(dbT + w * n + i));
            __m256i x = _mm256_xor_si256(d, _mm256_set1_epi64x(q[w]));
            accb = _mm256_add_epi8(accb, popcnt_bytes_avx2(x, lut, m0f));
        }
        __m256i sums = _mm256_sad_epu8(accb, zero);
        int64_t d4[4];
        _mm256_storeu_si256((__m256i *)d4, sums);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < HAMMING_TOPK_LANES; l++)
            if (d4[l] < thr) {
                simdq_topk_offer(&heap, d4[l], (int64_t)(i + l));
                thr = simdq_topk_threshold(&heap);
            }
    }
    for (size_t k = i; k < i1; k++) {
        int h = hamming_soa(dbT, n, words, k, q);
        if ((int64_t)h < simdq_topk_threshold(&heap))
            simdq_topk_offer(&heap, (int64_t)h, (int64_t)k);
    }
    simdq_topk_extract_sorted(&heap, out_d, out_i);
}

#elif defined(__aarch64__) || defined(__ARM_NEON)
#define HAMMING_TOPK_KERNEL_NAME "NEON-vcnt"
#define HAMMING_TOPK_LANES 2

/*
 * Per-lane popcount of a uint64x2 via native NEON vcntq_u8 + pairwise
 * widening add chain (byte->u16->u32->u64). Same helper as the top-1
 * kernel uses; duplicated locally because these two headers are never
 * included in the same TU (see #error guard above).
 */
static inline uint64x2_t popcnt_u64x2_topk(uint64x2_t x) {
    uint8x16_t cnts = vcntq_u8(vreinterpretq_u8_u64(x));
    return vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(cnts)));
}

/*
 * Single-query top-K Hamming SoA scan over codes [i0, i1) on NEON.
 * Inner loop mirrors scan_hamming_shard_topk for AVX2/AVX-512: XOR each
 * word against broadcast query, popcount, accumulate into a per-lane
 * u64 distance. After each 2-code block, read out the 2 per-lane
 * distances and offer to a bounded max-heap of size K.
 */
static inline void scan_hamming_shard_topk(const uint64_t *dbT, size_t n,
                                           size_t words, size_t i0, size_t i1,
                                           const uint64_t *q, int K,
                                           int64_t *out_d, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    size_t i = i0;
    for (; i + HAMMING_TOPK_LANES <= i1; i += HAMMING_TOPK_LANES) {
        uint64x2_t acc = vdupq_n_u64(0);
        for (size_t w = 0; w < words; w++) {
            uint64x2_t d = vld1q_u64(dbT + w * n + i);
            uint64x2_t x = veorq_u64(d, vdupq_n_u64(q[w]));
            acc = vaddq_u64(acc, popcnt_u64x2_topk(x));
        }
        uint64_t d2[HAMMING_TOPK_LANES];
        vst1q_u64(d2, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < HAMMING_TOPK_LANES; l++)
            if ((int64_t)d2[l] < thr) {
                simdq_topk_offer(&heap, (int64_t)d2[l], (int64_t)(i + l));
                thr = simdq_topk_threshold(&heap);
            }
    }
    for (size_t k = i; k < i1; k++) {
        int h = hamming_soa(dbT, n, words, k, q);
        if ((int64_t)h < simdq_topk_threshold(&heap))
            simdq_topk_offer(&heap, (int64_t)h, (int64_t)k);
    }
    simdq_topk_extract_sorted(&heap, out_d, out_i);
}
#endif

#if defined(HAMMING_TOPK_KERNEL_NAME) && defined(_OPENMP)
#include <omp.h>

/*
 * Threaded top-K Hamming scan. Each thread runs scan_hamming_shard_topk on
 * its range, then a critical-section merge folds per-thread heaps into the
 * global top-K. For T threads merging K entries each, merge cost is
 * O(T*K*log K) — negligible vs the scan.
 */
static inline void scan_hamming_topk_parallel(const uint64_t *dbT, size_t n,
                                              size_t words, const uint64_t *q,
                                              int K, int64_t *gd, int64_t *gi) {
    assert(K > 0 && K <= 256);
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
            for (int r = 0; r < K; r++) { ld[r] = INT64_MAX; li[r] = -1; }
            scan_hamming_shard_topk(dbT, n, words, i0, i1, q, K, ld, li);
            #pragma omp critical
            for (int r = 0; r < K; r++)
                if (li[r] >= 0 && ld[r] < simdq_topk_threshold(&global))
                    simdq_topk_offer(&global, ld[r], li[r]);
        }
    }
    simdq_topk_extract_sorted(&global, gd, gi);
}

/*
 * Top-K Hamming scan over a sub-range [r0, r1) of an SoA database (word stride
 * still n, the full database). Same threaded shard+merge as
 * scan_hamming_topk_parallel but bounded to [r0, r1) — used by the IVF path to
 * scan one cluster's contiguous index range without copying or transposing.
 * Returned indices are absolute (in [r0, r1)).
 */
static inline void scan_hamming_topk_parallel_range(const uint64_t *dbT, size_t n,
                                                    size_t words,
                                                    size_t r0, size_t r1,
                                                    const uint64_t *q, int K,
                                                    int64_t *gd, int64_t *gi) {
    assert(K > 0 && K <= 256);
    int64_t gkeys[256], gidxs[256];
    simdq_topk_t global;
    simdq_topk_init(&global, K, gkeys, gidxs);

    #pragma omp parallel
    {
        int t = omp_get_thread_num(), T = omp_get_num_threads();
        size_t span = r1 > r0 ? r1 - r0 : 0;
        size_t chunk = (span + (size_t)T - 1) / (size_t)T;
        size_t i0 = r0 + (size_t)t * chunk;
        size_t i1 = i0 + chunk < r1 ? i0 + chunk : r1;
        if (i0 < i1) {
            int64_t ld[256], li[256];
            for (int r = 0; r < K; r++) { ld[r] = INT64_MAX; li[r] = -1; }
            scan_hamming_shard_topk(dbT, n, words, i0, i1, q, K, ld, li);
            #pragma omp critical
            for (int r = 0; r < K; r++)
                if (li[r] >= 0 && ld[r] < simdq_topk_threshold(&global))
                    simdq_topk_offer(&global, ld[r], li[r]);
        }
    }
    simdq_topk_extract_sorted(&global, gd, gi);
}
#endif
