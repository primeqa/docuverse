// simdq_kernels_hamming_topk.h — top-K Hamming SoA scan kernels.
//
// Pattern: the inner SIMD loop computes per-lane distances exactly as in
// the top-1 path. After each HAMMING_TOPK_LANES-block, distances are stored to a small
// scratch array and offered to a per-shard simdq_topk_t heap. The hot
// loop is unchanged from the top-1 kernel; the only added work is a
// threshold compare + heap offer per HAMMING_TOPK_LANES-block (amortized over HAMMING_TOPK_LANES
// distances). For K << n this overhead is negligible.

#pragma once

#ifdef HAMMING_TOPK_KERNEL_NAME
#  error "simdq_kernels_hamming.h and simdq_kernels_hamming_topk.h must not be included in the same TU"
#endif

#include "simdq_common.h"
#include "simdq_topk.h"
#include <assert.h>
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
#define HAMMING_TOPK_KERNEL_NAME "AVX512-VPOPCNTDQ"
#define HAMMING_TOPK_LANES 8

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
    assert(K > 0 && K <= 256);
    int64_t hkeys[256], hidxs[256];     // K up to 256 supported on stack
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    size_t i = i0;
    for (; i + HAMMING_TOPK_LANES <= i1; i += HAMMING_TOPK_LANES) {
        __m512i acc = _mm512_setzero_si512();
        for (int w = 0; w < WORDS; w++) {
            __m512i d = _mm512_loadu_si512(dbT + (size_t)w * n + i);
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
        int h = hamming_soa(dbT, n, k, q);
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

static inline void scan_shard_topk(const uint64_t *dbT, size_t n,
                                   size_t i0, size_t i1,
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
        for (int l = 0; l < HAMMING_TOPK_LANES; l++)
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

#if defined(HAMMING_TOPK_KERNEL_NAME) && defined(_OPENMP)
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
