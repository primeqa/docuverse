// simdq_kernels_asym_b2.h — asymmetric float×2-bit scan, top-K, runtime `d`.
//
// b=2 packing: 4 codes per byte, 2 bits each, levels {-3,-1,+1,+3}
// encoded as {0,1,2,3}. Decoding: code -> levels[code] in int8.

#pragma once

#include "simdq_topk.h"
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#if defined(__AVX512F__)
#define ASYM_B2_KERNEL_NAME "AVX512F"
#define ASYM_B2_LANES 16

/*
 * Per dim w, we need ASYM_B2_LANES=16 codes' decoded level. With b=2 and 4
 * codes per byte, that's 4 bytes (16 codes / 4 codes-per-byte). The 16
 * 2-bit fields are unpacked into 16 int8 values via a vpshufb-style
 * lookup, then converted to fp32 and FMA'd against q'[w] broadcast.
 *
 * N stays (needed for row_bytes = (N + 3) / 4, the dim-stride).
 * d is the number of dimensions (inner-loop bound; replaces compile-time D).
 * i0/i1 define the code range to scan.
 */
static inline void scan_asym_b2_shard_topk(const uint8_t *codes, size_t N, size_t d,
                                           size_t i0, size_t i1,
                                           const float *q, int K,
                                           float *out_s, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    const size_t row_bytes = (N + 3) / 4;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    for (size_t ii = i0; ii + ASYM_B2_LANES <= i1; ii += ASYM_B2_LANES) {
        __m512 acc = _mm512_setzero_ps();
        for (size_t w = 0; w < d; w++) {
            // 4 bytes from this dim row, holding 16 codes' 2-bit values
            uint32_t packed = *(const uint32_t *)(codes + w * row_bytes + (ii >> 2));
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
        for (int l = 0; l < ASYM_B2_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(ii + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    // tail
    size_t i = i1 - ((i1 - i0) % ASYM_B2_LANES);
    static const int8_t levels[4] = {-3, -1, 1, 3};
    for (; i < i1; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
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

static inline void scan_asym_b2_topk(const uint8_t *codes, size_t N, size_t d,
                                     const float *q, int K,
                                     float *out_s, int64_t *out_i) {
    scan_asym_b2_shard_topk(codes, N, d, 0, N, q, K, out_s, out_i);
}

#elif defined(__AVX2__) && defined(__FMA__)
#define ASYM_B2_KERNEL_NAME "AVX2-FMA"
#define ASYM_B2_LANES 8

/*
 * N stays (needed for row_bytes = (N + 3) / 4, the dim-stride).
 * d is the number of dimensions (inner-loop bound; replaces compile-time D).
 * i0/i1 define the code range to scan.
 */
static inline void scan_asym_b2_shard_topk(const uint8_t *codes, size_t N, size_t d,
                                           size_t i0, size_t i1,
                                           const float *q, int K,
                                           float *out_s, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    const size_t row_bytes = (N + 3) / 4;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);
    static const int8_t levels[4] = {-3, -1, 1, 3};

    for (size_t ii = i0; ii + ASYM_B2_LANES <= i1; ii += ASYM_B2_LANES) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t w = 0; w < d; w++) {
            // 2 bytes hold 8 codes' 2-bit values
            uint16_t packed = *(const uint16_t *)(codes + w * row_bytes + (ii >> 2));
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
        for (int l = 0; l < ASYM_B2_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(ii + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    size_t i = i1 - ((i1 - i0) % ASYM_B2_LANES);
    for (; i < i1; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
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

static inline void scan_asym_b2_topk(const uint8_t *codes, size_t N, size_t d,
                                     const float *q, int K,
                                     float *out_s, int64_t *out_i) {
    scan_asym_b2_shard_topk(codes, N, d, 0, N, q, K, out_s, out_i);
}
#endif

#if defined(_OPENMP)
#include <omp.h>

/*
 * Threaded top-K asymmetric b=2 scan over [0, N). Each thread runs
 * scan_asym_b2_shard_topk on its range, then a critical-section
 * merge folds per-thread results into the global top-K. Per-thread
 * results from a partial-fill shard set unused entries to score=-INF
 * and idx=-1; the merge skips those.
 */
static inline void scan_asym_b2_topk_parallel(const uint8_t *codes, size_t N, size_t d,
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
            scan_asym_b2_shard_topk(codes, N, d, i0, i1, q, K, ls, li);
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
