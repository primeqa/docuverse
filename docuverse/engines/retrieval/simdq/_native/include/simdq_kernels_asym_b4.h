// simdq_kernels_asym_b4.h — asymmetric float×4-bit scan, top-K, runtime `d`.
//
// b=4 packing: 2 codes per byte, 4 bits each, levels {-15,-13,...,+13,+15}
// encoded as {0,1,...,15} mapping to 2*c - 15.

#pragma once

#include "simdq_topk.h"
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#if defined(__AVX512F__)
#define ASYM_B4_KERNEL_NAME "AVX512F"
#define ASYM_B4_LANES 16

/*
 * N stays (needed for row_bytes = (N + 1) / 2, the dim-stride).
 * d is the number of dimensions (inner-loop bound; replaces compile-time D).
 * i0/i1 define the code range to scan.
 */
static inline void scan_asym_b4_shard_topk(const uint8_t *codes, size_t N, size_t d,
                                           size_t i0, size_t i1,
                                           const float *q, int K,
                                           float *out_s, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    const size_t row_bytes = (N + 1) / 2;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

#if defined(__AVX512VBMI__) && defined(__AVX512VL__)
    // Lever 1 — vectorized 4-bit unpack (see asym_b2 for the technique).
    // vpmultishiftqb extracts the 16 nibbles of `packed` into 16 bytes; vpshufb
    // maps code -> level via a full 16-entry LUT (level = 2*code - 15).
    // Bit-identical to the scalar path.
    const __m128i k_ctrl = _mm_setr_epi8(0, 4, 8, 12, 16, 20, 24, 28,
                                         32, 36, 40, 44, 48, 52, 56, 60);
    const __m128i k_lut  = _mm_setr_epi8(-15, -13, -11, -9, -7, -5, -3, -1,
                                         1, 3, 5, 7, 9, 11, 13, 15);
    const __m128i k_lo4  = _mm_set1_epi8(0xF);
#endif

    for (size_t ii = i0; ii + ASYM_B4_LANES <= i1; ii += ASYM_B4_LANES) {
        __m512 acc = _mm512_setzero_ps();
        for (size_t w = 0; w < d; w++) {
            // 8 bytes hold 16 codes' 4-bit values
            uint64_t packed = *(const uint64_t *)(codes + w * row_bytes + (ii >> 1));
            __m512 v;
#if defined(__AVX512VBMI__) && defined(__AVX512VL__)
            __m128i data  = _mm_set1_epi64x((long long)packed);
            __m128i bytes = _mm_multishift_epi64_epi8(k_ctrl, data);
            __m128i code8 = _mm_and_si128(bytes, k_lo4);
            __m128i lev8  = _mm_shuffle_epi8(k_lut, code8);          // int8 levels
            v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lev8));
#else
            float vf[16];
            for (int l = 0; l < 16; l++) {
                uint8_t code = (uint8_t)((packed >> (l * 4)) & 0xF);
                vf[l] = (float)(2 * (int)code - 15);
            }
            v = _mm512_loadu_ps(vf);
#endif
            __m512 qb = _mm512_set1_ps(q[w]);
            acc = _mm512_fmadd_ps(qb, v, acc);
        }
        float sc[16] __attribute__((aligned(64)));
        _mm512_store_ps(sc, acc);
        int64_t thr = simdq_topk_threshold(&heap);
        for (int l = 0; l < ASYM_B4_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(ii + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    // tail
    size_t i = i1 - ((i1 - i0) % ASYM_B4_LANES);
    for (; i < i1; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
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

static inline void scan_asym_b4_topk(const uint8_t *codes, size_t N, size_t d,
                                     const float *q, int K,
                                     float *out_s, int64_t *out_i) {
    scan_asym_b4_shard_topk(codes, N, d, 0, N, q, K, out_s, out_i);
}

#elif defined(__AVX2__) && defined(__FMA__)
#define ASYM_B4_KERNEL_NAME "AVX2-FMA"
#define ASYM_B4_LANES 8

/*
 * N stays (needed for row_bytes = (N + 1) / 2, the dim-stride).
 * d is the number of dimensions (inner-loop bound; replaces compile-time D).
 * i0/i1 define the code range to scan.
 */
static inline void scan_asym_b4_shard_topk(const uint8_t *codes, size_t N, size_t d,
                                           size_t i0, size_t i1,
                                           const float *q, int K,
                                           float *out_s, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    const size_t row_bytes = (N + 1) / 2;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    for (size_t ii = i0; ii + ASYM_B4_LANES <= i1; ii += ASYM_B4_LANES) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t w = 0; w < d; w++) {
            uint32_t packed = *(const uint32_t *)(codes + w * row_bytes + (ii >> 1));
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
        for (int l = 0; l < ASYM_B4_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(ii + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    size_t i = i1 - ((i1 - i0) % ASYM_B4_LANES);
    for (; i < i1; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
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

static inline void scan_asym_b4_topk(const uint8_t *codes, size_t N, size_t d,
                                     const float *q, int K,
                                     float *out_s, int64_t *out_i) {
    scan_asym_b4_shard_topk(codes, N, d, 0, N, q, K, out_s, out_i);
}
#endif

#if defined(_OPENMP)
#include <omp.h>

/*
 * Threaded top-K asymmetric b=4 scan over [0, N). Each thread runs
 * scan_asym_b4_shard_topk on its range, then a critical-section
 * merge folds per-thread results into the global top-K. Per-thread
 * results from a partial-fill shard set unused entries to score=-INF
 * and idx=-1; the merge skips those.
 */
static inline void scan_asym_b4_topk_parallel(const uint8_t *codes, size_t N, size_t d,
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
        // Shard start must be byte-aligned: 4-bit codes are packed 2/byte and the
        // SIMD path reads vector ii from byte (ii>>1); an unaligned i0 reads the
        // wrong codes. Round chunk up to a multiple of 64 so every i0 is aligned.
        chunk = (chunk + 63) & ~(size_t)63;
        size_t i0 = (size_t)t * chunk;
        size_t i1 = i0 + chunk < N ? i0 + chunk : N;
        if (i0 < i1) {
            float ls[256]; int64_t li[256];
            // pre-fill so a partial shard is detectable
            for (int r = 0; r < K; r++) { ls[r] = -INFINITY; li[r] = -1; }
            scan_asym_b4_shard_topk(codes, N, d, i0, i1, q, K, ls, li);
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
