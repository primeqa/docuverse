// simdq_kernels_asym_b4.h — asymmetric float×4-bit scan, top-K, D=768.
//
// b=4 packing: 2 codes per byte, 4 bits each, levels {-15,-13,...,+13,+15}
// encoded as {0,1,...,15} mapping to 2*c - 15.

#pragma once

#include "simdq_topk.h"
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#define ASYM_B4_D 768

#if defined(__AVX512F__)
#define ASYM_B4_KERNEL_NAME "AVX512F"
#define ASYM_B4_LANES 16

static inline void scan_asym_b4_d768_topk(const uint8_t *codes, size_t N,
                                          const float *q, int K,
                                          float *out_s, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    const size_t row_bytes = (N + 1) / 2;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    for (size_t i0 = 0; i0 + ASYM_B4_LANES <= N; i0 += ASYM_B4_LANES) {
        __m512 acc = _mm512_setzero_ps();
        for (size_t w = 0; w < ASYM_B4_D; w++) {
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
        for (int l = 0; l < ASYM_B4_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(i0 + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    // tail
    size_t i = N - (N % ASYM_B4_LANES);
    for (; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < ASYM_B4_D; w++) {
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
#define ASYM_B4_KERNEL_NAME "AVX2-FMA"
#define ASYM_B4_LANES 8

static inline void scan_asym_b4_d768_topk(const uint8_t *codes, size_t N,
                                          const float *q, int K,
                                          float *out_s, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    const size_t row_bytes = (N + 1) / 2;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    for (size_t i0 = 0; i0 + ASYM_B4_LANES <= N; i0 += ASYM_B4_LANES) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t w = 0; w < ASYM_B4_D; w++) {
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
        for (int l = 0; l < ASYM_B4_LANES; l++) {
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(i0 + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    size_t i = N - (N % ASYM_B4_LANES);
    for (; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < ASYM_B4_D; w++) {
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
