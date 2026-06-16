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
