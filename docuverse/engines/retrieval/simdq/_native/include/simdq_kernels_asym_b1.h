// simdq_kernels_asym_b1.h — asymmetric float×1-bit scan, top-K, runtime `d`.
//
// Layout: codes is dim-major SoA, b=1 packing (8 codes per byte).
//   codes[w * row_bytes + (i >> 3)]  bit (i & 7)  = sign(y_i[w]/s_i)
// Score: s_i = sum_w q'[w] * v_i[w]   where v_i[w] in {-1, +1}.
//
// Per-dim inner loop expands 8 packed bits into 8 ±1 floats and FMAs them
// against q'[w] broadcast. Per-ASYM_B1_LANES-codes block, 8/16 codes' partial
// scores are accumulated; per-block we offer them to a top-K heap.

#pragma once

#include "simdq_topk.h"
#include <assert.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <math.h>
#include <stdint.h>

#if defined(__AVX512F__)
#define ASYM_B1_KERNEL_NAME "AVX512F"
#define ASYM_B1_LANES 16     // floats per AVX-512 register; 16 codes per block

/*
 * Shard top-K asymmetric b=1 scan over codes [i0, i1). Inner loop:
 * for each dim w, broadcast q'[w], unpack 16 codes' bit (one byte holds
 * 8 codes; 16 codes = 2 bytes via mask -> int8 -> float conversion),
 * FMA into per-lane accumulator. After d dims, store the 16 partial
 * scores and offer to top-K heap.
 *
 * N stays (needed for row_bytes = (N + 7) / 8, the dim-stride).
 * d is the number of dimensions (inner-loop bound; replaces compile-time D).
 * i0/i1 define the code range to scan.
 */
static inline void scan_asym_b1_shard_topk(const uint8_t *codes, size_t N, size_t d,
                                           size_t i0, size_t i1,
                                           const float *q, int K,
                                           float *out_s, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    const size_t row_bytes = (N + 7) / 8;
    // bounded max-heap of NEGATIVE scores, so larger -> "smaller" -> top
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    // Process 16 codes at a time; we need 2 bytes per dim (16 / 8).
    for (size_t ii = i0; ii + ASYM_B1_LANES <= i1; ii += ASYM_B1_LANES) {
        __m512 acc = _mm512_setzero_ps();
        for (size_t w = 0; w < d; w++) {
            // 2 bytes from this dim's row, holding codes [ii, ii+16)
            uint16_t bits = ((uint16_t)codes[w * row_bytes + (ii >> 3) + 1] << 8)
                          |  (uint16_t)codes[w * row_bytes + (ii >> 3)];
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
        for (int l = 0; l < ASYM_B1_LANES; l++) {
            // negate to convert top-largest into top-smallest
            int64_t neg = -(int64_t)(sc[l] * (float)(1 << 20));   // fixed-point key
            if (neg < thr) {
                simdq_topk_offer(&heap, neg, (int64_t)(ii + l));
                thr = simdq_topk_threshold(&heap);
            }
        }
    }
    // tail: scalar
    size_t i = i1 - ((i1 - i0) % ASYM_B1_LANES);
    for (; i < i1; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
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

static inline void scan_asym_b1_topk(const uint8_t *codes, size_t N, size_t d,
                                     const float *q, int K,
                                     float *out_s, int64_t *out_i) {
    scan_asym_b1_shard_topk(codes, N, d, 0, N, q, K, out_s, out_i);
}

#elif defined(__AVX2__) && defined(__FMA__)
#define ASYM_B1_KERNEL_NAME "AVX2-FMA"
#define ASYM_B1_LANES 8

/*
 * N stays (needed for row_bytes = (N + 7) / 8, the dim-stride).
 * d is the number of dimensions (inner-loop bound; replaces compile-time D).
 * i0/i1 define the code range to scan.
 */
static inline void scan_asym_b1_shard_topk(const uint8_t *codes, size_t N, size_t d,
                                           size_t i0, size_t i1,
                                           const float *q, int K,
                                           float *out_s, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    const size_t row_bytes = (N + 7) / 8;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    for (size_t ii = i0; ii + ASYM_B1_LANES <= i1; ii += ASYM_B1_LANES) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t w = 0; w < d; w++) {
            // Broadcast the byte to all 32 lanes; AND against per-lane single-bit
            // masks; lift the {zero, nonzero} result to a {-1, 0} per-byte mask
            // via cmpeq-with-zero (works for bit 7, where the masked value is
            // (char)128 = -128 — a signed cmpgt against zero would mis-classify
            // lane 7 because -128 is not > 0).
            uint8_t bits = codes[w * row_bytes + (ii >> 3)];
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
        }
        float sc[8] __attribute__((aligned(32)));
        _mm256_store_ps(sc, acc);
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
        for (size_t w = 0; w < d; w++) {
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

static inline void scan_asym_b1_topk(const uint8_t *codes, size_t N, size_t d,
                                     const float *q, int K,
                                     float *out_s, int64_t *out_i) {
    scan_asym_b1_shard_topk(codes, N, d, 0, N, q, K, out_s, out_i);
}

#elif defined(__aarch64__) || defined(__ARM_NEON)
#define ASYM_B1_KERNEL_NAME "NEON"
#define ASYM_B1_LANES 8

/*
 * NEON port. Matches AVX2 lane count (8 codes/block, one byte/dim) using
 * two float32x4_t accumulators. Bit expansion mirrors the AVX2 trick:
 * broadcast the byte to 8 u8 lanes, AND against per-lane single-bit masks,
 * cmpeq-with-zero to get 0xFF-where-clear, sign-extend to 32-bit per lane,
 * and vbslq_f32-select -1.0 (clear) vs +1.0 (set).
 *
 * N stays (needed for row_bytes = (N + 7) / 8, the dim-stride).
 * d is the number of dimensions.
 * i0/i1 define the code range to scan.
 */
static inline void scan_asym_b1_shard_topk(const uint8_t *codes, size_t N, size_t d,
                                           size_t i0, size_t i1,
                                           const float *q, int K,
                                           float *out_s, int64_t *out_i) {
    assert(K > 0 && K <= 256);
    const size_t row_bytes = (N + 7) / 8;
    int64_t hkeys[256], hidxs[256];
    simdq_topk_t heap;
    simdq_topk_init(&heap, K, hkeys, hidxs);

    static const uint8_t lane_mask_data[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x8_t lane_mask = vld1_u8(lane_mask_data);
    const float32x4_t pos1 = vdupq_n_f32( 1.0f);
    const float32x4_t neg1 = vdupq_n_f32(-1.0f);

    for (size_t ii = i0; ii + ASYM_B1_LANES <= i1; ii += ASYM_B1_LANES) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        for (size_t w = 0; w < d; w++) {
            uint8_t bits = codes[w * row_bytes + (ii >> 3)];
            uint8x8_t bbroad = vdup_n_u8(bits);
            uint8x8_t isset  = vand_u8(bbroad, lane_mask);
            // 0xFF where bit is CLEAR, 0x00 where SET (cmpeq-zero is sign-agnostic).
            uint8x8_t clear8 = vceq_u8(isset, vdup_n_u8(0));
            // Sign-extend byte mask to int16, then int32, preserving 0xFF -> 0xFFFFFFFF.
            int16x8_t clear16 = vmovl_s8(vreinterpret_s8_u8(clear8));
            uint32x4_t clear_lo = vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(clear16)));
            uint32x4_t clear_hi = vreinterpretq_u32_s32(vmovl_s16(vget_high_s16(clear16)));
            // vbslq_f32(mask, a, b): (mask & a) | (~mask & b).
            // clear lane (mask=all-ones) -> pick a=neg1; set lane (mask=0) -> pick b=pos1.
            float32x4_t v_lo = vbslq_f32(clear_lo, neg1, pos1);
            float32x4_t v_hi = vbslq_f32(clear_hi, neg1, pos1);
            float32x4_t qb = vdupq_n_f32(q[w]);
            acc0 = vfmaq_f32(acc0, qb, v_lo);
            acc1 = vfmaq_f32(acc1, qb, v_hi);
        }
        float sc[8] __attribute__((aligned(16)));
        vst1q_f32(sc, acc0);
        vst1q_f32(sc + 4, acc1);
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
        for (size_t w = 0; w < d; w++) {
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

static inline void scan_asym_b1_topk(const uint8_t *codes, size_t N, size_t d,
                                     const float *q, int K,
                                     float *out_s, int64_t *out_i) {
    scan_asym_b1_shard_topk(codes, N, d, 0, N, q, K, out_s, out_i);
}
#endif

#if defined(_OPENMP)
#include <omp.h>

/*
 * Threaded top-K asymmetric b=1 scan over [0, N). Each thread runs
 * scan_asym_b1_shard_topk on its range, then a critical-section
 * merge folds per-thread results into the global top-K. Per-thread
 * results from a partial-fill shard set unused entries to score=-INF
 * and idx=-1; the merge skips those.
 */
static inline void scan_asym_b1_topk_parallel(const uint8_t *codes, size_t N, size_t d,
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
        // Shard start must be byte-aligned: 1-bit codes are packed 8/byte and the
        // SIMD path reads vector ii from byte (ii>>3); an unaligned i0 reads the
        // wrong codes. Round chunk up to a multiple of 64 so every i0 is aligned.
        chunk = (chunk + 63) & ~(size_t)63;
        size_t i0 = (size_t)t * chunk;
        size_t i1 = i0 + chunk < N ? i0 + chunk : N;
        if (i0 < i1) {
            float ls[256]; int64_t li[256];
            // pre-fill so a partial shard is detectable
            for (int r = 0; r < K; r++) { ls[r] = -INFINITY; li[r] = -1; }
            scan_asym_b1_shard_topk(codes, N, d, i0, i1, q, K, ls, li);
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
