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
#include <stdlib.h>

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
