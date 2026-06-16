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

static void test_b4_saturation(void) {
    // Directly exercise the clamp at the level-computation level.
    // Mirrors the formula in simdq_pack_b4 so we know the boundary
    // arithmetic and the clamp guards are both correct.
    int level_under = (int)floorf((-20.0f + 15.0f) * 0.5f + 0.5f); // = -2
    int level_over  = (int)floorf((+20.0f + 15.0f) * 0.5f + 0.5f); // = 18
    CHECK(level_under < 0, "expected pre-clamp underflow, got %d", level_under);
    CHECK(level_over > 15, "expected pre-clamp overflow, got %d", level_over);

    // Inject a vector designed to force saturation in pack: dim 0 carries
    // all the magnitude, so after scale = ||y||/sqrt(d) = |y[0]|/sqrt(d),
    // we have yn[0] = sqrt(d). Choose d=256 so sqrt(d)=16 > 15, triggering
    // the upper clamp. Negate to also hit the lower clamp.
    size_t N = 2, d = 256;
    float Y[2 * 256] = {0};
    Y[0 * d + 0] = +100.0f;   // yn[0] = +sqrt(d) = +16 -> clamps to level 15 -> +15
    Y[1 * d + 0] = -100.0f;   // yn[0] = -sqrt(d) = -16 -> clamps to level 0  -> -15
    float *scales = simdq_pack_scales(Y, N, d);
    size_t row_bytes = (N + 1) / 2;
    uint8_t codes[256 * 1];   // d * ceil(2/2) = 256
    simdq_pack_b4(Y, N, d, scales, codes);
    int8_t got_pos = simdq_unpack_b4(codes, row_bytes, 0, 0);
    int8_t got_neg = simdq_unpack_b4(codes, row_bytes, 0, 1);
    CHECK(got_pos == 15,  "saturation +: got %d (want 15)",  got_pos);
    CHECK(got_neg == -15, "saturation -: got %d (want -15)", got_neg);
    free(scales);
}

int main(void) {
    srand(12345);
    test_b1_roundtrip();
    test_b2_roundtrip();
    test_b4_roundtrip();
    test_b4_saturation();
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all pack/unpack tests passed\n");
    return 0;
}
