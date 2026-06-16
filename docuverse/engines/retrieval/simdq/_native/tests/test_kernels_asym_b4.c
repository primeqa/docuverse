// test_kernels_asym_b4.c — correctness for asymmetric b=4 scan.

#include "simdq_kernels_asym_b4.h"
#include "simdq_pack.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef ASYM_KERNEL_NAME
#error "tests need at least AVX2"
#endif

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static void ref_topk_asym_b4(const uint8_t *codes, size_t N, size_t d,
                             const float *q, int K,
                             float *out_s, int64_t *out_i) {
    size_t row_bytes = (N + 1) / 2;
    for (int r = 0; r < K; r++) { out_s[r] = -INFINITY; out_i[r] = -1; }
    for (size_t i = 0; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
            int8_t v = simdq_unpack_b4(codes, row_bytes, w, i);
            s += q[w] * (float)v;
        }
        if (s > out_s[K - 1]) {
            int r = K - 1;
            while (r > 0 && out_s[r - 1] < s) {
                out_s[r] = out_s[r - 1]; out_i[r] = out_i[r - 1]; r--;
            }
            out_s[r] = s; out_i[r] = (int64_t)i;
        }
    }
}

static void test_random_b4(void) {
    size_t N = 1024, d = 768;
    float *Y = malloc(N * d * sizeof(float));
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, N, d);
    size_t row_bytes = (N + 1) / 2;
    uint8_t *codes = malloc(d * row_bytes);
    simdq_pack_b4(Y, N, d, scales, codes);

    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;

    int K = 10;
    float ks[10]; int64_t kis[10];
    scan_asym_b4_d768_topk(codes, N, q, K, ks, kis);
    float rs[10]; int64_t ris[10];
    ref_topk_asym_b4(codes, N, d, q, K, rs, ris);
    for (int r = 0; r < K; r++) {
        CHECK(fabsf(ks[r] - rs[r]) < 1e-3f,
              "r=%d got=%f ref=%f", r, ks[r], rs[r]);
        CHECK(kis[r] == ris[r] || fabsf(ks[r] - rs[r]) < 1e-4f,
              "r=%d idx got=%lld ref=%lld", r,
              (long long)kis[r], (long long)ris[r]);
    }
    free(Y); free(scales); free(codes);
}

static void test_planted_b4(void) {
    size_t N = 256, d = 768;
    float *Y = malloc(N * d * sizeof(float));
    for (size_t i = 0; i < N * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;
    for (size_t w = 0; w < 768; w++)
        Y[42 * d + w] = (q[w] >= 0) ? 100.0f : -100.0f;
    float *scales = simdq_pack_scales(Y, N, d);
    size_t row_bytes = (N + 1) / 2;
    uint8_t *codes = malloc(d * row_bytes);
    simdq_pack_b4(Y, N, d, scales, codes);
    float ks[3]; int64_t kis[3];
    scan_asym_b4_d768_topk(codes, N, q, 3, ks, kis);
    CHECK(kis[0] == 42, "planted top idx=%lld (want 42)", (long long)kis[0]);
    free(Y); free(scales); free(codes);
}

int main(void) {
    srand(9999);
    printf("asym b4 kernel path: %s\n", ASYM_KERNEL_NAME);
    test_random_b4();
    test_planted_b4();
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all asym b4 tests passed\n");
    return 0;
}
