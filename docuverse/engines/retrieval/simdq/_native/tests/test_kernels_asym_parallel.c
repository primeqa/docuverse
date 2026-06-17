// test_kernels_asym_parallel.c — verify scan_asym_b{1,2,4}_d768_topk_parallel
// produces identical top-K (scores + indices) to the single-threaded
// _shard_topk over the full range [0, N), for both SIMD paths and across
// 1, 2, 8 OpenMP threads.

#include "simdq_kernels_asym_b1.h"
#include "simdq_kernels_asym_b2.h"
#include "simdq_kernels_asym_b4.h"
#include "simdq_pack.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

#define D 768
#define N 4096
#define K 32

static void check_match(const char *label,
                        const float *a_s, const int64_t *a_i,
                        const float *b_s, const int64_t *b_i, int K_) {
    for (int r = 0; r < K_; r++) {
        // Scores are encoded through the same fixed-point quantization in
        // both kernels, so they should be byte-equal modulo extraction order.
        CHECK(a_i[r] == b_i[r],
              "%s r=%d idx serial=%lld parallel=%lld",
              label, r, (long long)a_i[r], (long long)b_i[r]);
        CHECK(fabsf(a_s[r] - b_s[r]) < 1e-4f,
              "%s r=%d score serial=%f parallel=%f",
              label, r, a_s[r], b_s[r]);
    }
}

static void test_one_b(int b) {
    float *Y = (float *)malloc((size_t)N * D * sizeof(float));
    for (size_t i = 0; i < (size_t)N * D; i++)
        Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, N, D);

    float q[D];
    for (size_t w = 0; w < D; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;

    uint8_t *codes;
    if (b == 1) {
        size_t row_bytes = (N + 7) / 8;
        codes = (uint8_t *)malloc(D * row_bytes);
        simdq_pack_b1(Y, N, D, scales, codes);
    } else if (b == 2) {
        size_t row_bytes = (N + 3) / 4;
        codes = (uint8_t *)malloc(D * row_bytes);
        simdq_pack_b2(Y, N, D, scales, codes);
    } else {
        size_t row_bytes = (N + 1) / 2;
        codes = (uint8_t *)malloc(D * row_bytes);
        simdq_pack_b4(Y, N, D, scales, codes);
    }

    float ss[K]; int64_t si[K];
    float ps[K]; int64_t pi[K];

    if (b == 1) {
        scan_asym_b1_d768_shard_topk(codes, N, 0, N, q, K, ss, si);
    } else if (b == 2) {
        scan_asym_b2_d768_shard_topk(codes, N, 0, N, q, K, ss, si);
    } else {
        scan_asym_b4_d768_shard_topk(codes, N, 0, N, q, K, ss, si);
    }

    int thread_counts[] = {1, 2, 8};
    for (size_t ti = 0; ti < sizeof(thread_counts) / sizeof(int); ti++) {
        omp_set_num_threads(thread_counts[ti]);
        char label[64];
        snprintf(label, sizeof(label), "b=%d threads=%d", b, thread_counts[ti]);
        if (b == 1) {
            scan_asym_b1_d768_topk_parallel(codes, N, q, K, ps, pi);
        } else if (b == 2) {
            scan_asym_b2_d768_topk_parallel(codes, N, q, K, ps, pi);
        } else {
            scan_asym_b4_d768_topk_parallel(codes, N, q, K, ps, pi);
        }
        check_match(label, ss, si, ps, pi, K);
    }

    free(Y); free(scales); free(codes);
}

int main(void) {
    srand(424242);
    test_one_b(1);
    test_one_b(2);
    test_one_b(4);
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all asym parallel tests passed\n");
    return 0;
}
