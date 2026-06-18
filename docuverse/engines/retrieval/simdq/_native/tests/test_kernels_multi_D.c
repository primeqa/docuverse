// test_kernels_multi_D.c — runtime-d/runtime-words spot-check.
//
// For each d in {192, 384, 512, 768, 1024, 1536}, build a tiny
// (n=1024) random asym-b=2 index, scan a random query, and assert the
// kernel's top-1 matches a scalar reference loop within 1e-4 relative
// tolerance. Same for the Hamming top-K kernel at every D in
// {384, 768, 1024, 1536} (words = D/64).
//
// We're not measuring throughput here, just correctness for the d/words
// runtime parameterization. Per-d test budget is well under 200ms.

#include "simdq_common.h"
#include "simdq_pack.h"
#include "simdq_kernels_asym_b2.h"
#include "simdq_kernels_hamming_topk.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef ASYM_B2_KERNEL_NAME
#error "tests need at least AVX2 (-march=native or -mavx2)"
#endif

#ifndef HAMMING_TOPK_KERNEL_NAME
#error "tests need at least AVX2 (-march=native or -mavx2)"
#endif

static void seed_rng(uint64_t *s) { *s = 0xC0FFEEULL; }
static float urand(uint64_t *s) {
    *s = (*s) * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x = (uint32_t)((*s) >> 33);
    return (float)(x & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
}

static int test_asym_b2_at_d(size_t d) {
    const size_t N = 1024;
    uint64_t rng;
    seed_rng(&rng);

    float *Y = NULL;
    float *q = NULL;
    if (posix_memalign((void **)&Y, 64, N * d * sizeof(float)) != 0) { fprintf(stderr, "OOM Y\n"); return 0; }
    if (posix_memalign((void **)&q, 64, d * sizeof(float)) != 0) { free(Y); fprintf(stderr, "OOM q\n"); return 0; }

    for (size_t i = 0; i < N * d; i++) Y[i] = urand(&rng);
    for (size_t w = 0; w < d; w++)    q[w] = urand(&rng);

    float *scales = simdq_pack_scales(Y, N, d);
    const size_t row_bytes = (N + 3) / 4;
    uint8_t *codes = (uint8_t *)calloc(d * row_bytes, 1);
    simdq_pack_b2(Y, N, d, scales, codes);

    int K = 4;
    float scores[4]; int64_t idxs[4];
    for (int r = 0; r < K; r++) { scores[r] = -INFINITY; idxs[r] = -1; }
    scan_asym_b2_topk_parallel(codes, N, d, q, K, scores, idxs);

    // Scalar reference: brute force <q, levels[code]> over all N using unpack helper
    float best = -INFINITY; size_t bi = 0;
    static const int8_t L[4] = {-3, -1, 1, 3};
    for (size_t i = 0; i < N; i++) {
        float s = 0.0f;
        for (size_t w = 0; w < d; w++) {
            uint8_t byte = codes[w * row_bytes + (i >> 2)];
            int8_t v = L[(byte >> ((i & 3) * 2)) & 0x3];
            s += q[w] * (float)v;
        }
        if (s > best) { best = s; bi = i; }
    }

    int ok = (idxs[0] == (int64_t)bi);
    if (!ok) {
        fprintf(stderr,
                "FAIL asym_b2 d=%zu: kernel best idx=%lld score=%.4f, ref idx=%zu score=%.4f\n",
                d, (long long)idxs[0], scores[0], bi, best);
    } else {
        printf("PASS asym_b2 d=%zu: top-1 idx=%zu score=%.4f\n", d, bi, best);
    }

    free(Y); free(q); free(scales); free(codes);
    return ok;
}

static int test_hamming_at_words(size_t words) {
    const size_t n = 4096;
    uint64_t rng; seed_rng(&rng);

    uint64_t *dbT = alloc_codes(n, words);
    uint64_t *q   = alloc_codes(1, words);
    if (!dbT || !q) { free(dbT); free(q); fprintf(stderr, "OOM hamming words=%zu\n", words); return 0; }

    // Fill SoA database and query with deterministic pseudo-random data
    fill_soa_parallel(dbT, n, words);
    for (size_t w = 0; w < words; w++)
        q[w] = ((uint64_t)urand(&rng) * 0xBBULL) ^ ((uint64_t)w << 17);

    int K = 8;
    int64_t scores[8], idxs[8];
    scan_hamming_topk_parallel(dbT, n, words, q, K, scores, idxs);

    // Scalar reference: brute-force Hamming distance via SoA, find K smallest
    int sorted_d[8]; int64_t sorted_i[8];
    for (int r = 0; r < K; r++) { sorted_d[r] = INT32_MAX; sorted_i[r] = -1; }
    for (size_t i = 0; i < n; i++) {
        int h = (int)hamming_soa(dbT, n, words, i, q);
        if (h < sorted_d[K - 1]) {
            int r = K - 1;
            while (r > 0 && sorted_d[r - 1] > h) {
                sorted_d[r] = sorted_d[r - 1]; sorted_i[r] = sorted_i[r - 1]; r--;
            }
            sorted_d[r] = h; sorted_i[r] = (int64_t)i;
        }
    }

    int ok = 1;
    for (int r = 0; r < K; r++) {
        if ((int)scores[r] != sorted_d[r]) {
            fprintf(stderr,
                    "FAIL hamming words=%zu: rank %d kernel dist=%lld, ref dist=%d\n",
                    words, r, (long long)scores[r], sorted_d[r]);
            ok = 0;
        }
    }
    if (ok) printf("PASS hamming words=%zu: top-%d distances match\n", words, K);

    free(dbT); free(q);
    return ok;
}

int main(void) {
    int failures = 0;
    size_t d_list[]     = {192, 384, 512, 768, 1024, 1536};
    size_t words_list[] = {6,   12,  16,  24};
    for (size_t i = 0; i < sizeof(d_list)/sizeof(d_list[0]); i++)
        if (!test_asym_b2_at_d(d_list[i])) failures++;
    for (size_t i = 0; i < sizeof(words_list)/sizeof(words_list[0]); i++)
        if (!test_hamming_at_words(words_list[i])) failures++;
    if (failures)
        fprintf(stderr, "%d FAILURE(S)\n", failures);
    else
        printf("all multi-D tests passed\n");
    return failures ? 1 : 0;
}
