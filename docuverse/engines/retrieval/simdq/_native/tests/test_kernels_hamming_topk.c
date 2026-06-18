// test_kernels_hamming_topk.c — correctness for top-K Hamming scans on
// both SIMD paths. Same shape as test_kernels_hamming.c (planted-best
// cases at boundaries, randomized + scalar reference) but exercising
// scan_hamming_shard_topk and scan_hamming_topk_parallel over K > 1.

#include "simdq_kernels_hamming_topk.h"
#include <stdio.h>
#include <string.h>

#ifndef HAMMING_TOPK_KERNEL_NAME
#error "tests need at least AVX2"
#endif

#define WORDS 12  // D=768 bits (local constant; simdq_common.h is now WORDS-free)

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

// independent scalar top-K reference, ascending order
static void ref_topk(const uint64_t *dbT, size_t n, const uint64_t *q,
                     int K, int64_t *out_d, int64_t *out_i) {
    // brute O(n*K) — fine for n <= a few thousand in tests
    for (int r = 0; r < K; r++) { out_d[r] = INT64_MAX; out_i[r] = -1; }
    for (size_t k = 0; k < n; k++) {
        int h = 0;
        for (int w = 0; w < WORDS; w++)
            h += __builtin_popcountll(q[w] ^ dbT[(size_t)w * n + k]);
        // insertion into sorted top-K
        if (h < out_d[K - 1]) {
            int r = K - 1;
            while (r > 0 && out_d[r - 1] > h) {
                out_d[r] = out_d[r - 1]; out_i[r] = out_i[r - 1]; r--;
            }
            out_d[r] = h; out_i[r] = (int64_t)k;
        }
    }
}

// plant `count` distinct unique-best codes for q at given indices, with
// strictly increasing distances 0, 1, 2, ... (count <= 6 for safety).
static void plant_ascending(uint64_t *dbT, size_t n, const uint64_t *q,
                            const size_t *positions, int count) {
    for (int p = 0; p < count; p++) {
        uint64_t code[WORDS];
        memcpy(code, q, WORDS * 8);
        for (int f = 0; f < p; f++) {
            // bits unique across plants: shift by p*64 + f
            unsigned bit = (unsigned)((p * 64 + f) % (WORDS * 64));
            code[bit / 64] ^= 1ull << (bit % 64);
        }
        for (int w = 0; w < WORDS; w++)
            dbT[(size_t)w * n + positions[p]] = code[w];
    }
}

static void test_topk_planted(void) {
    size_t n = 1000;
    int K = 5;
    uint64_t *dbT = alloc_codes(n, WORDS);
    uint64_t q[WORDS];
    fill_rnd64(q, WORDS);
    fill_rnd64(dbT, n * WORDS);
    size_t pos[5] = {0, 250, 500, 750, n - 1};
    plant_ascending(dbT, n, q, pos, 5);

    int64_t out_d[5], out_i[5];
    scan_hamming_shard_topk(dbT, n, WORDS, 0, n, q, K, out_d, out_i);

    int64_t ref_d[5], ref_i[5];
    ref_topk(dbT, n, q, K, ref_d, ref_i);

    for (int r = 0; r < K; r++)
        CHECK(out_d[r] == ref_d[r] && out_i[r] == ref_i[r],
              "K=%d r=%d got (%lld,%lld) ref (%lld,%lld)", K, r,
              (long long)out_d[r], (long long)out_i[r],
              (long long)ref_d[r], (long long)ref_i[r]);
    free(dbT);
}

static void test_topk_random(void) {
    size_t n = 2000;
    int K = 10;
    uint64_t *dbT = alloc_codes(n, WORDS);
    uint64_t q[WORDS];
    fill_rnd64(q, WORDS);
    fill_rnd64(dbT, n * WORDS);

    int64_t out_d[10], out_i[10];
    scan_hamming_shard_topk(dbT, n, WORDS, 0, n, q, K, out_d, out_i);

    int64_t ref_d[10], ref_i[10];
    ref_topk(dbT, n, q, K, ref_d, ref_i);

    // distances must match exactly; indices may differ on ties
    for (int r = 0; r < K; r++) {
        CHECK(out_d[r] == ref_d[r], "K=%d r=%d d=%lld ref=%lld",
              K, r, (long long)out_d[r], (long long)ref_d[r]);
        // verify index attains the claimed distance
        int h = 0;
        size_t k = (size_t)out_i[r];
        for (int w = 0; w < WORDS; w++)
            h += __builtin_popcountll(q[w] ^ dbT[(size_t)w * n + k]);
        CHECK(h == out_d[r], "K=%d r=%d idx=%lld dist mismatch", K, r,
              (long long)out_i[r]);
    }
    free(dbT);
}

static void test_topk_K_eq_1(void) {
    // K=1 must agree with the existing top-1 kernel
    size_t n = 500;
    uint64_t *dbT = alloc_codes(n, WORDS);
    uint64_t q[WORDS];
    fill_rnd64(q, WORDS);
    fill_rnd64(dbT, n * WORDS);

    int64_t out_d, out_i;
    scan_hamming_shard_topk(dbT, n, WORDS, 0, n, q, 1, &out_d, &out_i);

    int rd; size_t ri = ref_scan_soa_top1(dbT, n, WORDS, 0, n, q, &rd);
    CHECK(out_d == rd, "d=%lld ref=%d", (long long)out_d, rd);
    int h = 0;
    for (int w = 0; w < WORDS; w++)
        h += __builtin_popcountll(q[w] ^ dbT[(size_t)w * n + (size_t)out_i]);
    CHECK(h == rd, "K=1 idx=%lld dist mismatch", (long long)out_i);
    free(dbT);
}

static void test_topk_threaded(void) {
    size_t n = 10000;
    int K = 8;
    uint64_t *dbT = alloc_codes(n, WORDS);
    uint64_t q[WORDS];
    fill_rnd64(q, WORDS);
    fill_rnd64(dbT, n * WORDS);

    // plant 8 strictly-better codes spread across shards
    size_t pos[8];
    for (int p = 0; p < 8; p++) pos[p] = (size_t)p * n / 8 + 17;
    plant_ascending(dbT, n, q, pos, 8);

    int64_t gd[8], gi[8];
    scan_hamming_topk_parallel(dbT, n, WORDS, q, K, gd, gi);
    int64_t rd[8], ri[8];
    ref_topk(dbT, n, q, K, rd, ri);

    for (int r = 0; r < K; r++)
        CHECK(gd[r] == rd[r] && gi[r] == ri[r],
              "threaded K=%d r=%d got (%lld,%lld) ref (%lld,%lld)", K, r,
              (long long)gd[r], (long long)gi[r],
              (long long)rd[r], (long long)ri[r]);
    free(dbT);
}

int main(void) {
    srand(98765);
    printf("kernel path: %s\n", HAMMING_TOPK_KERNEL_NAME);
    test_topk_K_eq_1();
    test_topk_planted();
    test_topk_random();
#ifdef _OPENMP
    test_topk_threaded();
#endif
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all topk kernel tests passed\n");
    return 0;
}
