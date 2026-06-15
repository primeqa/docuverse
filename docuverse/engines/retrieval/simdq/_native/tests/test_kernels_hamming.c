// test_kernels_hamming.c — correctness tests for all Hamming scan kernels.
//
// Every kernel is checked against an independent scalar reference that uses
// __builtin_popcountll and its own bookkeeping (nothing shared with the
// kernels except the data layout). Covered:
//   hamming_soa, scan_scalar_aos          (simdq_common.h, always built)
//   scan_soa512, scan_shard, min_lanes8   (AVX-512 VPOPCNTDQ path)
//   scan_shard, popcnt_bytes              (AVX2 nibble-LUT path)
//   scan_batch_parallel                   (OpenMP driver, either path)
//
// simdq_kernels_hamming.h selects the SIMD path from compiler flags, so this file
// is built twice to cover both paths on one machine (see CMakeLists.txt):
//   -march=native    -> AVX-512 kernels (on VPOPCNTDQ hardware)
//   -mavx2 -mpopcnt  -> AVX2 kernels
//
// Randomized cases assert that the returned index attains the true minimum
// distance (when several codes tie, any minimizer is a legal answer).
// Planted-best cases overwrite one code with the query XOR a few flipped
// bits, making it a unique minimum — placed at the ends, in the n % LANES
// tail, and across shard boundaries — and assert the exact index.

#include "simdq_kernels_hamming.h"
#include <string.h>

#ifndef KERNEL_NAME
#error "tests need at least AVX2 (build with -march=native or -mavx2 -mpopcnt)"
#endif

static int failures;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        failures++; \
        printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } \
} while (0)

// ---------- independent scalar reference ----------

static int ref_dist_aos(const uint64_t *db, size_t k, const uint64_t *q) {
    int h = 0;
    for (int w = 0; w < WORDS; w++)
        h += __builtin_popcountll(q[w] ^ db[k * WORDS + w]);
    return h;
}

static int ref_dist_soa(const uint64_t *dbT, size_t n, size_t k,
                        const uint64_t *q) {
    int h = 0;
    for (int w = 0; w < WORDS; w++)
        h += __builtin_popcountll(q[w] ^ dbT[(size_t)w * n + k]);
    return h;
}

// first-wins top-1 over codes [i0, i1) of an SoA database
static size_t ref_scan_soa(const uint64_t *dbT, size_t n, size_t i0, size_t i1,
                           const uint64_t *q, int *out_d) {
    int best = INT32_MAX; size_t bi = i0;
    for (size_t k = i0; k < i1; k++) {
        int h = ref_dist_soa(dbT, n, k, q);
        if (h < best) { best = h; bi = k; }
    }
    *out_d = best;
    return bi;
}

static size_t ref_scan_aos(const uint64_t *db, size_t n, const uint64_t *q,
                           int *out_d) {
    int best = INT32_MAX; size_t bi = 0;
    for (size_t k = 0; k < n; k++) {
        int h = ref_dist_aos(db, k, q);
        if (h < best) { best = h; bi = k; }
    }
    *out_d = best;
    return bi;
}

// ---------- planted-best helpers ----------

// q with `flips` distinct bits flipped: distance from q is exactly `flips`.
// Random 768-bit codes sit at distance ~384 from q (min ~330 even for n in
// the millions), so a planted code with flips <= 3 is a unique minimum.
static void make_planted(uint64_t code[WORDS], size_t k, const uint64_t *q,
                         int flips) {
    memcpy(code, q, WORDS * 8);
    for (int f = 0; f < flips; f++) {
        unsigned p = (unsigned)((k * 131 + (unsigned)f * 257) % (WORDS * 64));
        code[p / 64] ^= 1ull << (p % 64);
    }
}

static void plant_soa(uint64_t *dbT, size_t n, size_t k, const uint64_t *q,
                      int flips) {
    uint64_t code[WORDS];
    make_planted(code, k, q, flips);
    for (int w = 0; w < WORDS; w++) dbT[(size_t)w * n + k] = code[w];
}

static void plant_aos(uint64_t *db, size_t k, const uint64_t *q, int flips) {
    make_planted(db + k * WORDS, k, q, flips);
}

// sizes chosen to exercise empty main loops (n < LANES), exact blocks,
// n % LANES tails, and n < NQ
static const size_t SIZES[] = {1, 2, 7, 8, 9, 16, 17, 100, 1000, 1031};
#define NSIZES (sizeof SIZES / sizeof *SIZES)

// ---------- tests ----------

static void test_hamming_soa(void) {
    size_t n = 129;
    uint64_t *dbT = alloc_codes(n);
    uint64_t q[WORDS];
    fill_rnd64(dbT, n * WORDS);
    fill_rnd64(q, WORDS);
    for (size_t k = 0; k < n; k++)
        CHECK(hamming_soa(dbT, n, k, q) == ref_dist_soa(dbT, n, k, q),
              "k=%zu", k);
    free(dbT);
}

static void test_scan_scalar_aos(void) {
    for (size_t s = 0; s < NSIZES; s++) {
        size_t n = SIZES[s];
        uint64_t *db = alloc_codes(n);
        uint64_t q[WORDS];
        fill_rnd64(q, WORDS);

        // random db: the scalar kernel is strictly first-wins, like the
        // reference, so indices must match exactly
        fill_rnd64(db, n * WORDS);
        int rd;
        size_t ri = ref_scan_aos(db, n, q, &rd);
        CHECK(scan_scalar_aos(db, n, q) == ri, "n=%zu", n);

        // planted unique best at the front, middle, and back
        size_t plants[3] = {0, n / 2, n - 1};
        for (int p = 0; p < 3; p++) {
            fill_rnd64(db, n * WORDS);
            plant_aos(db, plants[p], q, p);
            CHECK(scan_scalar_aos(db, n, q) == plants[p],
                  "n=%zu pos=%zu", n, plants[p]);
        }
        free(db);
    }
}

#ifdef __AVX512VPOPCNTDQ__
static void test_min_lanes8(void) {
    __m512i vi = _mm512_setr_epi64(100, 101, 102, 103, 104, 105, 106, 107);
    __m512i vd = _mm512_setr_epi64(9, 3, 9, 3, 9, 9, 9, 2);
    int64_t d, i;
    min_lanes8(vd, vi, &d, &i);
    CHECK(d == 2 && i == 107, "got d=%lld i=%lld", (long long)d, (long long)i);
    // all lanes tied: the lowest lane wins
    min_lanes8(_mm512_set1_epi64(4), vi, &d, &i);
    CHECK(d == 4 && i == 100, "got d=%lld i=%lld", (long long)d, (long long)i);
}

static void test_scan_soa512(void) {
    for (size_t s = 0; s < NSIZES; s++) {
        size_t n = SIZES[s];
        uint64_t *dbT = alloc_codes(n);
        uint64_t q[WORDS];
        fill_rnd64(q, WORDS);

        // random db: returned index must attain the true minimum distance
        fill_rnd64(dbT, n * WORDS);
        int rd;
        ref_scan_soa(dbT, n, 0, n, q, &rd);
        size_t idx = scan_soa512(dbT, n, q);
        CHECK(idx < n && ref_dist_soa(dbT, n, idx, q) == rd,
              "n=%zu idx=%zu", n, idx);

        // planted unique best; pos = n-1 lands in the scalar tail when
        // n % 8 != 0 (sizes 9, 17, 1031, ...)
        size_t plants[3] = {0, n / 2, n - 1};
        for (int p = 0; p < 3; p++) {
            fill_rnd64(dbT, n * WORDS);
            plant_soa(dbT, n, plants[p], q, p);
            CHECK(scan_soa512(dbT, n, q) == plants[p],
                  "n=%zu pos=%zu", n, plants[p]);
        }
        free(dbT);
    }
}

#elif defined(__AVX2__)
static void test_popcnt_bytes(void) {
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    uint8_t in[32], out[32];
    for (int trial = 0; trial < 4; trial++) {
        for (int b = 0; b < 32; b++) in[b] = (uint8_t)rand();
        in[0] = 0x00; in[1] = 0xFF;   // pin the extremes
        __m256i r = popcnt_bytes(_mm256_loadu_si256((const __m256i *)in),
                                 lut, m0f);
        _mm256_storeu_si256((__m256i *)out, r);
        for (int b = 0; b < 32; b++)
            CHECK(out[b] == __builtin_popcount(in[b]),
                  "byte=0x%02x got %d", in[b], out[b]);
    }
}
#endif

static void test_scan_shard_full(void) {
    for (size_t s = 0; s < NSIZES; s++) {
        size_t n = SIZES[s];
        uint64_t *dbT = alloc_codes(n);
        uint64_t qs[NQ][WORDS];
        fill_rnd64(dbT, n * WORDS);
        fill_rnd64(&qs[0][0], NQ * WORDS);

        // plant a unique best per query when there is room for NQ distinct
        // positions; otherwise just verify distances against the reference
        int planted = n >= NQ;
        if (planted)
            for (int j = 0; j < NQ; j++)
                plant_soa(dbT, n, (size_t)j * n / NQ, qs[j], j % 4);

        int64_t bd[NQ], bi[NQ];
        scan_shard(dbT, n, 0, n, qs, bd, bi);
        for (int j = 0; j < NQ; j++) {
            int rd;
            ref_scan_soa(dbT, n, 0, n, qs[j], &rd);
            CHECK(bd[j] == rd, "n=%zu j=%d d=%lld ref=%d",
                  n, j, (long long)bd[j], rd);
            CHECK((size_t)bi[j] < n &&
                  ref_dist_soa(dbT, n, (size_t)bi[j], qs[j]) == rd,
                  "n=%zu j=%d idx=%lld", n, j, (long long)bi[j]);
            if (planted)
                CHECK(bi[j] == (int64_t)((size_t)j * n / NQ) && bd[j] == j % 4,
                      "n=%zu j=%d planted idx=%lld d=%lld",
                      n, j, (long long)bi[j], (long long)bd[j]);
        }

        // unique best at n-1 for one query: lands in the scalar tail loop
        // whenever n % LANES != 0
        fill_rnd64(dbT, n * WORDS);
        plant_soa(dbT, n, n - 1, qs[2], 1);
        scan_shard(dbT, n, 0, n, qs, bd, bi);
        CHECK(bi[2] == (int64_t)(n - 1) && bd[2] == 1,
              "n=%zu tail plant idx=%lld d=%lld",
              n, (long long)bi[2], (long long)bd[2]);
        free(dbT);
    }
}

static void test_scan_shard_ranges(void) {
    size_t n = 1000;
    uint64_t *dbT = alloc_codes(n);
    uint64_t qs[NQ][WORDS];
    fill_rnd64(dbT, n * WORDS);
    fill_rnd64(&qs[0][0], NQ * WORDS);
    for (int j = 0; j < NQ; j++)
        plant_soa(dbT, n, (size_t)j * n / NQ, qs[j], j % 4);

    // shard at awkward boundaries (not multiples of LANES), check each
    // shard against the reference restricted to its range, and check that
    // merging the shards reproduces the global answer
    size_t cuts[] = {0, 1, 7, 64, 333, 999, 1000};
    size_t ncuts = sizeof cuts / sizeof *cuts;
    int64_t bd[NQ], bi[NQ];
    for (int j = 0; j < NQ; j++) { bd[j] = INT64_MAX; bi[j] = 0; }
    for (size_t c = 0; c + 1 < ncuts; c++) {
        int64_t ld[NQ], li[NQ];
        scan_shard(dbT, n, cuts[c], cuts[c + 1], qs, ld, li);
        for (int j = 0; j < NQ; j++) {
            int rd;
            ref_scan_soa(dbT, n, cuts[c], cuts[c + 1], qs[j], &rd);
            CHECK(ld[j] == rd && (size_t)li[j] >= cuts[c] &&
                  (size_t)li[j] < cuts[c + 1] &&
                  ref_dist_soa(dbT, n, (size_t)li[j], qs[j]) == rd,
                  "range [%zu,%zu) j=%d d=%lld ref=%d idx=%lld",
                  cuts[c], cuts[c + 1], j, (long long)ld[j], rd,
                  (long long)li[j]);
            if (ld[j] < bd[j]) { bd[j] = ld[j]; bi[j] = li[j]; }
        }
    }
    for (int j = 0; j < NQ; j++)
        CHECK(bd[j] == j % 4 && bi[j] == (int64_t)((size_t)j * n / NQ),
              "merged j=%d d=%lld idx=%lld",
              j, (long long)bd[j], (long long)bi[j]);

    // empty range: bests must stay at the sentinel
    int64_t ed[NQ], ei[NQ];
    scan_shard(dbT, n, 500, 500, qs, ed, ei);
    for (int j = 0; j < NQ; j++)
        CHECK(ed[j] == INT64_MAX, "empty range j=%d d=%lld",
              j, (long long)ed[j]);
    free(dbT);
}

#ifdef _OPENMP
static void test_scan_batch_parallel(void) {
    size_t n = 10000;
    uint64_t *dbT = alloc_codes(n);
    uint64_t qs[NQ][WORDS];
    fill_rnd64(dbT, n * WORDS);
    fill_rnd64(&qs[0][0], NQ * WORDS);
    for (int j = 0; j < NQ; j++)
        plant_soa(dbT, n, (size_t)j * n / NQ, qs[j], j % 4);

    int maxt = omp_get_max_threads();
    int tcs[] = {1, 2, 3, maxt};
    for (size_t t = 0; t < sizeof tcs / sizeof *tcs; t++) {
        omp_set_num_threads(tcs[t]);
        int64_t gd[NQ], gi[NQ];
        scan_batch_parallel(dbT, n, qs, gd, gi);
        for (int j = 0; j < NQ; j++)
            CHECK(gd[j] == j % 4 && gi[j] == (int64_t)((size_t)j * n / NQ),
                  "threads=%d j=%d d=%lld idx=%lld",
                  tcs[t], j, (long long)gd[j], (long long)gi[j]);
    }
    free(dbT);

    // n far below the thread count: most threads get an empty shard
    size_t n2 = 5;
    uint64_t *small = alloc_codes(n2);
    fill_rnd64(small, n2 * WORDS);
    int64_t gd[NQ], gi[NQ];
    scan_batch_parallel(small, n2, qs, gd, gi);
    for (int j = 0; j < NQ; j++) {
        int rd;
        ref_scan_soa(small, n2, 0, n2, qs[j], &rd);
        CHECK(gd[j] == rd &&
              ref_dist_soa(small, n2, (size_t)gi[j], qs[j]) == rd,
              "small n j=%d d=%lld ref=%d", j, (long long)gd[j], rd);
    }
    free(small);
    omp_set_num_threads(maxt);
}
#endif

static void test_ties_and_uniform(void) {
    size_t n = 100;
    uint64_t *dbT = alloc_codes(n);
    uint64_t qs[NQ][WORDS];
    fill_rnd64(&qs[0][0], NQ * WORDS);
    const uint64_t *q = qs[0];

    // two exact copies of q (distance 0 at indices 3 and 77): SIMD kernels
    // may return either minimizer
    memset(dbT, 0, n * WORDS * 8);
    plant_soa(dbT, n, 3, q, 0);
    plant_soa(dbT, n, 77, q, 0);
#ifdef __AVX512VPOPCNTDQ__
    size_t ti = scan_soa512(dbT, n, q);
    CHECK(ti == 3 || ti == 77, "tie idx=%zu", ti);
#endif
    uint64_t tq[NQ][WORDS];
    for (int j = 0; j < NQ; j++) memcpy(tq[j], q, WORDS * 8);
    int64_t bd[NQ], bi[NQ];
    scan_shard(dbT, n, 0, n, tq, bd, bi);
    for (int j = 0; j < NQ; j++)
        CHECK(bd[j] == 0 && (bi[j] == 3 || bi[j] == 77),
              "tie j=%d d=%lld idx=%lld", j, (long long)bd[j],
              (long long)bi[j]);

    // the scalar kernel is strictly first-wins even on ties
    uint64_t *db = alloc_codes(n);
    memset(db, 0, n * WORDS * 8);
    plant_aos(db, 3, q, 0);
    plant_aos(db, 77, q, 0);
    CHECK(scan_scalar_aos(db, n, q) == 3, "scalar tie not first-wins");
    free(db);

    // uniform all-zero db: every code is at distance popcount(q_j)
    memset(dbT, 0, n * WORDS * 8);
    scan_shard(dbT, n, 0, n, qs, bd, bi);
    for (int j = 0; j < NQ; j++) {
        int T = 0;
        for (int w = 0; w < WORDS; w++) T += __builtin_popcountll(qs[j][w]);
        CHECK(bd[j] == T && (size_t)bi[j] < n,
              "uniform j=%d d=%lld expect %d", j, (long long)bd[j], T);
    }
    free(dbT);
}

int main(void) {
    srand(12345);
    printf("kernel path: %s, NQ=%d, LANES=%d\n", KERNEL_NAME, NQ, LANES);
    test_hamming_soa();
    test_scan_scalar_aos();
#ifdef __AVX512VPOPCNTDQ__
    test_min_lanes8();
    test_scan_soa512();
#elif defined(__AVX2__)
    test_popcnt_bytes();
#endif
    test_scan_shard_full();
    test_scan_shard_ranges();
#ifdef _OPENMP
    test_scan_batch_parallel();
#endif
    test_ties_and_uniform();
    if (failures) {
        printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
