// simdq_kernels_hamming.h — SIMD scan kernels over an SoA database (dbT[w*n + i]),
// selected at compile time:
//   AVX-512 VPOPCNTDQ (Zen 4/5, Ice Lake+) : vertical kernel, 8 codes/iter;
//     per word, one 512-bit load is XORed with the broadcast query word and
//     popcounted with _mm512_popcnt_epi64. Best distance/index tracked per
//     lane via compare-mask + masked moves, reduced to scalar at the end.
//   AVX2 (Zen 1-3, Haswell+) : vpshufb nibble-LUT popcount, 4 codes/iter;
//     byte counts accumulate across all `words` words (at most words*8 ones
//     per byte; < 255 for words <= 23), then one vpsadbw reduction per block.
//
// Batched kernels service NQ queries per database load: each chunk is loaded
// once and reused for all NQ queries, cutting memory traffic by NQ x.
// KERNEL_NAME names the selected path; it is undefined if neither is built.
#pragma once



#include "simdq_common.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__AVX512VPOPCNTDQ__)
#define KERNEL_NAME "AVX512-VPOPCNTDQ"
#define LANES 8   // codes per iteration (512-bit = 8 x u64)

/*
 * Reduces 8 (distance, index) lane pairs to the scalar minimum: vd/vi hold
 * the per-lane best distances and the indices they came from; the overall
 * winners are written to *best_d / *best_i. Strict less-than keeps the
 * lowest lane on ties — note this is not necessarily the lowest index,
 * so when several codes tie the kernels may return any minimizer.
 */
static inline void min_lanes8(__m512i vd, __m512i vi,
                              int64_t *best_d, int64_t *best_i) {
    int64_t d8[8], i8[8];
    _mm512_storeu_si512(d8, vd);
    _mm512_storeu_si512(i8, vi);
    *best_d = INT64_MAX; *best_i = 0;
    for (int l = 0; l < 8; l++)
        if (d8[l] < *best_d) { *best_d = d8[l]; *best_i = i8[l]; }
}

/*
 * Single-query vertical AVX-512 scan; processes 8 codes per iteration.
 * For each of the `words` words, one 512-bit load fetches word w of 8
 * consecutive codes, XORs it with the broadcast query word, popcounts
 * with _mm512_popcnt_epi64, and adds into a per-lane distance accumulator
 * — no horizontal reduction and no branch in the hot loop. The running
 * best distance/index per lane is updated via compare-mask + masked
 * moves, reduced to scalar after the loop; the n % 8 tail runs scalar.
 * Returns the index of the closest code (top-1).
 */
static inline size_t scan_soa512(const uint64_t *dbT, size_t n, size_t words,
                                 const uint64_t *q) {
    __m512i bestd = _mm512_set1_epi64(INT64_MAX);
    __m512i besti = _mm512_setzero_si512();
    __m512i idx = _mm512_setr_epi64(0,1,2,3,4,5,6,7);
    const __m512i step = _mm512_set1_epi64(LANES);
    for (size_t i = 0; i + LANES <= n; i += LANES) {
        __m512i acc = _mm512_setzero_si512();
        for (size_t w = 0; w < words; w++) {
            __m512i d = _mm512_loadu_si512(dbT + w * n + i);
            acc = _mm512_add_epi64(acc,
                  _mm512_popcnt_epi64(_mm512_xor_si512(d, _mm512_set1_epi64(q[w]))));
        }
        __mmask8 lt = _mm512_cmplt_epi64_mask(acc, bestd);
        bestd = _mm512_mask_mov_epi64(bestd, lt, acc);
        besti = _mm512_mask_mov_epi64(besti, lt, idx);
        idx = _mm512_add_epi64(idx, step);
    }
    int64_t best, bi;
    min_lanes8(bestd, besti, &best, &bi);
    for (size_t k = n & ~(size_t)(LANES - 1); k < n; k++) {   // tail
        int h = hamming_soa(dbT, n, words, k, q);
        if (h < best) { best = h; bi = (int64_t)k; }
    }
    return (size_t)bi;
}

/*
 * Batched AVX-512 scan of codes [i0, i1) for NQ queries at once. Same
 * vertical loop as scan_soa512, but each 512-bit database load is reused
 * for all NQ queries (1 load -> NQ XOR+popcounts), cutting memory traffic
 * by NQ x while the arithmetic work stays the same. The word stride is
 * always n (the full database), so a shard is just an index range.
 * Writes the best distance and index per query to best_d / best_i.
 */
/* qs is a flat array of NQ query codes, each `words` uint64_t.
 * Access: query j word w -> qs[j * words + w]. */
static inline void scan_shard(const uint64_t *dbT, size_t n, size_t words,
                              size_t i0, size_t i1,
                              const uint64_t *qs,
                              int64_t best_d[NQ], int64_t best_i[NQ]) {
    __m512i bd[NQ], bi[NQ];
    for (int j = 0; j < NQ; j++) {
        bd[j] = _mm512_set1_epi64(INT64_MAX);
        bi[j] = _mm512_setzero_si512();
    }
    __m512i idx = _mm512_add_epi64(_mm512_setr_epi64(0,1,2,3,4,5,6,7),
                                   _mm512_set1_epi64((int64_t)i0));
    const __m512i step = _mm512_set1_epi64(LANES);
    size_t i = i0;
    for (; i + LANES <= i1; i += LANES) {
        __m512i acc[NQ];
        for (int j = 0; j < NQ; j++) acc[j] = _mm512_setzero_si512();
        for (size_t w = 0; w < words; w++) {
            __m512i d = _mm512_loadu_si512(dbT + w * n + i);  // one load,
            for (int j = 0; j < NQ; j++)                      // NQ uses
                acc[j] = _mm512_add_epi64(acc[j],
                         _mm512_popcnt_epi64(_mm512_xor_si512(d,
                             _mm512_set1_epi64(qs[j * words + w]))));
        }
        for (int j = 0; j < NQ; j++) {
            __mmask8 lt = _mm512_cmplt_epi64_mask(acc[j], bd[j]);
            bd[j] = _mm512_mask_mov_epi64(bd[j], lt, acc[j]);
            bi[j] = _mm512_mask_mov_epi64(bi[j], lt, idx);
        }
        idx = _mm512_add_epi64(idx, step);
    }
    for (int j = 0; j < NQ; j++) {
        min_lanes8(bd[j], bi[j], &best_d[j], &best_i[j]);
        for (size_t k = i; k < i1; k++) {                     // tail
            int h = hamming_soa(dbT, n, words, k, qs + j * words);
            if (h < best_d[j]) { best_d[j] = h; best_i[j] = (int64_t)k; }
        }
    }
}

#elif defined(__AVX2__)
#define KERNEL_NAME "AVX2-nibbleLUT"
#define LANES 4   // codes per iteration (256-bit = 4 x u64)

/*
 * Per-byte popcount of a 256-bit vector via two vpshufb lookups: the low
 * and high nibble of each byte index a 16-entry bit-count table and the
 * two results are summed. lut and m0f (the 0x0F mask) are passed in so
 * the caller can hoist them out of its loop.
 */
static inline __m256i popcnt_bytes(__m256i x, __m256i lut, __m256i m0f) {
    __m256i lo = _mm256_and_si256(x, m0f);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), m0f);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                           _mm256_shuffle_epi8(lut, hi));
}

/*
 * Batched AVX2 scan of codes [i0, i1) for NQ queries; 4 codes (256 bits)
 * per iteration. Distances accumulate as per-byte counts across all `words`
 * words (at most words * 8 ones per byte; <= 24*8=192 < 255, so the u8
 * accumulators cannot overflow for words <= 23), then one vpsadbw per block
 * reduces the byte counts to 4 u64 distances. Best tracking per lane is
 * scalar — there are no cheap masked moves before AVX-512. Like the
 * AVX-512 variant, each database load is reused for all NQ queries.
 * Writes the best distance and index per query to best_d / best_i.
 */
/* qs is a flat array of NQ query codes, each `words` uint64_t.
 * Access: query j word w -> qs[j * words + w]. */
static inline void scan_shard(const uint64_t *dbT, size_t n, size_t words,
                              size_t i0, size_t i1,
                              const uint64_t *qs,
                              int64_t best_d[NQ], int64_t best_i[NQ]) {
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();

    for (int j = 0; j < NQ; j++) { best_d[j] = INT64_MAX; best_i[j] = 0; }
    int64_t bd[NQ][LANES], bi[NQ][LANES];
    for (int j = 0; j < NQ; j++)
        for (int l = 0; l < LANES; l++) { bd[j][l] = INT64_MAX; bi[j][l] = 0; }

    size_t i = i0;
    for (; i + LANES <= i1; i += LANES) {
        // byte-count accumulators: words * max 8 ones per byte < 255 for words <= 23
        __m256i accb[NQ];
        for (int j = 0; j < NQ; j++) accb[j] = zero;
        for (size_t w = 0; w < words; w++) {
            __m256i d = _mm256_loadu_si256(
                (const __m256i*)(dbT + w * n + i));
            for (int j = 0; j < NQ; j++) {
                __m256i x = _mm256_xor_si256(d, _mm256_set1_epi64x(qs[j * words + w]));
                accb[j] = _mm256_add_epi8(accb[j], popcnt_bytes(x, lut, m0f));
            }
        }
        for (int j = 0; j < NQ; j++) {
            __m256i sums = _mm256_sad_epu8(accb[j], zero); // 4 x u64 distances
            int64_t d4[4];
            _mm256_storeu_si256((__m256i*)d4, sums);
            for (int l = 0; l < LANES; l++)
                if (d4[l] < bd[j][l]) { bd[j][l] = d4[l]; bi[j][l] = (int64_t)(i + l); }
        }
    }
    for (int j = 0; j < NQ; j++) {
        for (int l = 0; l < LANES; l++)
            if (bd[j][l] < best_d[j]) { best_d[j] = bd[j][l]; best_i[j] = bi[j][l]; }
        for (size_t k = i; k < i1; k++) {                     // tail
            int h = hamming_soa(dbT, n, words, k, qs + j * words);
            if (h < best_d[j]) { best_d[j] = h; best_i[j] = (int64_t)k; }
        }
    }
}

#elif defined(__aarch64__) || defined(__ARM_NEON)
#define KERNEL_NAME "NEON-vcnt"
#define LANES 2   // codes per iteration (128-bit = 2 x u64)

/*
 * Per-lane popcount of a uint64x2. NEON has native single-cycle vcntq_u8
 * for byte popcount; the pairwise-widening-add chain
 * (vpaddlq_u8 -> u16, vpaddlq_u16 -> u32, vpaddlq_u32 -> u64) reduces the
 * 16 byte counts to a per-u64-lane sum.
 */
static inline uint64x2_t popcnt_u64x2(uint64x2_t x) {
    uint8x16_t cnts = vcntq_u8(vreinterpretq_u8_u64(x));
    return vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(cnts)));
}

/*
 * Reduces 2 (distance, index) lane pairs to the scalar minimum. Same
 * semantics as min_lanes8 above but for the 2-lane NEON accumulator:
 * strict less-than keeps the lower lane on ties (so lane 0 wins when
 * distances tie).
 */
static inline void min_lanes2(uint64x2_t vd, uint64x2_t vi,
                              int64_t *best_d, int64_t *best_i) {
    uint64_t d2[2], i2[2];
    vst1q_u64(d2, vd);
    vst1q_u64(i2, vi);
    *best_d = INT64_MAX; *best_i = 0;
    if ((int64_t)d2[0] < *best_d) { *best_d = (int64_t)d2[0]; *best_i = (int64_t)i2[0]; }
    if ((int64_t)d2[1] < *best_d) { *best_d = (int64_t)d2[1]; *best_i = (int64_t)i2[1]; }
}

static inline size_t scan_soa512(const uint64_t *dbT, size_t n, size_t words,
                                 const uint64_t *q) {
    uint64x2_t bestd = vdupq_n_u64((uint64_t)INT64_MAX);
    uint64x2_t besti = vdupq_n_u64(0);
    const uint64_t idx_init[2] = {0, 1};
    uint64x2_t idx = vld1q_u64(idx_init);
    const uint64x2_t step = vdupq_n_u64(LANES);
    for (size_t i = 0; i + LANES <= n; i += LANES) {
        uint64x2_t acc = vdupq_n_u64(0);
        for (size_t w = 0; w < words; w++) {
            uint64x2_t d = vld1q_u64(dbT + w * n + i);
            uint64x2_t x = veorq_u64(d, vdupq_n_u64(q[w]));
            acc = vaddq_u64(acc, popcnt_u64x2(x));
        }
        uint64x2_t lt = vcltq_u64(acc, bestd);
        bestd = vbslq_u64(lt, acc, bestd);
        besti = vbslq_u64(lt, idx, besti);
        idx = vaddq_u64(idx, step);
    }
    int64_t best, bi;
    min_lanes2(bestd, besti, &best, &bi);
    for (size_t k = n & ~(size_t)(LANES - 1); k < n; k++) {   // tail
        int h = hamming_soa(dbT, n, words, k, q);
        if (h < best) { best = h; bi = (int64_t)k; }
    }
    return (size_t)bi;
}

/*
 * Batched NEON scan of codes [i0, i1) for NQ queries at once. Each 128-bit
 * database load is reused for all NQ queries (1 load -> NQ XOR+popcounts),
 * matching the AVX-512/AVX2 batched contract. Writes best distance and
 * index per query to best_d / best_i.
 */
static inline void scan_shard(const uint64_t *dbT, size_t n, size_t words,
                              size_t i0, size_t i1,
                              const uint64_t *qs,
                              int64_t best_d[NQ], int64_t best_i[NQ]) {
    uint64x2_t bd[NQ], bi[NQ];
    for (int j = 0; j < NQ; j++) {
        bd[j] = vdupq_n_u64((uint64_t)INT64_MAX);
        bi[j] = vdupq_n_u64(0);
    }
    const uint64_t idx_init[2] = {(uint64_t)i0, (uint64_t)i0 + 1};
    uint64x2_t idx = vld1q_u64(idx_init);
    const uint64x2_t step = vdupq_n_u64(LANES);
    size_t i = i0;
    for (; i + LANES <= i1; i += LANES) {
        uint64x2_t acc[NQ];
        for (int j = 0; j < NQ; j++) acc[j] = vdupq_n_u64(0);
        for (size_t w = 0; w < words; w++) {
            uint64x2_t d = vld1q_u64(dbT + w * n + i);          // one load,
            for (int j = 0; j < NQ; j++)                        // NQ uses
                acc[j] = vaddq_u64(acc[j],
                         popcnt_u64x2(veorq_u64(d, vdupq_n_u64(qs[j * words + w]))));
        }
        for (int j = 0; j < NQ; j++) {
            uint64x2_t lt = vcltq_u64(acc[j], bd[j]);
            bd[j] = vbslq_u64(lt, acc[j], bd[j]);
            bi[j] = vbslq_u64(lt, idx, bi[j]);
        }
        idx = vaddq_u64(idx, step);
    }
    for (int j = 0; j < NQ; j++) {
        min_lanes2(bd[j], bi[j], &best_d[j], &best_i[j]);
        for (size_t k = i; k < i1; k++) {                     // tail
            int h = hamming_soa(dbT, n, words, k, qs + j * words);
            if (h < best_d[j]) { best_d[j] = h; best_i[j] = (int64_t)k; }
        }
    }
}
#endif

#if defined(KERNEL_NAME) && defined(_OPENMP)
#include <omp.h>

/*
 * Threaded batched scan: the database is partitioned into contiguous
 * shards, one per OpenMP thread; each thread runs scan_shard on its range
 * independently, and an omp critical section merges the per-thread bests
 * into gd/gi (best distance and index per query). Resets gd/gi itself, so
 * each call is a complete top-1 search over [0, n).
 */
/* qs is a flat array of NQ query codes, each `words` uint64_t. */
static inline void scan_batch_parallel(const uint64_t *dbT, size_t n, size_t words,
                                       const uint64_t *qs,
                                       int64_t gd[NQ], int64_t gi[NQ]) {
    for (int j = 0; j < NQ; j++) { gd[j] = INT64_MAX; gi[j] = 0; }
    #pragma omp parallel
    {
        int t = omp_get_thread_num(), T = omp_get_num_threads();
        size_t chunk = (n + T - 1) / T;
        size_t i0 = (size_t)t * chunk;
        size_t i1 = i0 + chunk < n ? i0 + chunk : n;
        if (i0 < i1) {
            int64_t ld[NQ], li[NQ];
            scan_shard(dbT, n, words, i0, i1, qs, ld, li);
            #pragma omp critical
            for (int j = 0; j < NQ; j++)
                if (ld[j] < gd[j]) { gd[j] = ld[j]; gi[j] = li[j]; }
        }
    }
}
#endif
