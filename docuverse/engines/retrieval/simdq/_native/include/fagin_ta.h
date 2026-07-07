// fagin_ta.h — Fagin's Threshold Algorithm (TA) over per-dimension sorted
// lists (Fagin, Lotem, Naor, PODS'01). Exact top-K inner-product retrieval:
// lock-step sorted access over the active per-dimension lists, one full
// dot-product random access per first-seen document, halting when the kth
// best score reaches the threshold T = sum_j q_j * t_j.
//
// Negative query weights: lists are sorted descending by raw value; for
// q_j < 0 the scan walks the same list bottom-up (ascending raw value ==
// descending contribution q_j * x) — the "double scan". Dimensions with
// q_j == 0 are skipped entirely (they contribute 0 to every score and to
// the threshold, so skipping stays exact).
//
// The top-K heap is simdq_topk.h (int64 keys, keeps K smallest). Float
// scores are mapped through an order-preserving float32 -> uint32 -> int64
// transform, negated so "K smallest keys" == "K largest scores".

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "simdq_topk.h"

typedef struct {
    int64_t depth;            // rows of sorted access per active dim
    int64_t sorted_accesses;  // total sorted accesses (active dims x depth)
    int64_t random_accesses;  // candidates scored (full dot products)
    int64_t rounds;           // TA rounds executed
    int     exhausted;        // 1 if every row was scanned (depth == N)
} fagin_stats_t;

// Order-preserving float32 -> int64 key, negated for simdq_topk (which
// keeps the K *smallest* keys; we want the K *largest* scores). The
// standard monotone IEEE-754 transform: flip all bits of negatives, flip
// only the sign bit of non-negatives.
static inline int64_t fagin_score_key(float s) {
    uint32_t u;
    memcpy(&u, &s, sizeof u);
    u ^= (u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return -(int64_t)u;
}

static inline float fagin_key_score(int64_t k) {
    uint32_t u = (uint32_t)(-k);
    u ^= (u & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    float s;
    memcpy(&s, &u, sizeof u);
    return s;
}

// Generic-D dot product; -O3 -march=native vectorizes this to AVX2/AVX-512
// on x86 (NEON on arm64). No fixed-D whitelist.
static inline float fagin_dot(const float *restrict a, const float *restrict b,
                              int64_t D) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int64_t j = 0; j < D; j++) s += a[j] * b[j];
    return s;
}

// Runs TA for one query. Writes min(K, N) results (best first) into
// out_scores/out_idxs; remaining slots get score 0 / idx -1. Returns the
// number of valid results. `batch` = rows of sorted access per active dim
// per round; `epsilon` = additive halting slack (0 = exact); `max_depth`
// caps sorted-access depth (0 = unlimited).
static inline int fagin_ta_search(
        const float *Y,          // [N, D] row-major fp32
        const int32_t *order,    // [D, N] per-dim argsort of Y[:,j], descending
        const float *vals,       // [D, N] Y[order[j], j] (sorted values)
        int64_t N, int64_t D,
        const float *q,          // [D]
        int K, int64_t batch, float epsilon, int64_t max_depth,
        float *out_scores, int64_t *out_idxs, fagin_stats_t *stats) {

    memset(stats, 0, sizeof *stats);
    for (int i = 0; i < K; i++) { out_scores[i] = 0.0f; out_idxs[i] = -1; }

    int heap_cap = (int)((int64_t)K < N ? (int64_t)K : N);

    // Active dimensions (q_j != 0).
    int32_t *dims = (int32_t *)malloc((size_t)D * sizeof *dims);
    int64_t ndims = 0;
    for (int64_t j = 0; j < D; j++)
        if (q[j] != 0.0f) dims[ndims++] = (int32_t)j;

    if (ndims == 0) {
        // Every document scores 0; any min(K, N) docs are a correct top-K.
        for (int i = 0; i < heap_cap; i++) out_idxs[i] = i;
        stats->exhausted = 1;
        free(dims);
        return heap_cap;
    }

    uint64_t *seen = (uint64_t *)calloc(((size_t)N + 63) / 64, sizeof *seen);
    int64_t *cand = (int64_t *)malloc((size_t)ndims * (size_t)batch * sizeof *cand);
    float *cand_scores = (float *)malloc((size_t)ndims * (size_t)batch
                                         * sizeof *cand_scores);
    int64_t *hkeys = (int64_t *)malloc((size_t)heap_cap * sizeof *hkeys);
    int64_t *hidxs = (int64_t *)malloc((size_t)heap_cap * sizeof *hidxs);

    simdq_topk_t heap;
    simdq_topk_init(&heap, heap_cap, hkeys, hidxs);

    int64_t depth = 0;
    while (depth < N) {
        int64_t take = batch;
        if (take > N - depth) take = N - depth;
        if (max_depth > 0 && take > max_depth - depth) take = max_depth - depth;

        // Phase A (serial, cheap): sorted access — gather unseen candidates.
        int64_t n_cand = 0;
        for (int64_t a = 0; a < ndims; a++) {
            int64_t j = dims[a];
            const int32_t *lst = order + (size_t)j * (size_t)N;
            for (int64_t r = 0; r < take; r++) {
                int64_t pos = (q[j] > 0.0f) ? depth + r : N - 1 - depth - r;
                int64_t id = lst[pos];
                uint64_t bit = 1ull << (id & 63);
                if (!(seen[id >> 6] & bit)) {
                    seen[id >> 6] |= bit;
                    cand[n_cand++] = id;
                }
            }
        }
        stats->sorted_accesses += ndims * take;

        // Phase B (parallel): random access — one full dot per new candidate.
        #pragma omp parallel for schedule(static)
        for (int64_t c = 0; c < n_cand; c++)
            cand_scores[c] = fagin_dot(q, Y + (size_t)cand[c] * (size_t)D, D);
        stats->random_accesses += n_cand;

        // Phase C (serial): heap maintenance (K is small).
        for (int64_t c = 0; c < n_cand; c++) {
            int64_t key = fagin_score_key(cand_scores[c]);
            if (key < simdq_topk_threshold(&heap))
                simdq_topk_offer(&heap, key, cand[c]);
        }

        depth += take;
        stats->rounds += 1;

        // Threshold T = sum_j q_j * t_j at the current cursors.
        float T = 0.0f;
        for (int64_t a = 0; a < ndims; a++) {
            int64_t j = dims[a];
            int64_t pos = (q[j] > 0.0f) ? depth - 1 : N - depth;
            T += q[j] * vals[(size_t)j * (size_t)N + pos];
        }

        if (heap.size >= heap_cap) {
            float kth = fagin_key_score(simdq_topk_threshold(&heap));
            if (kth >= T - epsilon) break;   // slide-19 halting rule (+ eps slack)
        }
        if (max_depth > 0 && depth >= max_depth) break;
    }
    if (depth >= N) stats->exhausted = 1;
    stats->depth = depth;

    int64_t *okeys = (int64_t *)malloc((size_t)heap_cap * sizeof *okeys);
    int64_t *oidxs = (int64_t *)malloc((size_t)heap_cap * sizeof *oidxs);
    int n = simdq_topk_extract_sorted(&heap, okeys, oidxs);
    // Ascending keys == descending scores, so results come out best-first.
    for (int i = 0; i < n; i++) {
        out_scores[i] = fagin_key_score(okeys[i]);
        out_idxs[i] = oidxs[i];
    }
    free(okeys); free(oidxs);
    free(hkeys); free(hidxs);
    free(cand); free(cand_scores);
    free(seen); free(dims);
    return n;
}
