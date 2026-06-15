// simdq_topk.h — bounded max-heap of (key, index) pairs for top-K scans.
//
// Caller-allocated storage: `keys` and `idxs` arrays of length k, owned by
// the caller. The struct itself is small (2 pointers + 2 ints = 24 bytes on
// LP64) and lives on the stack in scan kernels.
//
// The heap holds the K *smallest* keys seen so far; the root (index 0) is
// the *largest* of the K kept — i.e. the eviction candidate. simdq_topk_offer
// replaces the root iff the new key is strictly less.
//
// For top-K-largest workflows (asymmetric scans where higher score = better),
// negate keys at the caller and convert back at extraction.
//
// Hot-path contract: simdq_topk_threshold(h) returns the root key and is
// safe to call when the heap is full. While the heap is not yet full,
// the threshold is INT64_MAX. Inner scan loops should compare candidate
// keys against this threshold before going through simdq_topk_offer.
//
// Callers must ensure candidate keys are < INT64_MAX; keys equal to the
// fill-phase sentinel are silently rejected.

#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int64_t *keys;   // heap-ordered; keys[0] is the largest of the K kept
    int64_t *idxs;
    int k;           // capacity
    int size;        // 0..k
} simdq_topk_t;

static inline void simdq_topk_init(simdq_topk_t *h, int k,
                                   int64_t *keys, int64_t *idxs) {
    assert(k > 0);
    h->keys = keys;
    h->idxs = idxs;
    h->k = k;
    h->size = 0;
}

static inline int64_t simdq_topk_threshold(const simdq_topk_t *h) {
    return (h->size < h->k) ? INT64_MAX : h->keys[0];
}

static inline void simdq_topk__sift_down(simdq_topk_t *h, int i) {
    while (1) {
        int l = 2*i + 1, r = 2*i + 2, m = i;
        if (l < h->size && h->keys[l] > h->keys[m]) m = l;
        if (r < h->size && h->keys[r] > h->keys[m]) m = r;
        if (m == i) break;
        int64_t tk = h->keys[i]; h->keys[i] = h->keys[m]; h->keys[m] = tk;
        int64_t ti = h->idxs[i]; h->idxs[i] = h->idxs[m]; h->idxs[m] = ti;
        i = m;
    }
}

static inline void simdq_topk__sift_up(simdq_topk_t *h, int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->keys[p] >= h->keys[i]) break;
        int64_t tk = h->keys[i]; h->keys[i] = h->keys[p]; h->keys[p] = tk;
        int64_t ti = h->idxs[i]; h->idxs[i] = h->idxs[p]; h->idxs[p] = ti;
        i = p;
    }
}

static inline void simdq_topk_offer(simdq_topk_t *h, int64_t key, int64_t idx) {
    if (h->size < h->k) {
        h->keys[h->size] = key;
        h->idxs[h->size] = idx;
        h->size++;
        simdq_topk__sift_up(h, h->size - 1);
    } else if (key < h->keys[0]) {
        h->keys[0] = key;
        h->idxs[0] = idx;
        simdq_topk__sift_down(h, 0);
    }
}

// Extracts heap contents in ASCENDING key order. Destroys the heap.
// Returns the number of elements written (= size at entry).
static inline int simdq_topk_extract_sorted(simdq_topk_t *h,
                                            int64_t *out_keys,
                                            int64_t *out_idxs) {
    int n = h->size;
    // pop the max repeatedly; results emerge largest-first, so write in reverse
    while (h->size > 0) {
        int64_t k = h->keys[0], i = h->idxs[0];
        h->size--;
        if (h->size > 0) {
            h->keys[0] = h->keys[h->size];
            h->idxs[0] = h->idxs[h->size];
            simdq_topk__sift_down(h, 0);
        }
        out_keys[h->size] = k;
        out_idxs[h->size] = i;
    }
    h->keys = NULL; h->idxs = NULL; h->k = 0;
    return n;
}
