// test_topk.c — bounded max-heap of (key, index) pairs.
//
// We test both the "min top-K" usage (keep K smallest keys; root holds the
// largest of the K kept) and the "max top-K" usage (keep K largest keys;
// root holds the smallest of the K kept). The data structure is a single
// bounded max-heap; "max top-K" is implemented by negating keys at the
// caller, so only one heap implementation is tested here.

#include "simdq_topk.h"
#include <stdio.h>

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static void test_init_and_threshold(void) {
    simdq_topk_t h;
    int64_t keys[8]; int64_t idxs[8];
    simdq_topk_init(&h, 8, keys, idxs);
    CHECK(h.k == 8, "k=%d", h.k);
    CHECK(h.size == 0, "size=%d", h.size);
    CHECK(simdq_topk_threshold(&h) == INT64_MAX, "empty threshold");
}

static void test_fill_to_capacity(void) {
    simdq_topk_t h;
    int64_t keys[4]; int64_t idxs[4];
    simdq_topk_init(&h, 4, keys, idxs);

    int64_t input_keys[] = {50, 10, 30, 70};
    for (int i = 0; i < 4; i++) simdq_topk_offer(&h, input_keys[i], i);
    CHECK(h.size == 4, "size=%d", h.size);
    CHECK(simdq_topk_threshold(&h) == 70, "root key=%lld",
          (long long)simdq_topk_threshold(&h));
}

static void test_evict_when_full(void) {
    simdq_topk_t h;
    int64_t keys[4]; int64_t idxs[4];
    simdq_topk_init(&h, 4, keys, idxs);
    int64_t input_keys[] = {50, 10, 30, 70};
    for (int i = 0; i < 4; i++) simdq_topk_offer(&h, input_keys[i], i);

    // 5 is smaller than root (70): should evict 70 and keep 5
    simdq_topk_offer(&h, 5, 100);
    CHECK(h.size == 4, "size after evict=%d", h.size);
    CHECK(simdq_topk_threshold(&h) == 50, "new root=%lld",
          (long long)simdq_topk_threshold(&h));

    // 90 is bigger than root (50): should be ignored
    simdq_topk_offer(&h, 90, 200);
    CHECK(simdq_topk_threshold(&h) == 50, "root unchanged after rejected offer");
}

static void test_extract_sorted(void) {
    simdq_topk_t h;
    int64_t keys[4]; int64_t idxs[4];
    simdq_topk_init(&h, 4, keys, idxs);
    int64_t input_keys[] = {50, 10, 30, 70};
    for (int i = 0; i < 4; i++) simdq_topk_offer(&h, input_keys[i], i);

    int64_t out_keys[4]; int64_t out_idxs[4];
    int n = simdq_topk_extract_sorted(&h, out_keys, out_idxs);
    CHECK(n == 4, "extract returned n=%d", n);

    // ascending by key (smallest first)
    int64_t want_keys[] = {10, 30, 50, 70};
    int64_t want_idxs[] = {1, 2, 0, 3};
    for (int i = 0; i < 4; i++) {
        CHECK(out_keys[i] == want_keys[i] && out_idxs[i] == want_idxs[i],
              "rank %d got (%lld,%lld) want (%lld,%lld)", i,
              (long long)out_keys[i], (long long)out_idxs[i],
              (long long)want_keys[i], (long long)want_idxs[i]);
    }
}

static void test_merge_two_heaps(void) {
    // mimic the per-thread-merge pattern used by the parallel driver
    simdq_topk_t a, b;
    int64_t ka[4], ia[4], kb[4], ib[4];
    simdq_topk_init(&a, 4, ka, ia);
    simdq_topk_init(&b, 4, kb, ib);

    int64_t aks[] = {10, 20, 30, 40};
    int64_t bks[] = {15, 25, 35, 45};
    for (int i = 0; i < 4; i++) simdq_topk_offer(&a, aks[i], i);
    for (int i = 0; i < 4; i++) simdq_topk_offer(&b, bks[i], 100 + i);

    // merge: drain b into a
    int64_t bk_out[4], bi_out[4];
    int nb = simdq_topk_extract_sorted(&b, bk_out, bi_out);
    for (int i = 0; i < nb; i++) simdq_topk_offer(&a, bk_out[i], bi_out[i]);

    int64_t ok[4], oi[4];
    int n = simdq_topk_extract_sorted(&a, ok, oi);
    CHECK(n == 4, "merged size=%d", n);
    int64_t want_keys[] = {10, 15, 20, 25};
    int64_t want_idxs[] = {0, 100, 1, 101};
    for (int i = 0; i < 4; i++)
        CHECK(ok[i] == want_keys[i] && oi[i] == want_idxs[i],
              "merge rank %d got (%lld,%lld) want (%lld,%lld)", i,
              (long long)ok[i], (long long)oi[i],
              (long long)want_keys[i], (long long)want_idxs[i]);
}

static void test_k_equals_1(void) {
    simdq_topk_t h;
    int64_t keys[1]; int64_t idxs[1];
    simdq_topk_init(&h, 1, keys, idxs);

    simdq_topk_offer(&h, 50, 0);
    CHECK(simdq_topk_threshold(&h) == 50, "k=1 single offer threshold");

    simdq_topk_offer(&h, 30, 1);   // smaller, should evict
    CHECK(simdq_topk_threshold(&h) == 30, "k=1 evict to smaller");

    simdq_topk_offer(&h, 40, 2);   // larger than 30, rejected
    CHECK(simdq_topk_threshold(&h) == 30, "k=1 reject larger");

    int64_t ok[1], oi[1];
    int n = simdq_topk_extract_sorted(&h, ok, oi);
    CHECK(n == 1 && ok[0] == 30 && oi[0] == 1, "k=1 extract");
}

static void test_partial_fill_extract(void) {
    simdq_topk_t h;
    int64_t keys[5]; int64_t idxs[5];
    simdq_topk_init(&h, 5, keys, idxs);

    // only 3 of 5 slots filled
    simdq_topk_offer(&h, 30, 0);
    simdq_topk_offer(&h, 10, 1);
    simdq_topk_offer(&h, 20, 2);
    CHECK(simdq_topk_threshold(&h) == INT64_MAX, "partial threshold sentinel");

    int64_t ok[5], oi[5];
    int n = simdq_topk_extract_sorted(&h, ok, oi);
    CHECK(n == 3, "partial extract n=%d", n);
    int64_t want_k[3] = {10, 20, 30};
    int64_t want_i[3] = {1, 2, 0};
    for (int i = 0; i < 3; i++)
        CHECK(ok[i] == want_k[i] && oi[i] == want_i[i],
              "partial rank %d got (%lld,%lld) want (%lld,%lld)", i,
              (long long)ok[i], (long long)oi[i],
              (long long)want_k[i], (long long)want_i[i]);
}

static void test_ties_rejected(void) {
    // strict-less-than in offer: equal keys are NOT replaced when the heap is full
    simdq_topk_t h;
    int64_t keys[2]; int64_t idxs[2];
    simdq_topk_init(&h, 2, keys, idxs);

    simdq_topk_offer(&h, 50, 0);
    simdq_topk_offer(&h, 50, 1);   // both fit during fill phase
    CHECK(h.size == 2, "ties fit during fill, size=%d", h.size);
    CHECK(simdq_topk_threshold(&h) == 50, "tie threshold");

    // now full; another key=50 should be rejected (strict <)
    simdq_topk_offer(&h, 50, 2);
    CHECK(h.size == 2, "tied offer rejected at full, size=%d", h.size);

    // a strictly-smaller key DOES replace
    simdq_topk_offer(&h, 49, 3);
    CHECK(simdq_topk_threshold(&h) == 50, "post-replace root still 50");
    CHECK(h.size == 2, "size unchanged after evict");
}

int main(void) {
    test_init_and_threshold();
    test_fill_to_capacity();
    test_evict_when_full();
    test_extract_sorted();
    test_merge_two_heaps();
    test_k_equals_1();
    test_partial_fill_extract();
    test_ties_rejected();
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("all topk tests passed\n");
    return 0;
}
