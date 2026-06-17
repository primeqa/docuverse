// bench_asym_b1.c — asymmetric b=1 (float query x 1-bit DB) top-K scan
// throughput.
//
// CLI: bench_asym_b1 [n] [reps] [K]

#include "simdq_kernels_asym_b1.h"
#include "simdq_pack.h"
#include "simdq_common.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    size_t n;
    int reps;
    parse_args(argc, argv, &n, &reps);
    int K = (argc > 3) ? atoi(argv[3]) : 100;
    if (K < 1 || K > 256) { fprintf(stderr, "K must be in [1,256]\n"); return 2; }
    srand(42);
    const size_t d = 768;

    // generate Y on the heap; for very large N this is the dominant alloc
    float *Y = malloc(n * d * sizeof(float));
    if (!Y) { fprintf(stderr, "alloc Y failed\n"); return 1; }
    for (size_t i = 0; i < n * d; i++) Y[i] = ((float)rand() / RAND_MAX) - 0.5f;
    float *scales = simdq_pack_scales(Y, n, d);
    size_t row_bytes = (n + 7) / 8;
    uint8_t *codes = aligned_alloc(64, d * row_bytes);
    simdq_pack_b1(Y, n, d, scales, codes);
    free(Y);   // we no longer need the floats

    float q[768];
    for (size_t w = 0; w < 768; w++) q[w] = ((float)rand() / RAND_MAX) - 0.5f;

    float out_s[256]; int64_t out_i[256];
    // warmup
    scan_asym_b1_d768_topk(codes, n, q, K, out_s, out_i);

    double tmin = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        scan_asym_b1_d768_topk(codes, n, q, K, out_s, out_i);
        double dt = now() - t0;
        if (dt < tmin) tmin = dt;
    }
    double cmps = (double)n / tmin;
    double bytes = (double)n * d / 8;
    double gbs = bytes / tmin / 1e9;
    printf("kernel=%s n=%zu K=%d reps=%d  best=%.3fms  cmp/s=%.2fM  GB/s=%.1f  top1=%lld score=%f\n",
           ASYM_B1_KERNEL_NAME, n, K, reps, tmin * 1e3,
           cmps / 1e6, gbs, (long long)out_i[0], out_s[0]);
    free(codes); free(scales);
    return 0;
}
