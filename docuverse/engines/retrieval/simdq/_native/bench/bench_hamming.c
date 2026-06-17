// bench_hamming.c — top-K Hamming SoA scan throughput.
//
// CLI: bench_hamming [n] [reps] [K]
//   n     number of database codes (default 1,000,000)
//   reps  timed repetitions; reports the minimum (default 5)
//   K     top-K depth (default 100)
// Threaded; honors OMP_NUM_THREADS.

#include "simdq_kernels_hamming_topk.h"
#include <stdio.h>

int main(int argc, char **argv) {
    size_t n;
    int reps;
    parse_args(argc, argv, &n, &reps);
    int K = (argc > 3) ? atoi(argv[3]) : 100;
    if (K < 1 || K > 256) { fprintf(stderr, "K must be in [1,256]\n"); return 2; }
    srand(42);

    uint64_t *dbT = alloc_codes(n);
    uint64_t q[WORDS];
    if (!dbT) { fprintf(stderr, "alloc failed\n"); return 1; }
    fill_soa_parallel(dbT, n);
    fill_rnd64(q, WORDS);

    int64_t out_d[256], out_i[256];
    // warmup
    scan_batch_parallel_topk(dbT, n, q, K, out_d, out_i);

    double tmin = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        scan_batch_parallel_topk(dbT, n, q, K, out_d, out_i);
        double dt = now() - t0;
        if (dt < tmin) tmin = dt;
    }
    double cmps = (double)n / tmin;
    double bytes = (double)n * WORDS * 8;
    double gbs = bytes / tmin / 1e9;
    printf("kernel=%s n=%zu K=%d reps=%d  best=%.3fms  cmp/s=%.2fM  GB/s=%.1f  top1=%lld\n",
           HAMMING_TOPK_KERNEL_NAME, n, K, reps, tmin * 1e3, cmps / 1e6, gbs,
           (long long)out_i[0]);
    free(dbT);
    return 0;
}
