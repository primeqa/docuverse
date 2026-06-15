// hamming_bench_v5.c — threaded, query-batched 768-bit Hamming scan with
// two scan_shard kernels selected at compile time (see hamming_kernels.h):
//   - AVX-512 VPOPCNTDQ path (Zen 4/5, Ice Lake+)   : as in v4 — 8 codes/iter,
//     per word one 512-bit load XORed with the broadcast query word and
//     popcounted with _mm512_popcnt_epi64; best distance/index tracked per
//     lane via compare-mask + masked moves.
//   - AVX2 path (Zen 1-3, Haswell+)                  : 4 codes/iter, vpshufb
//     nibble-LUT popcount; byte counts accumulated across all 12 words (max
//     12*8 = 96 per byte, < 255, no overflow), one vpsadbw reduction per
//     block; scalar per-lane best tracking.
// Both paths reuse each database load for all NQ queries, and both run under
// the scan_batch_parallel OpenMP shard/merge driver. KERNEL_NAME in the
// output line reports which path was compiled in.
//
// Build:  gcc -O3 -march=native -fopenmp hamming_bench_v5.c -o hb5
//         (-DNQ=32 etc. to change batch width; default 8)
// Run:    OMP_NUM_THREADS=t ./hb5 <num_codes> <reps>

#include "simdq_kernels_hamming.h"

#ifndef KERNEL_NAME
#error "Need at least AVX2"
#endif
#include <omp.h>

int main(int argc, char **argv) {
    size_t n; int reps;
    parse_args(argc, argv, &n, &reps);
    int nthreads = omp_get_max_threads();

    uint64_t *dbT = alloc_codes(n);
    if (!dbT) return 1;
    fill_soa_parallel(dbT, n);
    uint64_t qs[NQ][WORDS];
    srand(7);
    for (int j = 0; j < NQ; j++)
        fill_rnd64(qs[j], WORDS);

    int64_t gd[NQ], gi[NQ];
    double tbest = 1e18;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        scan_batch_parallel(dbT, n, qs, gd, gi);
        double dt = now() - t0;
        if (dt < tbest) tbest = dt;
    }

    printf("%-17s threads=%2d NQ=%2d : n=%zu (%.0f MB)  %9.1f M cmp/s  %6.1f GB/s memory  (%.1f ms/pass, top-1[0]=%lld d=%lld)\n",
           KERNEL_NAME, nthreads, NQ, n, n * WORDS * 8.0 / 1e6,
           (double)n * NQ / tbest / 1e6,
           (double)n * WORDS * 8 / tbest / 1e9,
           tbest * 1e3, (long long)gi[0], (long long)gd[0]);
    free(dbT);
    return 0;
}
