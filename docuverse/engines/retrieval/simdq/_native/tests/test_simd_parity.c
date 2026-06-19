// test_simd_parity.c — cross-SIMD parity for hamming + asym b in {1, 2, 4}.
//
// Compiled twice via CMake: once with -march=native (which on AVX-512
// hardware selects the AVX512F kernel paths) and once with
// -mavx2 -mfma -mpopcnt -mno-avx512f (which forces the AVX2 fallback).
// Each binary writes its top-K results for a deterministic fixed-seed
// input to a file named simd_parity_<arg>.out, where <arg> is argv[1]
// ("native" or "avx2"). CMake's add_test target then byte-compares the
// two output files.
//
// Output format (binary, little-endian, fixed layout for byte-equality):
//   section 1: hamming top-K, K records of (int64 dist, int64 idx)
//   section 2: asym b=1 top-K, K records of (int32 score_fp32_bits, int64 idx)
//   section 3: asym b=2 top-K, K records of (int32 score_fp32_bits, int64 idx)
//   section 4: asym b=4 top-K, K records of (int32 score_fp32_bits, int64 idx)
//
// Why include the score as the float bit pattern instead of the float
// itself? Byte-comparison demands a fixed serialization. fp32 stored as
// uint32 bit-pattern (memcpy float->uint32) is the natural fixed-width
// form. AVX-512 vs AVX-2 paths reduce in different orders, so the asym
// scores may legitimately differ in the low FP bits; if that happens,
// CMake's compare_files will fail and the engineer should switch
// simd_parity_dlopen to a tolerant-diff helper (see CMakeLists.txt note).
// For hamming (integer-only), bit-identical output is required.

#include "simdq_common.h"
#include "simdq_pack.h"
#include "simdq_kernels_hamming_topk.h"
#include "simdq_kernels_asym_b1.h"
#include "simdq_kernels_asym_b2.h"
#include "simdq_kernels_asym_b4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define N_CODES   10000  // small enough to run fast, large enough to scan many shards
#define D_DIMS    768    // standard simdq dim
#define K_TOPK    50
#define WORDS     (D_DIMS / 64)  // 12 uint64 words for hamming
#define SEED      0x515D9000     // fixed seed; the same on both binaries

// Fill a (length) fp32 array with a deterministic pseudo-uniform stream
// in [-0.5, +0.5]. We want the data identical across SIMD compile flavors,
// so use rand() seeded with srand(SEED) — same C runtime on the same host.
static void fill_fp32(float *p, size_t length) {
    for (size_t i = 0; i < length; i++)
        p[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
}

// Fill a uint64 SoA hamming code matrix sequentially via rnd64() (also
// deterministic given srand(SEED)).
static void fill_codes(uint64_t *p, size_t count) {
    for (size_t i = 0; i < count; i++) p[i] = rnd64();
}

static void run_hamming(FILE *out) {
    uint64_t *dbT = alloc_codes(N_CODES, WORDS);
    if (!dbT) { perror("alloc dbT"); exit(1); }
    fill_codes(dbT, (size_t)N_CODES * WORDS);

    uint64_t q[WORDS];
    fill_codes(q, WORDS);

    // Single-threaded shard scan: cross-SIMD parity is a per-kernel property.
    // The parallel variant adds OMP scheduling nondeterminism that breaks
    // tied-distance indices differently across runs even with the same SIMD
    // path; that's a cross-thread-count question, not a cross-SIMD one.
    int64_t out_d[K_TOPK], out_i[K_TOPK];
    scan_hamming_shard_topk(dbT, N_CODES, WORDS, 0, N_CODES, q, K_TOPK,
                            out_d, out_i);

    // Serialize as K records of (int64 dist, int64 idx) — both integer, so
    // bit-identical across SIMD paths is the contract.
    for (int r = 0; r < K_TOPK; r++) {
        if (fwrite(&out_d[r], sizeof(int64_t), 1, out) != 1) exit(2);
        if (fwrite(&out_i[r], sizeof(int64_t), 1, out) != 1) exit(2);
    }
    free(dbT);
}

// Pack Y to b-bit asym codes, run scan, serialize. b in {1, 2, 4}.
static void run_asym(FILE *out, int b, const float *Y, const float *scales,
                     const float *q) {
    size_t row_bytes_b1 = (N_CODES + 7) / 8;
    size_t row_bytes_b2 = (N_CODES + 3) / 4;
    size_t row_bytes_b4 = (N_CODES + 1) / 2;
    uint8_t *codes = NULL;
    float ks[K_TOPK];
    int64_t kis[K_TOPK];

    if (b == 1) {
        codes = (uint8_t *)malloc(D_DIMS * row_bytes_b1);
        if (!codes) { perror("alloc b1 codes"); exit(1); }
        simdq_pack_b1(Y, N_CODES, D_DIMS, scales, codes);
        scan_asym_b1_topk(codes, N_CODES, D_DIMS, q, K_TOPK, ks, kis);
    } else if (b == 2) {
        codes = (uint8_t *)malloc(D_DIMS * row_bytes_b2);
        if (!codes) { perror("alloc b2 codes"); exit(1); }
        simdq_pack_b2(Y, N_CODES, D_DIMS, scales, codes);
        scan_asym_b2_topk(codes, N_CODES, D_DIMS, q, K_TOPK, ks, kis);
    } else {
        codes = (uint8_t *)malloc(D_DIMS * row_bytes_b4);
        if (!codes) { perror("alloc b4 codes"); exit(1); }
        simdq_pack_b4(Y, N_CODES, D_DIMS, scales, codes);
        scan_asym_b4_topk(codes, N_CODES, D_DIMS, q, K_TOPK, ks, kis);
    }

    // Serialize as K records of (int32 score-bit-pattern, int64 idx).
    // The FP bit pattern is fixed-width and lossless; whether two scans on
    // different SIMD paths produce the same bit pattern is what the parity
    // diff actually checks.
    for (int r = 0; r < K_TOPK; r++) {
        uint32_t bits;
        memcpy(&bits, &ks[r], sizeof(uint32_t));
        if (fwrite(&bits, sizeof(uint32_t), 1, out) != 1) exit(2);
        if (fwrite(&kis[r], sizeof(int64_t), 1, out) != 1) exit(2);
    }
    free(codes);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <native|avx2>\n", argv[0]);
        return 2;
    }
    char path[128];
    snprintf(path, sizeof(path), "simd_parity_%s.out", argv[1]);
    FILE *out = fopen(path, "wb");
    if (!out) { perror(path); return 1; }

    // Single srand(SEED) so the data + query are identical across both
    // compiled binaries. The order matters: hamming first uses
    // count*WORDS + WORDS rnd64() calls, then asym uses N*D + D fp32 calls.
    srand(SEED);

    // ---- Hamming section (consumes 12 * (N+1) rnd64() calls) ----
    run_hamming(out);

    // ---- Asym sections share Y, scales, q across b=1, b=2, b=4 ----
    float *Y = (float *)malloc((size_t)N_CODES * D_DIMS * sizeof(float));
    if (!Y) { perror("alloc Y"); return 1; }
    fill_fp32(Y, (size_t)N_CODES * D_DIMS);
    float *scales = simdq_pack_scales(Y, N_CODES, D_DIMS);
    float q[D_DIMS];
    fill_fp32(q, D_DIMS);

    run_asym(out, 1, Y, scales, q);
    run_asym(out, 2, Y, scales, q);
    run_asym(out, 4, Y, scales, q);

    free(scales);
    free(Y);
    fclose(out);
    return 0;
}
