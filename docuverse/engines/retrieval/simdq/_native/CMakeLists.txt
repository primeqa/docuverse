cmake_minimum_required(VERSION 3.16)
project(simdq_native C)

set(CMAKE_C_STANDARD 11)
find_package(OpenMP REQUIRED)

set(COMMON_FLAGS -O3 -march=native)

# ---- benchmark binaries ----

add_executable(bench_hamming_top1 bench/bench_hamming_top1.c)
target_compile_options(bench_hamming_top1 PRIVATE ${COMMON_FLAGS})
target_include_directories(bench_hamming_top1 PRIVATE include)
target_link_libraries(bench_hamming_top1 PRIVATE OpenMP::OpenMP_C)

add_executable(bench_hamming bench/bench_hamming.c)
target_compile_options(bench_hamming PRIVATE ${COMMON_FLAGS})
target_include_directories(bench_hamming PRIVATE include)
target_link_libraries(bench_hamming PRIVATE OpenMP::OpenMP_C)

add_executable(bench_asym_b1 bench/bench_asym_b1.c)
target_compile_options(bench_asym_b1 PRIVATE ${COMMON_FLAGS})
target_include_directories(bench_asym_b1 PRIVATE include)
target_link_libraries(bench_asym_b1 PRIVATE OpenMP::OpenMP_C m)

add_executable(bench_asym_b2 bench/bench_asym_b2.c)
target_compile_options(bench_asym_b2 PRIVATE ${COMMON_FLAGS})
target_include_directories(bench_asym_b2 PRIVATE include)
target_link_libraries(bench_asym_b2 PRIVATE OpenMP::OpenMP_C m)

add_executable(bench_asym_b4 bench/bench_asym_b4.c)
target_compile_options(bench_asym_b4 PRIVATE ${COMMON_FLAGS})
target_include_directories(bench_asym_b4 PRIVATE include)
target_link_libraries(bench_asym_b4 PRIVATE OpenMP::OpenMP_C m)

# ---- tests (ctest) ----
enable_testing()

# native path: AVX-512 VPOPCNTDQ kernels on capable hardware
add_executable(test_kernels_hamming tests/test_kernels_hamming.c)
target_compile_options(test_kernels_hamming PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_hamming PRIVATE include)
target_link_libraries(test_kernels_hamming PRIVATE OpenMP::OpenMP_C)
add_test(NAME kernels_hamming_native COMMAND test_kernels_hamming)

# AVX2 nibble-LUT path, forced even on AVX-512 hardware
add_executable(test_kernels_hamming_avx2 tests/test_kernels_hamming.c)
target_compile_options(test_kernels_hamming_avx2 PRIVATE -O3 -mavx2 -mpopcnt)
target_include_directories(test_kernels_hamming_avx2 PRIVATE include)
target_link_libraries(test_kernels_hamming_avx2 PRIVATE OpenMP::OpenMP_C)
add_test(NAME kernels_hamming_avx2 COMMAND test_kernels_hamming_avx2)

add_executable(test_topk tests/test_topk.c)
target_compile_options(test_topk PRIVATE ${COMMON_FLAGS})
target_include_directories(test_topk PRIVATE include)
add_test(NAME topk COMMAND test_topk)

add_executable(test_kernels_hamming_topk tests/test_kernels_hamming_topk.c)
target_compile_options(test_kernels_hamming_topk PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_hamming_topk PRIVATE include)
target_link_libraries(test_kernels_hamming_topk PRIVATE OpenMP::OpenMP_C)
add_test(NAME kernels_hamming_topk_native COMMAND test_kernels_hamming_topk)

add_executable(test_kernels_hamming_topk_avx2 tests/test_kernels_hamming_topk.c)
target_compile_options(test_kernels_hamming_topk_avx2 PRIVATE -O3 -mavx2 -mpopcnt)
target_include_directories(test_kernels_hamming_topk_avx2 PRIVATE include)
target_link_libraries(test_kernels_hamming_topk_avx2 PRIVATE OpenMP::OpenMP_C)
add_test(NAME kernels_hamming_topk_avx2 COMMAND test_kernels_hamming_topk_avx2)

add_executable(test_pack tests/test_pack.c)
target_compile_options(test_pack PRIVATE ${COMMON_FLAGS})
target_include_directories(test_pack PRIVATE include)
target_link_libraries(test_pack PRIVATE m)
add_test(NAME pack COMMAND test_pack)

add_executable(test_kernels_asym_b1 tests/test_kernels_asym_b1.c)
target_compile_options(test_kernels_asym_b1 PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_asym_b1 PRIVATE include)
target_link_libraries(test_kernels_asym_b1 PRIVATE m)
add_test(NAME kernels_asym_b1_native COMMAND test_kernels_asym_b1)

add_executable(test_kernels_asym_b1_avx2 tests/test_kernels_asym_b1.c)
target_compile_options(test_kernels_asym_b1_avx2 PRIVATE -O3 -mavx2 -mfma)
target_include_directories(test_kernels_asym_b1_avx2 PRIVATE include)
target_link_libraries(test_kernels_asym_b1_avx2 PRIVATE m)
add_test(NAME kernels_asym_b1_avx2 COMMAND test_kernels_asym_b1_avx2)

add_executable(test_kernels_asym_b2 tests/test_kernels_asym_b2.c)
target_compile_options(test_kernels_asym_b2 PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_asym_b2 PRIVATE include)
target_link_libraries(test_kernels_asym_b2 PRIVATE m)
add_test(NAME kernels_asym_b2_native COMMAND test_kernels_asym_b2)

add_executable(test_kernels_asym_b2_avx2 tests/test_kernels_asym_b2.c)
target_compile_options(test_kernels_asym_b2_avx2 PRIVATE -O3 -mavx2 -mfma)
target_include_directories(test_kernels_asym_b2_avx2 PRIVATE include)
target_link_libraries(test_kernels_asym_b2_avx2 PRIVATE m)
add_test(NAME kernels_asym_b2_avx2 COMMAND test_kernels_asym_b2_avx2)

add_executable(test_kernels_asym_b4 tests/test_kernels_asym_b4.c)
target_compile_options(test_kernels_asym_b4 PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_asym_b4 PRIVATE include)
target_link_libraries(test_kernels_asym_b4 PRIVATE m)
add_test(NAME kernels_asym_b4_native COMMAND test_kernels_asym_b4)

add_executable(test_kernels_asym_b4_avx2 tests/test_kernels_asym_b4.c)
target_compile_options(test_kernels_asym_b4_avx2 PRIVATE -O3 -mavx2 -mfma)
target_include_directories(test_kernels_asym_b4_avx2 PRIVATE include)
target_link_libraries(test_kernels_asym_b4_avx2 PRIVATE m)
add_test(NAME kernels_asym_b4_avx2 COMMAND test_kernels_asym_b4_avx2)

add_executable(test_kernels_asym_parallel tests/test_kernels_asym_parallel.c)
target_compile_options(test_kernels_asym_parallel PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_asym_parallel PRIVATE include)
target_link_libraries(test_kernels_asym_parallel PRIVATE OpenMP::OpenMP_C m)
add_test(NAME kernels_asym_parallel_native COMMAND test_kernels_asym_parallel)

add_executable(test_kernels_asym_parallel_avx2 tests/test_kernels_asym_parallel.c)
target_compile_options(test_kernels_asym_parallel_avx2 PRIVATE -O3 -mavx2 -mfma)
target_include_directories(test_kernels_asym_parallel_avx2 PRIVATE include)
target_link_libraries(test_kernels_asym_parallel_avx2 PRIVATE OpenMP::OpenMP_C m)
add_test(NAME kernels_asym_parallel_avx2 COMMAND test_kernels_asym_parallel_avx2)

add_executable(test_kernels_multi_D tests/test_kernels_multi_D.c)
target_compile_options(test_kernels_multi_D PRIVATE ${COMMON_FLAGS})
target_include_directories(test_kernels_multi_D PRIVATE include)
target_link_libraries(test_kernels_multi_D PRIVATE OpenMP::OpenMP_C m)
add_test(NAME kernels_multi_D_native COMMAND test_kernels_multi_D)

add_executable(test_kernels_multi_D_avx2 tests/test_kernels_multi_D.c)
target_compile_options(test_kernels_multi_D_avx2 PRIVATE -O3 -mavx2 -mfma -mpopcnt)
target_include_directories(test_kernels_multi_D_avx2 PRIVATE include)
target_link_libraries(test_kernels_multi_D_avx2 PRIVATE OpenMP::OpenMP_C m)
add_test(NAME kernels_multi_D_avx2 COMMAND test_kernels_multi_D_avx2)

# ---- T2: cross-SIMD parity ----
#
# Same source compiled twice. simd_parity_native runs whichever SIMD path
# -march=native picks (AVX-512 on Sapphire Rapids / AMD Zen4+, AVX2 on
# older AMD/Intel). simd_parity_avx2 forces the AVX2 fallback. Each
# binary writes simd_parity_<arg>.out; simd_parity_dlopen byte-diffs
# the pair via cmake -E compare_files.
#
# Note: cmake -E compare_files is strict byte-equality. Hamming is
# integer-only and bit-identical across paths; asym scores may differ
# in low FP bits on hosts where AVX-512 actually changes the reduction
# order. If simd_parity_dlopen fails on a host that has AVX-512, the
# fix is to replace the diff command with a tolerant Python helper
# (tools/parity_check.py) that strict-checks the hamming section and
# tolerance-checks the asym section. Until that happens, the strict
# diff is the simpler and tighter assertion.

add_executable(test_simd_parity_native tests/test_simd_parity.c)
target_compile_options(test_simd_parity_native PRIVATE ${COMMON_FLAGS})
target_include_directories(test_simd_parity_native PRIVATE include)
target_link_libraries(test_simd_parity_native PRIVATE OpenMP::OpenMP_C m)

add_executable(test_simd_parity_avx2 tests/test_simd_parity.c)
target_compile_options(test_simd_parity_avx2 PRIVATE
    -O3 -mavx2 -mfma -mpopcnt -mno-avx512f)
target_include_directories(test_simd_parity_avx2 PRIVATE include)
target_link_libraries(test_simd_parity_avx2 PRIVATE OpenMP::OpenMP_C m)

# Per-binary smoke (writes its output file with no assertions; the diff
# happens in simd_parity_dlopen).
add_test(NAME simd_parity_native COMMAND test_simd_parity_native native)
add_test(NAME simd_parity_avx2   COMMAND test_simd_parity_avx2   avx2)
set_tests_properties(simd_parity_native simd_parity_avx2
    PROPERTIES FIXTURES_SETUP simd_parity_outputs)

add_test(NAME simd_parity_dlopen
    COMMAND ${CMAKE_COMMAND} -E compare_files
        simd_parity_native.out simd_parity_avx2.out)
set_tests_properties(simd_parity_dlopen
    PROPERTIES FIXTURES_REQUIRED simd_parity_outputs)
