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
