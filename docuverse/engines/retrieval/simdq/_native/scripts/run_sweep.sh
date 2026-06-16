#!/bin/bash
# run_sweep.sh — execute every Plan-1 benchmark over canned sizes and
# print a markdown-friendly table to stdout.
#
# Usage: bash scripts/run_sweep.sh [build_dir]
#   build_dir defaults to ./build
set -euo pipefail

BUILD="${1:-build}"
BINS=(
    bench_hamming_top1
    bench_hamming
    bench_asym_b1
    bench_asym_b2
    bench_asym_b4
)
SIZES=(500000 1000000 50000000)
REPS=(10 10 3)
K=100

if [[ ! -d "$BUILD" ]]; then
    echo "build dir '$BUILD' not found; build first" >&2
    exit 1
fi

THREADS="${OMP_NUM_THREADS:-$(nproc)}"
echo "# simdq Plan 1 kernel sweep — threads=$THREADS, K=$K"
echo
printf "| binary | n | reps | output |\n"
printf "|--------|---|------|--------|\n"
for bin in "${BINS[@]}"; do
    if [[ ! -x "$BUILD/$bin" ]]; then
        printf "| %s | — | — | (binary missing) |\n" "$bin"
        continue
    fi
    for i in "${!SIZES[@]}"; do
        n="${SIZES[$i]}"
        reps="${REPS[$i]}"
        out=$(OMP_NUM_THREADS="$THREADS" "$BUILD/$bin" "$n" "$reps" "$K" 2>&1 | tail -1)
        printf "| %s | %s | %s | %s |\n" "$bin" "$n" "$reps" "$out"
    done
done
