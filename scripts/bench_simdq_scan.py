#!/usr/bin/env python
"""bench_simdq_scan.py — isolate simdq scan time vs corpus size and thread count.

This benchmarks ONLY SimdqIndex.search (projection + native scan + rescore),
with queries encoded once up front, so the numbers are free of GPU-encode and
parallel_process overhead. It answers two questions directly:

  1. At what corpus size does simdq's compressed SIMD scan start to win?
  2. For a given corpus size, how many threads is optimal? (OpenMP parallel-
     region + critical-merge overhead makes "all cores" a loss at small N.)

Vectors are random unit vectors (the scan cost is data-independent, so this is
representative of real corpora of the same shape). For each corpus size we
build the index once, then sweep --threads, timing a fixed batch of queries.

Usage:
    python scripts/bench_simdq_scan.py
    python scripts/bench_simdq_scan.py \
        --sizes 5000,50000,500000 --threads 1,2,4,8,0 \
        --dim 768 --b 2 --queries 500 --top-k 100 --alpha 10
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex


def _unit_rows(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal((n, d), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="5000,50000,500000",
                    help="comma-separated corpus sizes N to sweep")
    ap.add_argument("--threads", default="1,2,4,8,0",
                    help="comma-separated num_threads to sweep (0 = all cores / OMP default)")
    ap.add_argument("--dim", type=int, default=768, help="embedding dim D (384/768/1024/1536)")
    ap.add_argument("--family", default="asymmetric", choices=["asymmetric", "hamming"])
    ap.add_argument("--b", type=int, default=2, choices=[1, 2, 4], help="bits (asymmetric only)")
    ap.add_argument("--queries", type=int, default=500, help="number of queries per timing run")
    ap.add_argument("--top-k", type=int, default=100, help="K")
    ap.add_argument("--alpha", type=int, default=10, help="rescore alpha (K_prime = K*alpha, capped 256)")
    ap.add_argument("--store-floats", action="store_true", default=True,
                    help="store fp16 floats for two-stage rescore (default on)")
    ap.add_argument("--no-store-floats", dest="store_floats", action="store_false")
    ap.add_argument("--warmup", type=int, default=20, help="warmup queries before timing")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sizes = _parse_int_list(args.sizes)
    threads = _parse_int_list(args.threads)
    rng = np.random.default_rng(args.seed)

    K = args.top_k
    K_prime = max(K, min(K * args.alpha, 256))

    print(f"# simdq scan sweep | family={args.family} b={args.b} D={args.dim} "
          f"K={K} K'={K_prime} store_floats={args.store_floats} queries={args.queries}")
    print(f"# scan time isolates SimdqIndex.search (projection + native scan + rescore)")
    print()
    header = f"{'N':>10} " + " ".join(f"t={t if t else 'all':>10}" for t in threads)
    print(header + "    (median ms/query ; lower is better)")
    print("-" * len(header))

    # A query set is reused across sizes/threads; queries are independent of N.
    q_all = _unit_rows(args.queries + args.warmup, args.dim, rng)

    for n in sizes:
        vecs = _unit_rows(n, args.dim, rng)
        index = SimdqIndex.build(
            vecs,
            family=args.family,
            b=args.b,
            d=None,
            projection="identity",
            store_floats=args.store_floats,
        )
        row = [f"{n:>10}"]
        best_t, best_ms = None, float("inf")
        for t in threads:
            # warmup (thread-pool spin-up, page faults, caches)
            for i in range(args.warmup):
                index.search(q_all[i], K=K, K_prime=K_prime, num_threads=t)
            per_query = np.empty(args.queries, dtype=np.float64)
            for i in range(args.queries):
                q = q_all[args.warmup + i]
                t0 = time.perf_counter()
                index.search(q, K=K, K_prime=K_prime, num_threads=t)
                per_query[i] = (time.perf_counter() - t0) * 1e3
            med = float(np.median(per_query))
            row.append(f"{med:>10.3f}")
            if med < best_ms:
                best_ms, best_t = med, t
        best_label = best_t if best_t else "all"
        print(" ".join(row) + f"   <- best: t={best_label} ({best_ms:.3f} ms, {1000.0/best_ms:,.0f} q/s)")


if __name__ == "__main__":
    main()
