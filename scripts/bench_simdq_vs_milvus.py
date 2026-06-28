#!/usr/bin/env python
"""bench_simdq_vs_milvus.py — apples-to-apples search-speed comparison.

Benchmarks simdq's in-process scan against Milvus FLAT (exact fp32 brute force)
on the **same vectors, same machine, same query set**, with a single identical
harness. Search-only: queries are random unit vectors generated up front (the
scan cost is data-independent), so the numbers exclude GPU encode and reflect
just the retrieval engine.

Vectors come straight from an existing simdq index's stored fp16 floats
(`floats.bin`), so Milvus FLAT scans exactly the corpus simdq was built on. With
the default `projection=identity` those are the original encoder embeddings.

For each engine it reports two metrics:
  * serial      — one query per call (latency; Milvus is dominated by per-call
                  RPC overhead here, so treat it as a loose bound).
  * concurrent  — `--workers` queries in flight via a thread pool (throughput;
                  the meaningful "serving speed"). simdq's native scan and
                  pymilvus's RPC both release the GIL, so threads parallelize.

Milvus must be reachable at `--milvus-uri` (default a local standalone server).
This produced the numbers in docs/simdq_768_investigation.md (Phase 5) and the
slides' speed table.

Usage:
    # one or more pre-built simdq indexes (768d and 384d shown)
    python scripts/bench_simdq_vs_milvus.py \
        experiments/nq_new/simdq_data/nq_dev-simdq-simdq-granite97m-512-100-20260306 \
        experiments/nq_new/simdq_data/nq_dev-simdq-simdq-granite311m-512-100-20260306

    python scripts/bench_simdq_vs_milvus.py <index_dir> \
        --queries 400 --top-k 100 --alpha 10 --workers 16 \
        --milvus-uri http://localhost:19530
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex


def _unit_queries(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, d), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return q


def _median_serial(call, qs) -> float:
    times = np.empty(len(qs), dtype=np.float64)
    for i in range(len(qs)):
        t0 = time.perf_counter()
        call(qs[i])
        times[i] = (time.perf_counter() - t0) * 1e3
    return float(np.median(times))


def _concurrent_ms_per_q(call, qs, workers: int) -> float:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(call, qs))
    return (time.perf_counter() - t0) / len(qs) * 1e3


def _bench_milvus(uri, vecs, qs, K, workers, warmup):
    """Build a FLAT collection, return (serial_ms, concurrent_ms)."""
    from pymilvus import MilvusClient, DataType

    N, D = vecs.shape
    cli = MilvusClient(uri=uri)
    coll = f"bench_flat_{D}"
    if cli.has_collection(coll):
        cli.drop_collection(coll)
    schema = cli.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("v", DataType.FLOAT_VECTOR, dim=D)
    cli.create_collection(coll, schema=schema)
    for s in range(0, N, 20000):
        e = min(s + 20000, N)
        cli.insert(coll, [{"id": i, "v": vecs[i]} for i in range(s, e)])
    ip = cli.prepare_index_params()
    ip.add_index(field_name="v", index_type="FLAT", metric_type="IP")
    cli.create_index(coll, ip)
    cli.load_collection(coll)
    time.sleep(2)
    sp = {"metric_type": "IP"}

    def call(q):
        cli.search(coll, [q.tolist()], limit=K, search_params=sp, anns_field="v")

    for i in range(warmup):
        call(qs[i])
    try:
        return _median_serial(call, qs), _concurrent_ms_per_q(call, qs, workers)
    finally:
        cli.drop_collection(coll)


def _bench_simdq(idx, qs, K, K_prime, workers, warmup):
    """Return (serial_ms[all-core scan], concurrent_ms[1-thread scan per query])."""
    for i in range(warmup):
        idx.search(qs[i], K=K, K_prime=K_prime, num_threads=0)
    # serial: a single query may use all cores for its scan
    serial = _median_serial(
        lambda q: idx.search(q, K=K, K_prime=K_prime, num_threads=0), qs)
    # concurrent: parallelism is over queries, each scan single-threaded
    conc = _concurrent_ms_per_q(
        lambda q: idx.search(q, K=K, K_prime=K_prime, num_threads=1), qs, workers)
    return serial, conc


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indexes", nargs="+", help="simdq index dir(s) to benchmark")
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=100, help="K")
    ap.add_argument("--alpha", type=int, default=10,
                    help="rescore alpha (K' = min(K*alpha, 256))")
    ap.add_argument("--workers", type=int, default=16, help="concurrent query workers")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--milvus-uri", default="http://localhost:19530")
    ap.add_argument("--no-milvus", action="store_true",
                    help="benchmark simdq only (skip Milvus)")
    args = ap.parse_args()

    K = args.top_k
    K_prime = max(K, min(K * args.alpha, 256))

    print(f"# bench: queries={args.queries} K={K} K'={K_prime} "
          f"workers={args.workers} milvus={'off' if args.no_milvus else args.milvus_uri}")

    for path in args.indexes:
        idx = SimdqIndex.load(path)
        N, D = idx.n_vectors, idx.D_orig
        vecs = np.ascontiguousarray(np.asarray(idx.floats_mmap, dtype=np.float32)) \
            if idx.has_floats else None
        qs = _unit_queries(args.queries, D, args.seed)

        print(f"\n=== {path.rstrip('/').split('/')[-1]}: N={N} D={D} b={idx.b} ===")

        if not args.no_milvus:
            if vecs is None:
                print("  Milvus  : skipped (index has no stored floats to load)")
                m_ser = m_cc = None
            else:
                m_ser, m_cc = _bench_milvus(
                    args.milvus_uri, vecs, qs, K, args.workers, args.warmup)
                print(f"  Milvus FLAT : serial {m_ser:8.2f} ms/q | "
                      f"{args.workers}-conc {m_cc:6.2f} ms/q ({1000 / m_cc:.0f} q/s)")
        else:
            m_cc = None

        s_ser, s_cc = _bench_simdq(idx, qs, K, K_prime, args.workers, args.warmup)
        print(f"  simdq  b={idx.b}  : serial {s_ser:8.2f} ms/q | "
              f"{args.workers}-conc {s_cc:6.2f} ms/q ({1000 / s_cc:.0f} q/s)")
        if m_cc is not None:
            print(f"  -> simdq is {m_cc / s_cc:.1f}x faster (concurrent)")


if __name__ == "__main__":
    main()
