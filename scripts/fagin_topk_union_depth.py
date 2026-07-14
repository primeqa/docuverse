#!/usr/bin/env python
"""Fagin sorted-access "union depth" to surface the true top-K.

For each query, take its exact top-K documents (fp32 inner product), then find
the sorted-access depth d such that every one of those K docs has appeared by
position d in AT LEAST ONE column's sorted list. Per doc that is
    min over dims j of (rank of the doc in column j),
and the query's answer is the max of that over the K docs:
    union_depth(q) = max_{d in topK} min_j rank_j(d).

Columns are read in the query-aware direction (descending where q_j > 0,
ascending where q_j < 0), matching FaginIndex sorted access. This is the
first-surfacing / union bound; it is a LOWER bound on where plain Fagin TA can
halt (TA additionally needs the threshold to drop below the k-th score).

Vectors come from a simdq index's floats.bin (the same fp32 source the FLAT /
Fagin baselines scan); queries from a per-model .npy.

Usage:
    conda activate ndocu
    python scripts/fagin_topk_union_depth.py \
        --index experiments/nq_new/simq_data/nq_dev-simdq-simdq-granite97m-512-100-20260702 \
        --queries /home/raduf/sandbox2/docuverse/scratch/nq_hw/q97m.npy \
        --nq 50 --topk 100

See research/2026-07-14-fagin-schedule-comparison-nq.md.
"""
import argparse

import numpy as np

from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex


def union_depths(V: np.ndarray, Q: np.ndarray, topk: int,
                 col_chunk: int = 64) -> np.ndarray:
    """Return the per-query union depth (see module docstring)."""
    N, D = V.shape
    nq = Q.shape[0]

    scores = Q @ V.T                                        # (nq, N)
    top = np.argpartition(-scores, topk, axis=1)[:, :topk]  # (nq, topk) unordered

    # Gather the descending rank of every needed doc in every column once.
    union = np.unique(top)
    pos = {int(d): i for i, d in enumerate(union)}
    rankdesc = np.empty((len(union), D), dtype=np.int64)    # 0 == largest value
    for c0 in range(0, D, col_chunk):
        cols = V[:, c0:c0 + col_chunk]
        order = np.argsort(-cols, axis=0, kind="stable")
        rk = np.empty_like(order)
        rows = np.broadcast_to(np.arange(N)[:, None], order.shape)
        np.put_along_axis(rk, order, rows, axis=0)
        rankdesc[:, c0:c0 + col_chunk] = rk[union]

    depths = np.empty(nq, dtype=np.int64)
    for i in range(nq):
        ridx = np.array([pos[int(d)] for d in top[i]])
        rd = rankdesc[ridx]                                 # (topk, D) desc ranks
        # 1-indexed depth: desc rank if q_j > 0 else ascending rank (N-1-rd).
        depth_col = np.where(Q[i] > 0, rd, N - 1 - rd) + 1
        depths[i] = int(depth_col.min(axis=1).max())        # deepest of the K
    return depths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="simdq index dir (floats.bin)")
    ap.add_argument("--queries", required=True, help="query vectors .npy")
    ap.add_argument("--nq", type=int, default=50, help="use the first NQ queries")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    idx = SimdqIndex.load(args.index)
    V = np.ascontiguousarray(np.asarray(idx.floats_mmap, dtype=np.float32))
    N, D = V.shape
    Q = np.load(args.queries).astype(np.float32)[:args.nq]

    depths = union_depths(V, Q, args.topk)

    print(f"=== {args.index}")
    print(f"    N={N}, D={D}, first {args.nq} queries, top-{args.topk} ===")
    print(f"depth to surface all top-{args.topk} in some column, per query:")
    print(f"  min={depths.min()}  median={int(np.median(depths))}  "
          f"mean={depths.mean():.1f}  max={depths.max()}  "
          f"p90={int(np.percentile(depths, 90))}  "
          f"p99={int(np.percentile(depths, 99))}")
    print(f"  as fraction of corpus: median={np.median(depths) / N:.4%}  "
          f"max={depths.max() / N:.4%}")
    print("  per-query depths:")
    for i in range(0, args.nq, 10):
        print("   ", " ".join(f"{d:6d}" for d in depths[i:i + 10]))


if __name__ == "__main__":
    main()
