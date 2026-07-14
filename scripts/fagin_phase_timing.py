#!/usr/bin/env python
"""Per-phase wall-clock timing of the Fagin kernel: where does time go?

For each schedule (TA / TASD / GTA / GTASD) runs the first NQ queries at
epsilon=0 (exact) and reports mean ms/query split by phase, using the ns_*
timers added to the kernel stats dict:

    sorted    - Phase A: sorted access (gather unseen candidates)
    random    - Phase B: full dot products (the "random access" scoring)
    heap      - Phase C: top-K heap maintenance
    threshold - halting-threshold computation (incl. norm-aware water-filling)
    other     - ns_total minus the four phases (glue / setup / extract)

The hypothesis this script tests: `random` (full dot products) dominates for all
four schedules, because random_accesses == N (Fagin scores the whole corpus).
If so, candidate pruning (Phase 3 of the plan) is the priority, not schedule
tuning. num_threads is pinned to 1 so the split is interpretable.

Usage:
    conda activate ndocu
    python scripts/fagin_phase_timing.py --tag "granite-97m-r2 (384d)" \
        --index experiments/nq_new/simq_data/nq_dev-simdq-simdq-granite97m-512-100-20260702 \
        --queries /home/raduf/sandbox2/docuverse/scratch/nq_hw/q97m.npy \
        --nq 50 --topk 100 --batch 2048

See docs/superpowers/plans/2026-07-14-fagin-candidate-pruning.md and
research/2026-07-14-fagin-schedule-comparison-nq.md.
"""
import argparse

import numpy as np

from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
from docuverse.engines.retrieval.fagin.fagin_index import FaginIndex

SCHEDS = ["lockstep", "steepest", "lockstep_norm", "steepest_norm"]
NAME = {"lockstep": "TA", "steepest": "TASD",
        "lockstep_norm": "GTA", "steepest_norm": "GTASD"}
PHASES = ["ns_sorted", "ns_random", "ns_heap", "ns_threshold"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--index", required=True, help="simdq index dir (floats.bin)")
    ap.add_argument("--queries", required=True, help="query vectors .npy")
    ap.add_argument("--nq", type=int, default=50)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()

    idx = SimdqIndex.load(args.index)
    V = np.ascontiguousarray(np.asarray(idx.floats_mmap, dtype=np.float32))
    N, D = V.shape
    Q = np.load(args.queries).astype(np.float32)[:args.nq]
    fg = FaginIndex.build(V)

    # Warm caches: one throwaway pass per schedule (not timed).
    for s in SCHEDS:
        fg.search(np.ascontiguousarray(Q[0]), K=args.topk, batch=args.batch,
                  epsilon=0.0, max_depth=0, num_threads=1, schedule=s)

    agg = {s: {k: 0 for k in PHASES + ["ns_total", "random_accesses"]}
           for s in SCHEDS}
    for i in range(args.nq):
        q = np.ascontiguousarray(Q[i])
        per = {}
        for s in SCHEDS:
            ii, _ss, st = fg.search(q, K=args.topk, batch=args.batch, epsilon=0.0,
                                    max_depth=0, num_threads=1, schedule=s)
            per[s] = set(ii.tolist())
            for k in PHASES + ["ns_total", "random_accesses"]:
                agg[s][k] += st[k]
        ref = per["lockstep"]
        for s in SCHEDS:
            assert per[s] == ref, f"{args.tag} q{i} {NAME[s]} disagrees with TA"

    nq = args.nq
    print(f"\n=== {args.tag}  (N={N}, D={D}, first {nq} q, top-{args.topk}, "
          f"batch={args.batch}, 1 thread) — mean ms/query by phase ===")
    hdr = (f"{'sched':6s} {'total':>8s} {'sorted':>8s} {'random':>8s} "
           f"{'heap':>8s} {'thresh':>8s} {'other':>8s}  {'rand/N':>7s}  "
           f"{'random%':>8s}")
    print(hdr)
    for s in SCHEDS:
        a = agg[s]
        tot = a["ns_total"] / nq / 1e6
        sr = a["ns_sorted"] / nq / 1e6
        rd = a["ns_random"] / nq / 1e6
        hp = a["ns_heap"] / nq / 1e6
        th = a["ns_threshold"] / nq / 1e6
        other = tot - sr - rd - hp - th
        randfrac = a["random_accesses"] / nq / N
        randpct = a["ns_random"] / a["ns_total"] if a["ns_total"] else 0.0
        print(f"{NAME[s]:6s} {tot:8.2f} {sr:8.2f} {rd:8.2f} {hp:8.2f} "
              f"{th:8.2f} {other:8.2f}  {randfrac:7.1%}  {randpct:8.1%}")


if __name__ == "__main__":
    main()
