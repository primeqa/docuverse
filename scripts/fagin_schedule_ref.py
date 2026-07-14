#!/usr/bin/env python
"""fagin_schedule_ref.py — Python reference + benchmark for weighted sorted-access
schedules in Fagin's Threshold Algorithm.

Compares three sorted-access schedules on the SAME index + queries, all exact
(epsilon=0), and reports the cost metric that matters for a native port:

  * random_accesses — distinct docs scored (full dot products). This dominates
    TA cost and is implementation-order-independent given the halting point.
  * sorted_accesses — total rows consumed across per-dim lists (Sum_j d_j).
  * rounds          — heap/threshold updates (a proxy for control overhead).
  * agree@K vs the exact FLAT top-K (must be 1.0000 at epsilon=0 for all
    schedules — the whole point is that reordering sorted access stays exact).

Schedules
  lockstep      round-robin: advance every active dim by `batch` each round
                (reproduces the current fagin_ta.h kernel).
  contribution  priority by current frontier contribution f_j = q_j * t_j —
                advance the list holding the biggest remaining slice of T.
                (Known trap: never halts — kept for reference.)
  steepest      priority by marginal threshold drop over the next chunk
                (Quick-Combine indicator): advance the list that pulls T down
                fastest per row consumed.
  gated-<base>  <base> advancement + candidate gating (CA / TA-theta style):
                sorted access accumulates each doc's exact partial score
                instead of triggering an immediate random access. A doc is
                random-accessed only when its upper bound
                    ub(d) = partial(d) + sum_{unseen j} q_j * t_j
                could still crack the current kth score (and the RA computes
                only the unseen dims). ub(d) and the frontier contributions
                are monotone non-increasing while the kth is non-decreasing,
                so ub(d) <= kth + eps prunes a candidate PERMANENTLY. Docs
                seen in all active dims are promoted to the heap for free.
                e.g. gated-steepest, gated-lockstep.

Because gating shifts cost from random access into sorted access (each sorted
row now does one multiply-add for the partial score), the table also reports
`flops` = madds spent on scoring (sorted-access accumulation + RA dims;
ungated schedules count ra * n_active). Flat scan baseline = N * n_active.

The threshold bound T = Sum_j f_j(d_j) with f_j = q_j * (last consumed value)
stays a valid upper bound on any unseen doc's score regardless of how far each
individual cursor has advanced, so ANY per-list schedule is exact — the lock-
step schedule is just one (weight-blind) choice.

Usage:
    python scripts/fagin_schedule_ref.py \
        --index experiments/nq_new/simq_data/nq_dev-simdq-simdq-granite97m-512-100-20260702 \
        --queries scratch/nq_hw/q97m.npy --nq 200 --K 100 --batch 64
"""
from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _flat_topk(Y: np.ndarray, q: np.ndarray, K: int) -> np.ndarray:
    s = Y @ q
    part = np.argpartition(-s, K - 1)[:K]
    return part[np.argsort(-s[part])]


def run_schedule(q: np.ndarray, Y: np.ndarray, order: np.ndarray,
                 vals: np.ndarray, K: int, batch: int, schedule: str,
                 epsilon: float = 0.0) -> dict:
    """One query, one schedule. Returns stats + retrieved ids (best-first).

    order/vals are (D, N): order[j] = argsort of Y[:,j] desc, vals[j] the
    matching sorted values. Positive-weight dims walk the list top-down;
    negative-weight dims walk it bottom-up (the double scan).
    """
    N, D = Y.shape
    gated = schedule.startswith("gated-")
    base = schedule[len("gated-"):] if gated else schedule
    active = np.nonzero(q != 0.0)[0]
    na = active.shape[0]
    if na == 0:
        idxs = np.arange(min(K, N))
        return {"random_accesses": 0, "sorted_accesses": 0, "rounds": 0,
                "depth_max": 0, "flops": 0, "promoted": 0, "idxs": idxs}

    qa = q[active].astype(np.float32)
    pos = qa > 0.0                       # per-active-dim scan direction

    d = np.zeros(na, dtype=np.int64)     # rows consumed per active dim
    seen = np.zeros(N, dtype=bool)       # fully scored (heap-considered)
    heap: list[float] = []               # min-heap of top-K scores
    ra = 0                               # random accesses (distinct docs scored)
    sa = 0                               # sorted accesses (rows consumed)
    rounds = 0
    flops = 0                            # scoring madds (partials + RA dims)
    promoted = 0                         # docs completed via sorted access only

    if gated:
        partial = np.zeros(N, dtype=np.float64)   # exact partial scores
        nseen_d = np.zeros(N, dtype=np.int32)     # active dims seen per doc
        seen_mask = np.zeros((N, na), dtype=bool)  # which active dims seen
        dead = np.zeros(N, dtype=bool)             # pruned forever

    def frontier_contrib(a: int, di: int) -> float:
        """f_a = q_a * (last consumed value) after `di` rows; loose bound at di=0.
        Clamps di to [0, N] so a lookahead past the list end just returns the
        last real frontier value (contribution stops dropping at exhaustion)."""
        j = active[a]
        di = min(di, N)
        if pos[a]:
            idx = di - 1 if di >= 1 else 0
            return float(qa[a]) * float(vals[j, idx])
        idx = N - di if di >= 1 else N - 1
        return float(qa[a]) * float(vals[j, idx])

    # T = sum of current frontier contributions (upper bound on any unseen doc).
    contrib = np.array([frontier_contrib(a, 0) for a in range(na)],
                       dtype=np.float64)
    T = float(contrib.sum())

    def next_rows(a: int, take: int) -> tuple[np.ndarray, np.ndarray]:
        """(doc ids, values) for the next `take` rows of active dim a."""
        j = active[a]
        di = int(d[a])
        take = min(take, N - di)
        if pos[a]:
            return order[j, di:di + take], vals[j, di:di + take]
        return (order[j, N - di - take:N - di],
                vals[j, N - di - take:N - di])

    def heap_push(sc: float, did: int) -> None:
        if len(heap) < K:
            heapq.heappush(heap, (sc, did))
        elif sc > heap[0][0]:
            heapq.heapreplace(heap, (sc, did))

    def advance(a: int, take: int) -> None:
        nonlocal ra, sa, T, flops, promoted
        ids, vs = next_rows(a, take)
        take = ids.shape[0]
        if take == 0:
            return
        d[a] += take
        sa += take
        if not gated:
            fresh = ids[~seen[ids]]
            if fresh.shape[0]:
                seen[fresh] = True
                scores = Y[fresh] @ q        # random access: full dot products
                ra += fresh.shape[0]
                flops += fresh.shape[0] * na
                for did, sc in zip(fresh.tolist(), scores.tolist()):
                    heap_push(sc, did)
        else:
            m = ~(seen[ids] | dead[ids])
            alive = ids[m]
            if alive.shape[0]:
                partial[alive] += float(qa[a]) * vs[m].astype(np.float64)
                flops += alive.shape[0]
                seen_mask[alive, a] = True
                nseen_d[alive] += 1
                comp = alive[nseen_d[alive] == na]
                for did in comp.tolist():    # partial is exact: promote free
                    seen[did] = True
                    promoted += 1
                    heap_push(float(partial[did]), did)
        new_f = frontier_contrib(a, int(d[a]))
        T += new_f - contrib[a]
        contrib[a] = new_f

    # ---- gated helpers -----------------------------------------------------
    def candidate_ids() -> np.ndarray:
        return np.nonzero((nseen_d > 0) & ~seen & ~dead)[0]

    def ubs_of(cids: np.ndarray) -> np.ndarray:
        """ub(d) = partial(d) + T - (contrib over dims already seen for d)."""
        rem = T - seen_mask[cids].astype(np.float64) @ contrib
        return partial[cids] + rem

    def gated_ra(did: int) -> None:
        """Random-access only the dims not yet covered by sorted access."""
        nonlocal ra, flops
        unseen_dims = active[~seen_mask[did]]
        sc = float(partial[did])
        if unseen_dims.shape[0]:
            sc += float(Y[did, unseen_dims] @ q[unseen_dims])
            flops += unseen_dims.shape[0]
        ra += 1
        seen[did] = True
        heap_push(sc, did)

    def bootstrap() -> None:
        """Fill the heap with the K best candidates by upper bound."""
        need = K - len(heap)
        cids = candidate_ids()
        if need <= 0 or cids.shape[0] == 0:
            return
        top = np.argsort(-ubs_of(cids))[:need]
        for did in cids[top].tolist():
            gated_ra(did)

    def drain() -> bool:
        """RA every candidate whose bound could crack the kth; prune the rest.
        ub snapshots stay valid during the pass (contribs are static here and
        only the kth rises), and pruned docs can never come back because ub
        is non-increasing while the kth is non-decreasing."""
        cids = candidate_ids()
        if cids.shape[0] == 0:
            return True
        ub = ubs_of(cids)
        order_ub = np.argsort(-ub)
        for i, oi in enumerate(order_ub.tolist()):
            if ub[oi] <= heap[0][0] + epsilon:
                dead[cids[order_ub[i:]]] = True
                break
            gated_ra(int(cids[oi]))
        return True

    def halt() -> bool:
        if gated and len(heap) < K:
            bootstrap()
        if len(heap) < K or heap[0][0] < T - epsilon:
            return False
        return drain() if gated else True

    if base == "lockstep":
        while int(d.min()) < N:
            for a in range(na):
                advance(a, batch)
            rounds += 1
            if halt():
                break
    elif base in ("contribution", "steepest"):
        # Max-priority queue over active dims (heapq is a min-heap -> negate).
        def key(a: int) -> float:
            if int(d[a]) >= N:
                return -np.inf
            if base == "contribution":
                return float(contrib[a])            # biggest remaining slice of T
            nxt = frontier_contrib(a, int(d[a]) + batch)
            return float(contrib[a] - nxt)          # marginal drop over next chunk
        pq = [(-key(a), a) for a in range(na)]
        heapq.heapify(pq)
        while pq:
            _, a = heapq.heappop(pq)
            advance(a, batch)
            rounds += 1
            if halt():
                break
            k = key(a)
            if k > -np.inf:
                heapq.heappush(pq, (-k, a))
    else:
        raise ValueError(f"unknown schedule {schedule!r}")

    top = sorted(heap, key=lambda x: -x[0])
    idxs = np.array([did for _, did in top], dtype=np.int64)
    return {"random_accesses": ra, "sorted_accesses": sa, "rounds": rounds,
            "depth_max": int(d.max()), "flops": flops, "promoted": promoted,
            "idxs": idxs}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="SimdqIndex or FaginIndex dir")
    ap.add_argument("--queries", required=True, help="(Q, D) fp32 .npy")
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epsilon-sweep", dest="epsilon_sweep",
                    default="0.0,0.3,0.5,0.7,0.8,0.9",
                    help="comma-separated epsilons for the recall-vs-cost frontier")
    ap.add_argument("--schedules", default="lockstep,steepest,gated-steepest")
    args = ap.parse_args()

    idx_dir = Path(args.index)
    if (idx_dir / "floats.bin").exists():
        from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
        Y = np.asarray(SimdqIndex.load(idx_dir).floats_mmap, dtype=np.float32)
    else:
        from docuverse.engines.retrieval.fagin.fagin_index import FaginIndex
        Y = np.asarray(FaginIndex.load(idx_dir).Y, dtype=np.float32)
    Y = np.ascontiguousarray(Y)
    N, D = Y.shape

    # per-dim descending sort (same as FaginIndex.build)
    perm = np.argsort(-Y, axis=0, kind="stable")
    vals = np.ascontiguousarray(np.take_along_axis(Y, perm, axis=0).T)  # (D,N)
    order = np.ascontiguousarray(perm.T).astype(np.int32)               # (D,N)

    Q = np.load(args.queries, mmap_mode="r")
    nq = min(args.nq, Q.shape[0])
    qs = np.ascontiguousarray(Q[:nq].astype(np.float32))
    if qs.shape[1] != D:
        sys.exit(f"query dim {qs.shape[1]} != index dim {D}")
    print(f"# index N={N} D={D}  queries={nq}  K={args.K}  batch={args.batch}")

    flat = [set(_flat_topk(Y, qs[i], args.K).tolist()) for i in range(nq)]

    scheds = [s.strip() for s in args.schedules.split(",") if s.strip()]
    epsilons = [float(e) for e in args.epsilon_sweep.split(",")]

    # Recall-vs-cost frontier: for each schedule, sweep epsilon and record mean
    # random accesses (the dominant cost) against mean agree@K vs exact FLAT.
    # A better schedule reaches a given recall at fewer random accesses.
    print(f"# flat-scan baseline: {N * D / 1e6:.1f}M madds/query")
    for s in scheds:
        print(f"\n## schedule = {s}", flush=True)
        print(f"{'epsilon':>8s} {'rand_acc':>10s} {'sorted_acc':>12s} "
              f"{'flops_M':>9s} {'promo':>7s} "
              f"{'depth_max':>10s} {'rounds':>8s} {'agree@K':>9s}")
        for eps in epsilons:
            tot_ra = tot_sa = tot_r = tot_dm = tot_fl = tot_pr = 0
            agree = 0.0
            for i in range(nq):
                r = run_schedule(qs[i], Y, order, vals, args.K, args.batch, s,
                                 eps)
                tot_ra += r["random_accesses"]
                tot_sa += r["sorted_accesses"]
                tot_r += r["rounds"]
                tot_dm += r["depth_max"]
                tot_fl += r["flops"]
                tot_pr += r["promoted"]
                agree += len(set(r["idxs"].tolist()) & flat[i]) / args.K
            print(f"{eps:8.3f} {tot_ra / nq:10.1f} {tot_sa / nq:12.1f} "
                  f"{tot_fl / nq / 1e6:9.2f} {tot_pr / nq:7.1f} "
                  f"{tot_dm / nq:10.1f} {tot_r / nq:8.1f} {agree / nq:9.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
