# Exact candidate pruning cannot beat brute force in Fagin on dense embeddings

**Date:** 2026-07-14
**Status:** Negative result — proven and measured. Exact frontier / block-max
pruning is void on L2-normalized dense embeddings; incremental score
accumulation conserves (and likely regresses) total work.
**Files:** `scripts/fagin_phase_timing.py` (cost model),
`docuverse/engines/retrieval/simdq/_native/include/fagin_ta.h` (kernel).

Related: [[2026-07-14-fagin-schedule-comparison-nq]] (per-phase timing that
motivated this),
`docs/superpowers/plans/2026-07-14-fagin-candidate-pruning.md` (the plan whose
Phase 3 this result cancels).

## Context

Per-phase timing of the Fagin kernel on NQ (see the schedule-comparison note)
showed that **random access — the full fp32 dot products — dominates wall clock**
and that `random_accesses == N` for every schedule: Fagin scores the *entire*
corpus. That looks like enormous waste — only `K` docs can be in the answer — so
the natural optimization is to prune: use the current k-th-best score to skip
full dot products (WAND / MaxScore / block-max, the IR family), or to accumulate
scores incrementally during sorted access.

This note shows both ideas fail on dense embeddings, one by a theorem and one by
conservation of arithmetic.

## Measured ceiling

`granite-97m`, N=225,669, D=384, first 50 NQ-dev queries, top-100, ε=0, lockstep:

| quantity | value |
|---|---:|
| docs scored (random accesses) / query | 225,669 (100% of corpus) |
| docs that are *real* contenders (true score ≥ kth) / query | 100.0 |
| dot products spent on non-contenders | **99.96%** |

So 99.96% of the scoring work is on docs that never had a chance. The question is
whether any *exact* mechanism can reclaim it. It cannot.

## Result 1 — frontier upper-bound pruning is mathematically void

**Claim.** In exact TA, every document surfaced by sorted access has a score
upper bound `UB ≥ T`, the current halt threshold. Since scoring only continues
while `kth < T`, no surfaced doc can ever be pruned before the algorithm halts
globally. Pruning removes zero dot products.

**Proof.** When a doc `id` is surfaced, split the dims into `S` (lanes where it
has already been passed in sorted order — its exact contribution is known) and
`U` (not yet seen). Its score upper bound is

    UB(id) = Σ_{j∈S} q_j·y_{id,j}  +  Σ_{j∈U} q_j·t_j          (1)

where `t_j` is lane `j`'s current frontier value (`q_j·t_j` is the standard TA
per-lane bound). The threshold is `T = Σ_{all j} q_j·t_j`.

A doc surfaces in lane `j` *because* its cursor reached it, i.e. it sits at or
above the frontier in that lane's scan direction, so

    q_j·y_{id,j} ≥ q_j·t_j   for every j ∈ S.                  (2)

Substituting (2) into (1),

    UB(id) ≥ Σ_{j∈S} q_j·t_j + Σ_{j∈U} q_j·t_j = Σ_{all j} q_j·t_j = T.

So `UB(id) ≥ T`. The halt rule keeps scoring only while `kth < T`, hence
`UB(id) ≥ T > kth`: the candidate cannot be excluded from the top-K on bounds
alone. Pruning fires only *at* the halt point, where the algorithm already
stops. ∎

This is Fagin's instance-optimality surfacing: TA already stops as early as any
bound-based algorithm can. There is no slack for a pruning layer to exploit.
Block-max WAND is the same bound with per-block granularity, so it inherits the
same wall — plus it needs *sparsity* (a doc absent from most query terms) to have
a small UB, which dense embeddings (mass in all D dims) never provide. This is
the same reason the norm-aware GTA halt gave zero benefit under lockstep in the
companion note.

## Result 2 — incremental accumulation conserves total work (and regresses)

The second idea: accumulate `q_j·y_{id,j}` into a per-doc partial score *during*
sorted access, so random access only fills the unseen lanes instead of
recomputing the whole dot product.

**Arithmetic is conserved.** Because `random_accesses == N` regardless (every doc
is surfaced and must ultimately be scored), the total useful multiply-adds are
fixed at `N·D`:

    97m: N·D = 225,669 × 384 = 86.7M mul-adds.

Sorted access touches `D × depth ≈ 28.75M` (doc, lane) pairs. Accumulating those
early shrinks random access to `86.7M − 28.75M = 57.9M` — but adds the same
28.75M to the sorted-access phase. Total: **86.7M, unchanged.** Accumulation
moves work between phases; it does not remove any.

**And the redistribution is a net loss**, for three compounding reasons:

1. **Contiguous SIMD → scattered gather.** The random-access dot product streams
   contiguous memory (`Y + id·D`, `q`), which `-O3 -march=native` vectorizes to
   full-width AVX. Incremental accumulation does `partial[id] += q_j·val` with
   `id` effectively random (whoever sits at that sorted position) — 28.75M
   scattered read-modify-writes into a 225K array. Cache-hostile; far slower per
   op than streaming SIMD.
2. **New reads added to Phase A.** Sorted access currently reads only the id
   (`order[j][pos]`) and a seen-bit; it never touches `vals`. Accumulation forces
   it to also read the value and scatter it. Pure addition.
3. **De-vectorizes the remainder.** Skipping already-seen lanes in random access
   requires a per-doc lane mask; a masked/gathered partial dot over an arbitrary
   subset of D dims does not vectorize. Computing all D contiguously with SIMD is
   cheaper than computing a subset with a gather.

So incremental accumulation trades a fast contiguous loop for a slow scattered
one and de-vectorizes the leftover — a likely regression for zero arithmetic
saved.

## Conclusion

On L2-normalized dense embeddings, **exact top-K over per-dimension sorted lists
is fundamentally ~brute-force in the scoring phase.** Fagin's optimality closes
the door on bound-based pruning, and dense magnitude structure closes it on
sparsity-based (WAND/block-max) pruning. Neither is an implementation gap — both
are provable.

The levers that remain actually change the term count or per-term cost:

- **Cheaper per-doc scoring** — quantized codes instead of fp32 dots. **simdq
  already does this** (2-bit asymmetric + 1-bit hamming); it is the real
  production answer to "scoring dominates."
- **Approximate top-K (ε>0)** — the honest speed/accuracy knob: prune when
  `UB < kth·(1+ε)`. Already supported by the ε parameter.
- **Graph ANN (HNSW)** — sublinear in *docs visited*, which is why it dominates
  the frontier in the main simdq report.

What is *not* void and remains worth doing is reducing the **per-round overhead**
of the steepest schedules (the GTASD water-filling threshold recompute that the
timing note showed dominates its latency) — that is Phase 2 of the plan, pursued
next, and is independent of everything above.
