# FaginThresholdEngine — Design

**Date:** 2026-07-06
**Source:** Fagin's Threshold Algorithm (TA), `docs/rob-fagin-slides.pptx` slide 19 (context: slides 13–35); Fagin, Lotem, Naor, PODS '01.

## Goal

A new in-process retrieval engine, `FaginThresholdEngine`, that ingests dense
embeddings and answers **exact top-k inner-product** queries using the
Threshold Algorithm over per-dimension sorted lists, with a native
(C + OpenMP + SIMD) inner loop. It is registered as a first-class docuverse
engine (`db_engine: fagin`) so it drops into the existing speed/quality
benchmark configs alongside simdq, FAISS, and Milvus.

Positioning: a *fast competitor*, not a teaching demo — but exactness is the
default and the differentiator. Approximation is opt-in via config knobs.

## Algorithm mapping (slide 19 → dense retrieval)

- **Objects** = documents; **m attributes** = embedding dimensions;
  **scoring function** f = inner product `score(d) = Σ_j q_j · Y[d, j]`
  (monotone in each per-list contribution `q_j · x`).
- **Sorted access**: per-dimension lists sorted descending by raw value
  `Y[:, j]`. For a query, descending *contribution* order is:
  - `q_j > 0`: scan from the **top** (head) of the list;
  - `q_j < 0`: scan from the **bottom** (tail) — ascending raw value is
    descending contribution. This is the "double scan" that handles negative
    query weights and negative document values with a single stored order.
  - `q_j = 0`: dimension skipped entirely (contributes 0 to scores and to the
    threshold — remains exact).
- **Random access**: full SIMD dot product `q · Y[id]` on first sighting of a
  document.
- **Threshold**: `T = Σ_j q_j · t_j` with `t_j` = value at dimension j's
  current cursor.
- **Halting**: stop when the top-k heap holds k documents with
  `kth_score ≥ T − ε` (ε = 0 default ⇒ exact slide-19 rule), or at
  `max_depth`, or at exhaustion (`depth = N`).

**Deviation (deliberate):** the ε knob is **additive** (`T − ε`), not
multiplicative (`T/(1+ε)`), because inner-product scores/thresholds can be
negative, where division by `(1+ε)` tightens instead of relaxes.

## Architecture

Approach chosen: new `fagin/` engine package; TA kernel compiled into the
**existing `_simdq_native` extension** (reuses CMake/setup.py arch detection,
OpenMP wiring, and `simdq_topk.h`). If the kernel later deserves its own
extension, it is a self-contained header + one binding function to move.

### Index (`FaginIndex`)

Built at ingest from the fp32 embedding matrix `Y[N, D]`:

| Artifact | Type/shape | Purpose |
|---|---|---|
| `Y` | fp32 `[N, D]` row-major | random-access scoring (one dot product per candidate) |
| `order` | int32 `[D, N]` | per-dimension argsort of `Y[:, j]`, descending; built with NumPy at ingest |
| `vals` | fp32 `[D, N]` | `Y[order[j], j]` — sorted values, contiguous, for cursor/threshold reads |

- Memory: 3 × 4·N·D bytes (~25 GB at NQ scale 2.7M × 768); all three are
  mmap-loaded at search time so only touched pages are resident.
- Persist layout mirrors simdq:
  `<persist_directory>/fagin_data/<index_name>/` containing `meta.json`,
  `Y.npy`, `order.bin`, `vals.bin`, and the `engine_metadata.json`
  id-map/metadata sidecar.

### Native kernel (in `_simdq_native`)

- `_native/include/fagin_ta.h`, included from `bindings/module.c`.
- Generic-D dot product (AVX2 + remainder loop, `#pragma omp simd` fallback);
  **no fixed-D whitelist** — TA has no packed-code layout constraint.
- One binding:

  ```
  fagin_search(Y, order, vals, q, K, batch, epsilon, max_depth, num_threads)
      -> (scores: fp32[K], indices: int64[K], stats: dict)
  ```

  Validates C-contiguity/dtypes, releases the GIL, honors `num_threads` via
  OpenMP — same conventions as `scan_b*`.
- Per-query scan state: `seen` bitmap (N bits, atomic test-and-set for
  cross-thread dedupe), top-k min-heap from `simdq_topk.h`, per-dimension
  cursors advancing in lock-step rounds of `batch` (=`B`) rows.
- Round structure: gather up to `active_dims × B` candidate ids → dedupe via
  bitmap → OpenMP parallel-for over unique candidates' dot products (the
  memory-bound part) → serial heap pushes (K is small) → recompute `T` →
  halting check.
- `stats`: depth reached, sorted accesses, random accesses (candidates
  scored), rounds — benchmarks report access counts, not just wall-clock.
- When `K > N`, the kernel returns N valid entries and pads the remainder
  with index −1 (same convention as the simdq scans, which the engine's
  result loop already skips).

### Engine (`FaginThresholdEngine`)

- Files: `docuverse/engines/retrieval/fagin/{__init__.py, fagin_index.py,
  fagin_engine.py}`; `FaginThresholdEngine(RetrievalEngine)`.
- Mirrors the `SimdqEngine` flow verbatim (copied-and-adapted, matching how
  the codebase treats backends — no shared-base refactor): lazy
  `DenseEmbeddingFunction` load, batched corpus encode at ingest,
  build+save in one shot, `_ensure_loaded` mmap load on first query,
  `precompute_query_embeddings`, and `search_all` (one batched GPU encode,
  then a thread pool of GIL-releasing scans).
- `SUBDIR = "fagin_data"`.

### Config (`RetrievalArguments`, HF-style `field(metadata={"help": ...})`)

| Field | Default | Meaning |
|---|---|---|
| `fagin_batch_rows` | 64 | rows per dimension per round (B) |
| `fagin_epsilon` | 0.0 | additive halting slack; 0 = exact |
| `fagin_max_depth` | 0 | depth cap in rows; 0 = unlimited |
| `fagin_num_threads` | -1 | OpenMP threads for the scan; -1 = all cores |

### Registration

`docuverse/utils/retrievers.py`: accept `fagin`, `fagin-threshold`,
`fagin_threshold`; lazy import with the same ImportError message pattern as
simdq (extension not built → `pip install -e .`); add to the supported-engines
error string.

## Error handling

- Binding raises `ValueError` on wrong dtype/contiguity/shape mismatches
  (Y vs order vs vals vs q), and on `K < 1` or `batch < 1`.
- Engine raises `RuntimeError` on encoder-dim mismatch and empty ingest,
  same as simdq.
- Missing native extension surfaces at engine import through the
  `retrievers.py` try/except with the build hint.

## Testing

1. **Core invariant:** exact mode (ε=0, no depth cap) matches brute-force
   `np.argsort(Y @ q)` top-k on random data — including negative and zero
   query components, negative document values, ties, `K ≥ N`, and D not a
   multiple of 8.
2. **Knobs:** with ε > 0 the returned kth score is never worse than `T − ε`
   at halt; `max_depth` caps the reported depth; stats are consistent
   (random accesses ≤ N; sorted accesses = active_dims × depth).
3. **Persistence/engine:** save→load round-trip equality; mocked-encoder
   ingest/search smoke test (no live services, existing test style); engine
   dispatch cases for all three names.
4. **Benchmark hookup:** works via `ingest_and_test` by switching
   `db_engine: fagin` in existing comparison configs; per-query stats
   aggregated into the timer/report output.

## Out of scope (YAGNI)

- Dimension-subset sorted access (top-|q_j| lists only) — noted as a possible
  future knob; not in this build.
- Incremental partial-sum scoring (vs full dot product per candidate).
- A separate `_fagin_native` extension.
- NRA / no-random-access variants.
