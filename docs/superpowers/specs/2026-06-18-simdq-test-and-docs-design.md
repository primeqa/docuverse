# `simdq` — comprehensive test plan & practitioner documentation (design)

**Date:** 2026-06-18
**Status:** Design — pending implementation plan
**Branch base:** `v0.2.0`
**Predecessor specs:** [`2026-06-15-simdq-engine-design.md`](2026-06-15-simdq-engine-design.md)
**Plan status:** [`../plans/2026-06-17-simdq-status.md`](../plans/2026-06-17-simdq-status.md) (Plans 1–3 complete)

## 1. Goals & non-goals

### Goals

1. Produce a **5-tier test pyramid** that gates recall, kernel correctness,
   AVX-512 ↔ AVX2 parity, save/load durability, and large-N stress, runnable
   in three latency bands:
   - **fast lane** (<30s, every `pytest`),
   - **slow lane** (<2min, nightly + on-demand),
   - **bench lane** (~5min, manual / release gate).

2. Produce **Sphinx-integrated documentation** under `docs/simdq/` covering
   quickstart, parameter reference, tuning binary embeddings, adapting to a
   new dataset, and troubleshooting — layered so a newcomer can copy-paste
   and a granite-team researcher can dial in recipes.

3. Lock a **SciFact recall regression baseline** (NDCG@10 + Recall@100 per
   recipe R0–R5) into CI so quality regressions surface on PR rather than
   in user data.

### Non-goals

- New kernel work, NQ-batched scan, AVX-512 VNNI, or learned projection.
  Those are deferred to v2 / Plan 3.5 per the existing spec.
- Replacing or rewriting the whitepaper. The tuning doc *cites*
  `docs/whitepaper_simdq_bit_hashing.md` for proofs; it does not duplicate
  them.
- AVX-512 *throughput* numbers. The dev host lacks AVX-512F, so the AVX-512
  inline kernel paths are correctness-tested but not measured. The
  parity test (T2) asserts the AVX-512 path matches AVX2 bit-identically
  *when both can run*; throughput rows in `BENCHMARKS.md` stay marked
  "to be measured."
- Re-running R0–R6 on multiple corpora. SciFact is the regression gate;
  FiQA / multi-corpus runs remain ad-hoc per the existing
  `bench_simdq_beir.py`.

## 2. Test pyramid

Five tiers, each with a clear contract and a runtime budget:

| Tier | Where | What it proves | Runtime | When it runs |
|---|---|---|---|---|
| **T1 Kernel** | `_native/tests/*.c` (existing 16 + new) | Each `(family, b, d, simd_path)` kernel matches a scalar reference | ~45s | every `cmake --build` |
| **T2 Cross-SIMD parity** *(new)* | `_native/tests/test_simd_parity.c` | AVX-512 and AVX2 paths produce **bit-identical** top-K on the same input | ~10s | every build |
| **T3 Python unit** | `tests/test_simdq_*.py` (existing 64 + ~20 new) | API contracts: round-trip, dispatch, mode parity, error paths, edge cases (K=1, N=1, empty corpus, K_prime>256) | <10s | every `pytest` |
| **T4 Property-based fuzz** *(new)* | `tests/test_simdq_fuzz.py` (Hypothesis) | Random `(N, D, d, b, family, projection)` combos: index round-trips, scan returns valid indices, scores monotone vs scalar reference | ~20s | every `pytest` |
| **T5 Recall regression** *(new)* | `tests/test_simdq_recall.py` (`@pytest.mark.slow`) | NDCG@10 on SciFact for each recipe R0–R5 within ±0.005 of locked baseline | ~90s | nightly / `pytest -m slow` |
| **T6 Stress / large-N** *(new)* | `tests/test_simdq_stress.py` (`@pytest.mark.bench`) | 50M synthetic vectors: ingest peak RAM, scan throughput floor, save/load durability under SIGKILL mid-build | ~5min | manual / `pytest -m bench` |

Three pytest markers added to `pyproject.toml`: `slow`, `bench`, plus the
existing default. Default `pytest tests/` runs T3 + T4 only — keeps the
fast lane under 30s.

### T2 — Cross-SIMD parity

Build twice with `-march=native` and `-mavx2 -mfma -mpopcnt -mno-avx512f`.

Assertion rules (handles FP non-associativity in the AVX-512 vs AVX2
reduction order):

- **Hamming family** — XOR-popcount is integer; assert top-K indices
  *and* distances are **exactly equal** across paths.
- **Asymmetric family** — score arrays differ by `≤ 1e-6` element-wise;
  indices match wherever the score gap to the next-best candidate
  exceeds `1e-5`. Mismatches with smaller gaps are logged but not
  asserted (legitimate tie-break divergence). Test fails if any *index*
  swap produces a score regression beyond `1e-6`.

Coverage: every `(family, b, d) ∈ {hamming, asym}×{1,2,4}×{384, 768, 1024,
1536}` with `d ∈ {D, D/2}`. Same RNG seed across paths.

CTest targets:

- `simd_parity_native`  — `-march=native` (AVX-512 if available)
- `simd_parity_avx2`    — `-mavx2 -mfma -mpopcnt -mno-avx512f`
- `simd_parity_dlopen`  — single binary `dlopen`s both `.so` files and
  diffs results in-process. Skipped at runtime if the host lacks AVX-512F
  (status: SKIPPED, not FAILED).

### T3 — Python unit gap-fill

Added to existing `tests/test_simdq_*.py`:

- **`test_simdq_index.py`**: empty corpus error path, N=1 corpus, K=1,
  K_prime=K vs K_prime=256, save → load → save round-trip metadata
  stability, corrupted `meta.json` rejected with a clear error,
  `format_version` mismatch rejected (test writes a synthetic
  `meta.json` with `format_version: 999` to a temp dir, asserts
  `SimdqIndex.load()` raises `ValueError` with a message naming both
  versions; only v1 ships today, the path exists for the next bump).
- **`test_simdq_engine.py`**: factory dispatch with all 6 recipe configs
  (R0–R5), `delete_index` + re-ingest, `has_index` correctness, missing
  encoder model raises a clean `RuntimeError` (not `AttributeError`).
- **`test_simdq_quantization.py`**: scale-fitting on heavy-tailed inputs
  (Cauchy-distributed), b=4 saturation behavior at extreme magnitudes,
  all-zero vector handled (no NaN scales).
- **`test_simdq_projection.py`**: random_orthogonal idempotence
  (`W @ W.T ≈ I_d` to `1e-5`), seed determinism across processes (run
  twice via `subprocess`, assert byte-identical W).

### T4 — Property-based fuzz (Hypothesis)

Strategy:

```python
@st.composite
def simdq_config(draw):
    D = draw(st.sampled_from([384, 768, 1024, 1536]))
    d = draw(st.sampled_from([D, D // 2]))
    family = draw(st.sampled_from(["asymmetric", "hamming"]))
    b = draw(st.sampled_from([1, 2, 4])) if family == "asymmetric" else None
    projection = "identity" if d == D else "random_orthogonal"
    N = draw(st.integers(min_value=1, max_value=1000))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    return dict(D=D, d=d, family=family, b=b, projection=projection,
                N=N, seed=seed)
```

Properties:

- Build never crashes for any config drawn from the strategy.
- Returned indices are in `[0, N)`.
- No duplicate indices in top-K.
- Codes-only score ranking matches `np.argsort` of a scalar reference
  within Spearman `ρ ≥ 0.9` for `N ≥ 100` (looser for small N where ties
  dominate).
- Save → load → search returns identical results to in-memory search.

Settings: 200 examples per CI run, `derandomize=True` in CI for
reproducibility. Hypothesis example database checked in under
`tests/.hypothesis/` so failing seeds shrink across runs.

### T5 — Recall regression on SciFact

Fixture: SciFact corpus + queries + qrels, downloaded from BEIR's HF
mirror to `~/.cache/docuverse/beir/scifact` on first run; cached
afterwards.

Encoder: `ibm-granite/granite-embedding-311m-multilingual-r2` (D=768).
Pinned via `pytest.importorskip("sentence_transformers")` and a
module-level model-name constant.

Baseline file: `tests/fixtures/simdq_scifact_baseline.json`. Schema
(values populated by the first `--update-baseline` run; numbers below
are *illustrative shape only*, not measured):

```json
{
  "encoder": "ibm-granite/granite-embedding-311m-multilingual-r2",
  "scifact_revision": "<HF dataset SHA from datasets.load_dataset>",
  "captured_at": "<ISO-8601, set on baseline update>",
  "captured_on": "<host CPU model, AVX-512 yes/no>",
  "recipes": {
    "R0": {"ndcg10": null, "recall100": null},
    "R1": {"ndcg10": null, "recall100": null},
    "R2": {"ndcg10": null, "recall100": null},
    "R3": {"ndcg10": null, "recall100": null},
    "R4": {"ndcg10": null, "recall100": null},
    "R5": {"ndcg10": null, "recall100": null}
  }
}
```

The first run with `--update-baseline` populates every `null`. Until
that run lands, T5 is `pytest.skip`ped with a message naming the
required step.

Tolerance:

- `|ndcg10 - baseline.ndcg10| ≤ 0.005`
- `|recall100 - baseline.recall100| ≤ 0.01`

Failures dump full per-query results to `_artifacts/scifact_<recipe>.json`
for diff inspection.

Baseline update workflow: `pytest tests/test_simdq_recall.py
--update-baseline`. Eyeball the diff, commit the updated JSON with a
recipe-aware message naming the kernel/encoder change that justified the
move.

### T6 — Stress / large-N

50M synthetic Gaussians, D=768, b=2:

- **RAM ceiling.** Assert ingest peak RSS `< 4·N·D·1.2` bytes (the 1.2
  slack covers numpy temporaries and the index codes). Measured via
  `resource.getrusage(RUSAGE_SELF).ru_maxrss`.
- **Throughput floor.** Scan throughput `≥ 20M cmp/s/thread` on AVX2.
- **Crash recovery.** Spawn `multiprocessing.Process` that calls
  `index.save(path)`; from the parent, `os.kill(pid, SIGKILL)` mid-build.
  Assert the partial `<path>.tmp/` directory exists but `<path>/` does
  not — the atomic-rename guarantee. Verifies the spec §7 "build to .tmp,
  fsync, rename" pattern.
- **Concurrent searchers.** 8 Python threads doing `index.search()`
  against a single loaded index. No segfaults; results identical to
  serial baseline.

## 3. Documentation architecture

```
docs/
├── index.rst                  # MOD: add 'simdq/index' to top-level toctree
└── simdq/                     # NEW dir
    ├── index.rst              # NEW: landing page; routes by audience
    ├── quickstart.rst         # NEW: copy-pasteable YAML + CLI on SciFact
    ├── parameters.rst         # NEW: every simdq_* field, table form
    ├── tuning.rst             # NEW: recipe decision tree + per-parameter math
    ├── adapting.rst           # NEW: new corpus / new encoder / unsupported D
    └── troubleshooting.rst    # NEW: errors + perf debug + recall debug
```

### Routing on the landing page

```
If you want to ...                          → read
─────────────────────────────────────────────────────
... try simdq in 5 minutes                  quickstart.rst
... look up what `simdq_b` means            parameters.rst
... pick a recipe for your corpus           tuning.rst (Recipe matrix)
... understand WHY b=2 vs 1+rescore         tuning.rst (Math intuition)
... ingest your own dataset                  adapting.rst
... run the R0–R6 sweep                      adapting.rst (Recipe sweep)
... debug a crash or low recall              troubleshooting.rst
```

Sphinx wiring: `docs/conf.py` already exists; we add the `simdq/` toctree
entry to `docs/index.rst` and use plain `.rst` (matching `cli.rst` /
`presets.rst`) so no `myst-parser` dependency is added.

## 4. Documentation content per page

### `quickstart.rst` — ~80 lines, one-screen-readable

- One-liner: "simdq is an in-process binary-quantized retrieval engine
  for ≤10M-vector BEIR-scale evaluation."
- Install: `pip install -e ".[simdq]"`. Verify with
  `python -c "from docuverse.engines.retrieval.simdq import SimdqIndex"`.
- 30-line YAML for SciFact + b=2 + granite-278m.
- CLI: `python -m docuverse.utils.ingest_and_test --config
  beir_scifact_simdq.yaml --actions "ire"`.
- Expected output line and the on-disk path to the resulting index.

### `parameters.rst` — full parameter reference, table-driven

One row per `simdq_*` field on `RetrievalArguments`:

| Field | Type | Default | Valid values | Effect | When to change |
|---|---|---|---|---|---|
| `simdq_family` | str | `"asymmetric"` | `asymmetric` \| `hamming` | Selects scan kernel family | Switch to `hamming` for 1-bit Hamming + rescore (R0) |
| `simdq_b` | int | 2 | 1, 2, 4 | Bits/dim in asymmetric codes | b=1 → 1-byte/8 dims; b=2 → 1-byte/4 dims; b=4 → 1-byte/2 dims |
| `simdq_d` | int? | None | None, D, D/2 | Reduced dim; None ⇒ D | Halve to shrink index 2× at the cost of recall |
| `simdq_projection` | str | `"identity"` | `identity` \| `random_orthogonal` | Projection W (d×D) | `random_orthogonal` is required for d=D/2 |
| `simdq_projection_seed` | int | 42 | any int | RNG seed for the projection | Pin for reproducibility across hosts |
| `simdq_store_floats` | bool | `True` | True / False | Whether `floats.bin` is written for rescore | False saves disk; disables two-stage rescore |
| `simdq_rescore_alpha` | int | 10 | ≥ 1 | K' = α·K candidates rescored | 1 ⇒ codes-only; 10 closes most of the recall gap |
| `simdq_num_threads` | int | 0 | 0…N | OMP threads for scan; 0 = OMP default | Tune to `#physical cores` for memory-bound asym scans |

Cross-cutting (already documented elsewhere; brief pointer here):
`top_k`, `index_name`, `project_dir`, `model_name`, `bulk_batch`.

### `tuning.rst` — two halves

**Half 1 — Recipe decision tree** (cookbook, no math)

A flowchart-style decision tree:

```
Variable-norm corpus (e.g. heterogeneous text length)?
    yes → use store_floats=True + rescore_alpha=10 (R0 or R2)
    no  → codes-only is fine

Disk constrained?
    yes → b=1 + rescore_alpha=10  (R2: small codes, fp16 floats)
    no  → b=2 no rescore           (R3: larger codes, no floats)

Latency-tight, RAM-cheap?
    → b=2, store_floats=False, num_threads=#physical cores  (R3)

Want to halve the index?
    → b=2 d=D/2 random_orthogonal  (R4) — same recall as R3 in many cases
    → b=4 d=D/2 random_orthogonal  (R5) — same byte budget as R3, more bits
```

A table mapping R0–R5 to typical use-cases with example corpora.

**Half 2 — Per-parameter math intuition** (the "why", with whitepaper cites)

- **`b` (bits per dim).** "1-bit gives binary Hamming; 2-bit gives ±1,±3
  levels; 4-bit gives 16 levels. The asymmetric estimator's variance
  shrinks like 1/2^(2b) — so b=2 typically halves the rescoring need vs
  b=1." → cite whitepaper §4.
- **`d` (reduced dim).** "JL lemma: random orthogonal projection
  preserves cosine within ε with high probability for `d ≳ log(N)/ε²`.
  At N=10M, d=384 gives ε≈0.05." → cite whitepaper §3.
- **`projection`.** Identity is free but rejects `d<D`; random_orthogonal
  costs an N×d×D matmul at ingest for d=D/2.
- **`store_floats` + `rescore_alpha`.** "Two-stage rescore costs +2·N·d
  bytes on disk and +α·d FMAs per query. Rule of thumb: α=10 closes 80%
  of the recall gap to the float baseline."
- **`num_threads`.** OMP threading; sweet spot is `#physical cores` for
  memory-bound asymmetric scans.

Closes with a "what to measure when tuning" checklist: NDCG@10,
Recall@100, p50/p99 latency, on-disk bytes, peak RSS.

### `adapting.rst` — new dataset / encoder / dim

- **New BEIR-format dataset.** Drop in `passages.jsonl + queries.jsonl +
  qrels.tsv`, set the three paths in YAML, run `bench_simdq_beir.py
  --dataset <name> --passages ... --queries ... --qrels ... --encoder-dim
  768`.
- **New encoder.** Set `model_name`. If D ∉ {384,768,1024,1536}: document
  the two paths — (1) project to nearest supported D in caller code, or
  (2) extend the kernel template (out of scope for v1; pointer to spec
  §6).
- **Custom corpus format.** Point at `data_template` configs in `config/`
  and the `text_header` / `title_header` / `id_header` overrides on
  `RetrievalArguments`.
- **Plugging into the recipe sweep.** How `bench_simdq_beir.py` generates
  per-recipe YAMLs from a base template; how to add a new recipe
  (R7 etc.); how to subset (`--recipes R0 R3`).

### `troubleshooting.rst` — errors + perf + recall

- **Errors with copy-paste-able messages**, each followed by cause + fix:
  `simdq build: D must be one of {384,...}`,
  `K_prime > 256`,
  `projection='identity' requires d == D`,
  `format_version mismatch`,
  `simdq ingest: encoder produced dim X but hidden_dim=Y`.
- **Perf debug.** `OMP_NUM_THREADS` ignored? — check `simdq_num_threads`.
  mmap thrash? — check `floats.bin` size vs RAM. AVX-512 not used? —
  `cat /proc/cpuinfo | grep avx512f` and rebuild.
- **Recall debug decision tree.** "low NDCG@10 → first try
  `rescore_alpha=10`; still low → switch from b=1 to b=2; still low →
  drop `random_orthogonal`, use identity at d=D."

## 5. Module / file layout

```
docs/
├── index.rst                                  # MOD: toctree entry for simdq/
└── simdq/                                     # NEW dir
    ├── index.rst                              # NEW
    ├── quickstart.rst                         # NEW
    ├── parameters.rst                         # NEW
    ├── tuning.rst                             # NEW
    ├── adapting.rst                           # NEW
    └── troubleshooting.rst                    # NEW

docuverse/engines/retrieval/simdq/
└── TESTING.md                                 # NEW: pyramid overview, runbook

tests/
├── test_simdq_index.py                        # MOD: gap-fill cases (T3)
├── test_simdq_engine.py                       # MOD: factory + dispatch (T3)
├── test_simdq_quantization.py                 # MOD: heavy-tail, all-zero, b=4 sat (T3)
├── test_simdq_projection.py                   # MOD: idempotence, cross-process seed (T3)
├── test_simdq_fuzz.py                         # NEW: Hypothesis (T4)
├── test_simdq_recall.py                       # NEW: SciFact baseline (T5, slow)
├── test_simdq_stress.py                       # NEW: 50M synthetic + crash (T6, bench)
└── fixtures/
    └── simdq_scifact_baseline.json            # NEW: locked NDCG/recall per recipe

docuverse/engines/retrieval/simdq/_native/tests/
└── test_simd_parity.c                         # NEW: AVX-512 ↔ AVX2 (T2)

docuverse/engines/retrieval/simdq/_native/CMakeLists.txt   # MOD: 3 new ctest targets
                                                           # (simd_parity_native,
                                                           #  simd_parity_avx2,
                                                           #  simd_parity_dlopen)

pyproject.toml                                 # MOD: register pytest markers
                                                           # (slow, bench);
                                                           # add 'hypothesis' to dev deps
```

**~14 new files, ~7 modified files.** Net adds: 1 new doc dir (6 .rst),
1 testing runbook, 3 new pytests, 1 new C test, 3 new ctest targets.

## 6. CI / runtime tiers

### Fast lane (default)

- Triggered by `pytest tests/` and every `cmake --build`.
- Runs T1 ctest (~45s) + T2 parity (~10s) + T3 unit (<10s) + T4 fuzz
  (~20s).
- **Total ~85s including build.** Per-PR gate.

### Slow lane

- Triggered by `pytest tests/ -m slow` (or `--slow`).
- Runs all of fast + T5 SciFact recall (~90s including encoder warmup;
  ~30s if encoder cached).
- **Total ~3min.** Nightly cron, plus on-demand for kernel-touching PRs.

### Bench lane

- Triggered by `pytest tests/ -m bench`.
- Runs T6 stress only (~5min).
- **Manual; release gate before tagging.**

### Hypothesis seed strategy

`derandomize=True` in CI for reproducibility; example database checked in
under `tests/.hypothesis/` so failing seeds shrink across runs.

### Baseline update workflow

When a kernel/encoder change legitimately moves recall:

```bash
pytest tests/test_simdq_recall.py --update-baseline
git diff tests/fixtures/simdq_scifact_baseline.json   # eyeball
git commit tests/fixtures/simdq_scifact_baseline.json -m "..."
```

## 7. Acceptance criteria

When this work is done:

1. `pytest tests/` exits 0 in <30s on the dev box, covering T3 + T4.
2. `cmake --build _native/build && ctest -j` exits 0 with **19 ctest
   targets** (16 existing + 3 new parity).
3. `pytest tests/ -m slow` exits 0 in <2min on a warm cache, locking
   SciFact NDCG@10 within ±0.005 of baseline for R0–R5.
4. `pytest tests/ -m bench` runs to completion (no segfault, no OOM)
   with 50M vectors on a 64GB host.
5. `cd docs && make html` builds with **zero warnings**; the new
   `simdq/` toctree renders.
6. A practitioner cloning the repo can: read `quickstart.rst` → run
   `bench_simdq_beir.py` on SciFact → look up any `simdq_*` parameter in
   `parameters.rst` → understand *why* to pick R3 vs R0 from
   `tuning.rst`. End-to-end without asking the granite team.

## 8. Open questions for the implementation plan

These are pinned now, called out so the implementation plan can resolve
them with the smallest concrete decisions:

- **SciFact download path.** `~/.cache/docuverse/beir/scifact` vs
  `tests/fixtures/scifact/` (checked in). Recommend the cache path —
  data is large, license requires attribution per access.
- **Hypothesis dev-dep gating.** Add to `[project.optional-dependencies]
  test` so `pip install -e ".[test]"` pulls it; default install stays
  lean.
- **CTest `simd_parity_dlopen` linker layout.** The two `.so` files need
  distinct symbol prefixes so a single binary can `dlopen` both. Either
  link-time `-Wl,--wrap` or compile each path into a separate
  translation unit with a path-prefixed namespace. Implementation plan
  picks one based on what the existing `_native/CMakeLists.txt` already
  does for `kernels_native` vs `kernels_avx2`.
- **Sphinx build in CI.** Currently no `make html` step in any CI
  config. Add as a separate workflow, or a step in the existing pytest
  workflow? Recommend separate so doc-only PRs don't run the full
  pytest suite.
