# `simdq` test pyramid runbook

Five tiers, three latency lanes. Run the right one for the change you made.

## Lanes

| Lane | Command | Runtime | Includes |
|---|---|---|---|
| **fast** | `pytest tests/` | <1 min | T3 unit, T4 fuzz |
| **slow** | `pytest tests/ -m slow` | <2 min | adds T5 SciFact recall regression |
| **bench** | `pytest tests/ -m bench` | ~5-10 min | adds T6 50M (or 10M) stress |
| **kernel** | `cd _native/build && ctest` | ~1 sec | T1 ctest + T2 cross-SIMD parity |

Picking a lane:

- **Kernel-touching change** (anything under `_native/`) — run **kernel + slow** before the PR.
- **Python-only change** — run **fast** locally; **slow** runs in CI.
- **Release tag** — run **kernel + slow + bench** end-to-end.

The fast lane is gated by `addopts = -m 'not slow and not bench'` in
`pyproject.toml`, so the slow and bench lanes need explicit opt-in.

## Updating the SciFact baseline

`tests/fixtures/simdq_scifact_baseline.json` ships with `null` recipe values
and `test_simdq_recall.py` `pytest.skip`s until the JSON is populated. To
populate (or refresh after a kernel/encoder change that legitimately moves
NDCG@10 or Recall@100):

```bash
SIMDQ_UPDATE_BASELINE=1 pytest tests/test_simdq_recall.py -m slow
git diff tests/fixtures/simdq_scifact_baseline.json
git commit tests/fixtures/simdq_scifact_baseline.json -m "Update SciFact baseline: <one-line reason>"
```

The commit message must name the kernel/encoder change that justified the
move. The baseline JSON is the single source of truth for the recall gate.

`SIMDQ_UPDATE_BASELINE=1` is an environment variable, not a CLI flag —
the test file lives in `tests/` and there's no `tests/conftest.py` to
register a custom `--update-baseline` argument.

## Cross-SIMD parity (T2)

The same C source (`_native/tests/test_simd_parity.c`) is compiled twice:
once with `-march=native` (selects AVX-512 on capable hardware) and once
with `-mavx2 -mfma -mpopcnt -mno-avx512f` (forces the AVX2 fallback).
Both binaries write top-K results for a deterministic fixed-seed input
to `simd_parity_<arg>.out`; CMake's `simd_parity_dlopen` test
byte-diffs the two via `cmake -E compare_files`.

If `simd_parity_dlopen` fails on a host with AVX-512:

1. Inspect the diff:
   ```bash
   xxd build/simd_parity_native.out | diff -u - <(xxd build/simd_parity_avx2.out) | head
   ```
2. The hamming portion (bytes 0..799, 50 records of `int64 dist + int64 idx`)
   MUST be byte-identical — that's integer-only popcount, no FP rounding.
   **If the diff is in the hamming bytes, it's a real correctness divergence.**
3. The asym portion (bytes 800..) holds 50 * 12-byte records each
   (`uint32 score-bit-pattern + int64 idx`), three sections (b=1, b=2, b=4).
   AVX-512 vs AVX-2 reduce in different orders; low-bit FP differences here
   are not a bug. If the diff is asym-only, replace the strict `compare_files`
   target with a tolerant Python helper (see CMakeLists.txt comment in
   the T2 block).

## Hypothesis seeds

T4 (`tests/test_simdq_fuzz.py`) uses `derandomize=True` so failing seeds
reproduce on demand. If you hit a fuzz failure locally, save the shrunk
example for the follow-up bug — Hypothesis will print it under
"Falsifying example".

## SciFact corpus cache

The first run of `test_simdq_recall.py` downloads the BEIR/SciFact corpus
(~5,200 passages, ~300 queries) to
`~/.cache/huggingface/datasets/BeIR___scifact`. Pre-download in CI by
running:

```bash
python -c "from datasets import load_dataset; load_dataset('BeIR/scifact', 'corpus'); load_dataset('BeIR/scifact', 'queries'); load_dataset('BeIR/scifact-qrels')"
```

The recall test also needs `datasets` installed (not in the project's
default dep set):

```bash
pip install datasets
```

## Test inventory

| File | Tests | Lane |
|---|---|---|
| `tests/test_simdq_index.py` | round-trip, mode parity, edge cases (T3 gap-fill: empty corpus xfail, N=1, K=1, K_prime extremes, save/load/save metadata stability, corrupted/version-mismatched meta.json) | fast |
| `tests/test_simdq_engine.py` | dispatch, round-trip with real encoder, T3 gap-fill: factory dispatch over R0-R3, delete+reingest, missing-encoder error | fast |
| `tests/test_simdq_quantization.py` | pack round-trips at b=1/2/4, hamming round-trip, T3 gap-fill: Cauchy inputs, all-zero rows, b=4 saturation | fast |
| `tests/test_simdq_projection.py` | identity, random_orthogonal shape/seed, multi-d, T3 gap-fill: idempotence, cross-process determinism | fast |
| `tests/test_simdq_fuzz.py` | T4 Hypothesis: build/search invariants, save/load score-equality, codes-only ranking uplift | fast |
| `tests/test_simdq_recall.py` | T5 SciFact recall regression (skips on null baseline) | slow |
| `tests/test_simdq_stress.py` | T6 RAM ceiling, throughput floor, SIGKILL atomicity, concurrent searchers | bench |
| `_native/tests/test_simd_parity.c` | T2 cross-SIMD parity (hamming + asym b=1/2/4) | kernel |

## Failure-triage cheatsheet

- **`pytest tests/` fails with import error in `test_simdq_recall.py`** —
  `datasets` package isn't installed. Either `pip install datasets` (for
  CI machines) or accept the module-skip (for hosts where SciFact won't
  run anyway).
- **`pytest -m slow` says "SciFact baseline contains null values"** —
  the baseline hasn't been populated yet. Run `SIMDQ_UPDATE_BASELINE=1`
  once on a clean host (see "Updating the SciFact baseline" above).
- **`ctest` shows `simd_parity_dlopen` failing on a non-AVX-512 host** —
  shouldn't happen; both binaries should pick the AVX2 path. Investigate
  the diff (see "Cross-SIMD parity" above).
- **`pytest -m bench` OOMs** — `SIMDQ_STRESS_N=10000000` is the default;
  drop to `1000000` if 30GB fp32 doesn't fit. The throughput floor of 10M
  cmp/s/thread is independent of N within reasonable ranges.
- **`pytest tests/test_simdq_index.py::test_empty_corpus_rejected` is XFAIL** —
  expected. `SimdqIndex.build` currently accepts `(0, D)` arrays silently
  and the later `search` crashes with `IndexError`. Build should validate
  `N >= 1` up front; the xfail captures the follow-up.
