# Design: make simdq thread-scaling sweeps optional (default off)

**Date:** 2026-07-15
**Status:** approved (design), pending implementation
**Scope:** gate the simdq thread-scaling sweeps in
`scripts/run_retrieval_experiment.py` behind a new opt-in toggle so they no
longer run by default.

## Goal

The thread-scaling sweeps are the slowest part of a speed run: three full
sweeps of `SimdqIndex.search` across thread counts `[1, 2, 4, 8, 16, all]` —
Phase 1 (asym b=2, default BLAS), Phase 1 (asym b=2, `OPENBLAS_NUM_THREADS=1`),
and Phase 6/8 (Hamming SoA) — producing the `.sweeps.svg` chart. Most runs
don't need them. Make them **opt-in, default off**, without touching the
Phase 5b head-to-head, quality, or the frontier chart.

## What runs today

In `main()` (currently lines 2259–2273):

```python
if st["run_speed"]:
    if st.get("run_simdq", True):
        common = dict(threads=st["threads"], ...)
        p1_default = run_sweep_phase("Phase 1 — asym b=2", "asymmetric", 2, ...,
                                     blas_threads=None, **common)
        p1_blas1   = run_sweep_phase("Phase 1 — asym b=2", "asymmetric", 2, ...,
                                     blas_threads=1, **common)
        p68        = run_sweep_phase("Phase 6/8 — Hamming SoA", "hamming", None,
                                     ..., blas_threads=None, **common)
    if (st["run_milvus"] or st["run_faiss"] or ...):
        head = bench_head_to_head(...)
```

`p1_default`, `p1_blas1`, `p68` are initialized to `None` before this block.

## Change

### 1. New setting (default off)

Add to `DEFAULT_SETTINGS`, next to `run_simdq`:

```python
"run_thread_sweeps": False,   # Phase 1 / Phase 6-8 thread-scaling sweeps
                              # (+ .sweeps.svg); opt-in via --thread-sweeps
```

### 2. New CLI flag

Add `--thread-sweeps` (`action="store_true"`) in `main()`'s argparse block,
near the other run toggles. It is applied through `_cli_overrides`:

```python
if args.thread_sweeps:
    ov["settings.run_thread_sweeps"] = True
```

Because the override is only added when the flag is passed, a YAML config can
still enable sweeps on its own (`settings.run_thread_sweeps: true`) and the
flag never forces them off — consistent with how `--no-milvus` etc. work
(flag sets a value; absence leaves YAML/default in place). Note this flag is
enable-only; there is no `--no-thread-sweeps` because the default is already
off (matching the intent "default no").

### 3. Gate the sweep block

Change the inner condition from:

```python
    if st.get("run_simdq", True):
```

to:

```python
    if st.get("run_simdq", True) and st.get("run_thread_sweeps", False):
```

The sweeps run only when simdq is enabled **and** sweeps are explicitly
requested. Everything else in the speed phase (Phase 5b head-to-head incl.
simdq rows) is unchanged.

## Downstream — no report changes needed

When the sweeps are skipped, `p1_default` / `p1_blas1` / `p68` remain `None`:

- `write_report` already guards the whole "Thread scaling" section and the
  `.sweeps.svg` write with `if p1_default is not None:`, so both are omitted
  cleanly.
- The `.json` sidecar already coalesces each with `... or []`, so it records
  empty sweep lists rather than erroring.

No changes to `write_report`, the chart code, or the JSON block are required.

## Docstring / comments

Update the module docstring's Phase 4 bullet (which currently states the speed
phase runs Phase 1 and Phase 6/8) and the inline comments to note the sweeps
are opt-in via `--thread-sweeps` (default off). Keep the description of what
the sweeps do; only add that they no longer run by default.

## Testing

`main()` has no unit tests (needs models + a full run), so cover the two
testable seams:

1. **Default value** — assert `DEFAULT_SETTINGS["run_thread_sweeps"] is False`.
2. **CLI override mapping** — build an `argparse.Namespace` (or invoke the
   arg parser) and assert:
   - with `--thread-sweeps`: `_cli_overrides(args)["settings.run_thread_sweeps"]
     is True`.
   - without it: `"settings.run_thread_sweeps"` is **absent** from the
     `_cli_overrides` dict (so YAML/default wins).
3. **Help smoke check** — `python scripts/run_retrieval_experiment.py --help`
   lists `--thread-sweeps` and exits 0.

The `_cli_overrides` tests require constructing a Namespace with the attributes
the function reads. The test will build one with `argparse.Namespace(...)`
setting `thread_sweeps` plus the other attributes `_cli_overrides` accesses
(all the existing flag/value attrs), or more robustly parse an argv via the
script's own parser. The implementation plan will pick the least brittle of
these.

## Out of scope

- The Phase 5b head-to-head simdq rows (still governed by `run_simdq`).
- The existing `--no-simdq` / `--skip-speed` flags (unchanged).
- Any change to sweep internals (`run_sweep_phase`) or the chart rendering.
