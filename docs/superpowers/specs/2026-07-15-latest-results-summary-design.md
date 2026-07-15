# Design: `--latest` summary table for `run_retrieval_experiment.py`

**Date:** 2026-07-15
**Status:** approved (design), pending implementation
**Scope:** add a read-only `--latest` reporting action to
`scripts/run_retrieval_experiment.py` that summarizes the newest result per
method type from the persistent SQLite results DB, printed to screen and
optionally written as CSV.

## Goal

`--latest` reads the results database (default `experiments/simdq/results.db`,
overridable with the existing `--results-db`) and prints a compact table with
the latest results for each method type (e.g. Fagin GTASD, FAISS HNSW, Milvus
FLAT, simdq asym b=2 +rescore). For method types that carry an epsilon (Fagin
schedules), it shows a few representative epsilons spanning the speed/accuracy
trade-off rather than every swept value. `--latest-csv PATH` additionally
writes the same table as a CSV file.

`--latest` runs no experiment: it summarizes existing data and exits.

## Behavior

- `--latest` is handled at the very top of `main()`, before `load_config` —
  it needs no models or config, only the DB path. It reads
  `args.results_db` (the `--results-db` value) or falls back to the default
  `DEFAULT_SETTINGS["results_db"]` (`experiments/simdq/results.db`), resolves
  it with `_resolve`, runs the report, and returns.
- Default output: an aligned monospace table to stdout.
- `--latest-csv PATH`: also write the table as CSV to PATH (screen table still
  prints). PATH is resolved with `_resolve`.
- Missing DB file or empty/absent `runs` table → a friendly message to stdout
  ("no results DB at <path>" / "results DB has no rows yet") and a clean exit
  (no traceback).

## "Latest per method type"

Scope decided in brainstorming: **latest per method, globally** (no dataset
filter — kept simple).

1. Read every row from `runs`.
2. Group by the exact `method` string (for Fagin this already encodes the
   epsilon, e.g. `Fagin GTASD (eps=0.01, depth=0)`).
3. Within each `method` group, keep the row with the newest `run_ts`
   (tiebreak: larger DB `id`).
4. Bucket each kept row into a **method type** (see below), then reduce to the
   display rows per the epsilon rules.

Because "latest" is keyed on the method string, if the DB holds multiple
datasets or encoders the newest measurement wins for that method. The output
includes `dataset` and `encoder` columns so this stays transparent. No
`--latest-dataset` filter (out of scope).

## Method type grouping

`_method_type(method, family)`:

- Fagin (`family == "Fagin"`): `f"Fagin {algo}"` where `algo` comes from
  `_parse_fagin_name(method)` (TA / TASD / GTA / GTASD).
- Everything else: `method.split(" (")[0]`. This yields `FAISS HNSW`,
  `FAISS FlatIP`, `Milvus FLAT`, `Milvus HNSW`, `FLAT`, and for simdq the full
  label (`asym b=2, +rescore`, `1-bit ham, no rescore`, etc., which have no
  ` (` suffix so pass through unchanged).

## Rows emitted per method type

- **Non-epsilon types** (everything except Fagin): one display row — the
  latest across all `method` strings that map to this type (so differing HNSW
  params collapse to the freshest one).
- **Fagin (epsilon) types:** representative epsilons only, chosen from the
  distinct epsilon values present for that type in the DB by
  `_select_representative_epsilons(values)`:
  - Always include `0.0` (exact) if present.
  - From the nonzero values, include smallest (low), largest (high), and the
    median (mid).
  - Deduplicate while preserving that order; result is up to 4 epsilons.
  - If a type has ≤4 distinct epsilons total, show all of them.
  Each selected epsilon contributes the latest row for its
  `(type, epsilon)` pair.

## Columns

`dataset · encoder · method_type · epsilon · agree@10 · agree@100 · nDCG@10 ·
ms/q · q/s · run_ts`

- `agree@100` is the stored `agree_at_k` (equals agree@100 when `top_k=100`;
  the header is labeled `agree@K` — see note below).
- `epsilon` renders blank for non-Fagin rows (NULL epsilon), `0` for exact
  Fagin, else the value (`:g` formatting).
- Numeric metric cells render blank when NULL (phase was skipped).
- Header note: to stay honest when `top_k != 100`, the agree@100 column header
  reads `agree@K`. (All current data uses top_k=100.)

Sort order: `method_family`, then `method_type`, then `epsilon` ascending.

## Units (all added to `scripts/run_retrieval_experiment.py`)

- `_method_type(method, family) -> str`
- `_select_representative_epsilons(values: list[float]) -> list[float]`
- `query_latest_rows(db_path) -> list[dict]` — reads DB; returns `[]` if the
  file or `runs` table is absent. Uses `sqlite3.Row` for dict access.
- `build_latest_summary(rows) -> list[dict]` — the pure reducer: latest per
  method, bucket into types, apply epsilon selection, sort. Returns ordered
  display-row dicts (keys = column ids). This is the core tested unit.
- `render_latest_table(display_rows) -> str` — aligned monospace table.
- `write_latest_csv(display_rows, path) -> None` — CSV via stdlib `csv`.
- `run_latest_report(db_path, csv_path) -> None` — orchestrator: query, build,
  print table, optionally write CSV, handle empty/missing gracefully.

`main()` gains, near the top:

```python
if args.latest:
    db = _resolve(args.results_db or DEFAULT_SETTINGS["results_db"])
    run_latest_report(db, _resolve(args.latest_csv) if args.latest_csv else None)
    return
```

New argparse flags (next to `--results-db`):
- `--latest` (store_true) — print the latest-per-method summary and exit.
- `--latest-csv PATH` — also write the summary as CSV.

## Error handling

- No DB file → print `no results DB at <path>`, return.
- Table missing or zero rows → print `results DB has no rows yet`, return.
- CSV parent dir auto-created.

## Testing

- `_method_type`: Fagin names → `Fagin GTASD` etc.; `FAISS HNSW (…)` → `FAISS
  HNSW`; simdq labels pass through; `FLAT (fp32 IP, exact)` → `FLAT`.
- `_select_representative_epsilons`: `[0,0.001,0.005,0.01,0.05]` → `[0, 0.001,
  0.005, 0.05]` (exact, low, mid, high); `[0.01]` → `[0.01]`; `[0,0.01]` →
  `[0, 0.01]`; all-distinct ≤4 returns all.
- `build_latest_summary`: synthetic rows across two runs (older + newer) with a
  Fagin epsilon sweep and non-Fagin engines — assert latest wins, one row per
  non-epsilon type, representative epsilons for Fagin, correct sort.
- `run_latest_report` end-to-end: write rows with `write_results_db` into a
  temp DB across two timestamps, capture stdout, assert the newer values
  appear and a `--latest-csv` file is produced with matching rows. Missing-DB
  and empty-DB paths print the friendly message and don't raise.

## Out of scope

- Dataset/encoder filtering (`--latest-dataset`).
- Any change to how rows are written, or to the markdown/JSON/SVG outputs.
- Aggregation across runs (mean/stddev) — this shows single latest rows.
