# Design: persistent SQLite results store for `run_retrieval_experiment.py`

**Date:** 2026-07-15
**Status:** approved (design), pending implementation
**Scope:** add a persistent, common results database to
`scripts/run_retrieval_experiment.py`, in addition to the existing per-run
Markdown report + `.json` sidecar.

## Goal

Every invocation of the experiment runner should append its results to a
single, persistent SQLite database so runs accumulate over time in one
queryable place. For each method/engine measured in a run, record:

- date/time the run happened,
- method type,
- epsilon (for Fagin methods),
- agree@10, agree@100,
- nDCG@10,
- runtime and other timing.

The existing Markdown + `.json` outputs are unchanged; the DB is additive.

## Storage

- **Backend:** stdlib `sqlite3` (no new dependency; consistent with the
  CLAUDE.md gotcha that optional backends use lazy imports — here we avoid a
  dependency entirely).
- **Default path:** `experiments/simdq/results.db`.
- **Override:** new CLI flag `--results-db PATH` and YAML key
  `settings.results_db`. Precedence follows the existing pattern: CLI wins,
  then YAML, then the default. Wired through `_CLI_SETTINGS_MAP` /
  `DEFAULT_SETTINGS` like the other scalar settings.
- **Self-initializing:** `CREATE TABLE IF NOT EXISTS` runs on every
  invocation, so a fresh box just works. Append-only; no migrations planned.
- Path is resolved with the existing `_resolve()` helper (repo-relative unless
  absolute); parent directory is created if needed.

## Schema

Single table `runs`, one row per (run × model × method):

```sql
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,   -- uuid4 hex, one per invocation
    run_ts            TEXT NOT NULL,   -- ISO-8601 local datetime, seconds precision
    hostname          TEXT,
    dataset           TEXT,
    top_k             INTEGER,
    workers           INTEGER,
    encoder           TEXT,            -- model tag
    method            TEXT NOT NULL,   -- full engine/variant name
    method_family     TEXT,            -- simdq / FAISS / Milvus / Fagin / FLAT
    epsilon           REAL,            -- Fagin only; NULL otherwise
    -- quality (NULL when --skip-quality)
    ndcg_at_10        REAL,
    recall_at_10      REAL,
    recall_at_k       REAL,
    mrr_at_10         REAL,
    agree_at_10       REAL,
    agree_at_k        REAL,            -- agree@K; == agree@100 when top_k=100
    n_queries         INTEGER,
    -- speed (NULL when --skip-speed or engine absent from Phase 5b)
    runtime_ms_per_q  REAL,            -- Phase 5b concurrent ms/query
    queries_per_sec   REAL             -- 1000 / runtime_ms_per_q
);
```

Notes:
- `agree_at_k` holds the general `agree@K` value. When `top_k == 100` (the
  config default) this is exactly the requested agree@100. The report already
  treats agree@K this way; we do not add a separate agree@100 column because K
  is configurable and storing the raw agree@K keeps the row honest.
- `run_id` + `run_ts` group all rows from one invocation. `run_id` is a
  `uuid.uuid4().hex` generated once in `main()`. `run_ts` is
  `datetime.now().isoformat(timespec="seconds")` — captures date **and**
  time-of-day. (Note: the script elsewhere uses `date.today()` for report
  filenames; the DB deliberately records full datetime.)
- No index beyond the primary key initially; the table is small and queried
  ad hoc.

## Row grain and the phase join

Quality and speed are computed in separate phases with slightly different
method labels:

- `metric_rows`: `{encoder, variant, recall@10, recall@K, MRR@10, nDCG@10,
  n_queries}`
- `agree_rows`: `{encoder, variant, agree@10, agree@K}`
- `head[].systems`: `{name, ms, agree}` (Phase 5b concurrent ms/query)

A new helper builds one DB row per (model × method) by joining these three,
**reusing the existing name↔variant mapping** that `_frontier_chart_svg`
already relies on:

- `family_of(name)` → method family bucket.
- `quality_of(tag, name)` → the matching `metric_rows` entry for a Phase 5b
  system name (handles the `simdq asym`/`simdq 1-bit` → `asym b=2`/`1-bit ham`
  ± rescore mapping, and the exact-engine → FLAT-ceiling fallback).
- `_parse_fagin_name(name)` → `(algo, epsilon)`; used to populate `epsilon`.

Join strategy (union of methods seen across phases, keyed by (encoder,
method)):

1. Start from Phase 5b systems (they carry runtime); attach quality via
   `quality_of`.
2. Add any quality-only methods that never appeared in Phase 5b (e.g. when
   `--skip-speed`, or the `asym b=2, no rescore` variants that quality reports
   but the frontier maps): these get `runtime_ms_per_q`/`queries_per_sec`
   NULL.
3. `agree_at_10` / `agree_at_k` come from the `agree_rows` entry matched by the
   same (encoder, variant) key that quality uses.

Methods present in only one phase still produce a row with the absent
columns left NULL. This makes the writer robust to `--skip-quality`,
`--skip-speed`, `--no-simdq`, `--no-fagin`, etc.

Because the exact set of "methods" and the label-matching already live in
`_frontier_chart_svg`, the join helper will factor out / reuse that logic
rather than duplicate it, to avoid drift between the chart and the DB.

## Where it hooks in

- `main()` generates `run_id` and `run_ts` near the top (before any phase), so
  all rows share them.
- After the `.json` sidecar is written at the end of `main()`, call:

  ```python
  write_results_db(db_path, cfg, isa, run_id, run_ts,
                   metric_rows, agree_rows, head)
  ```

  wrapped in `try/except` that prints a warning to stderr on failure — a DB
  error must never lose the Markdown/JSON outputs or crash the run.
- `write_results_db` opens the connection, ensures the table, builds the joined
  rows, `executemany` inserts, commits, closes, and prints
  `# results-db -> {path} (+N rows)`.

## Error handling

- DB write failures are caught and logged; the run still succeeds.
- Parent dir auto-created.
- `CREATE TABLE IF NOT EXISTS` tolerates an existing DB.

## Testing

- Unit test the join helper: given small synthetic `metric_rows`,
  `agree_rows`, and `head`, assert the produced rows have the right method
  families, epsilon parsing (Fagin vs non-Fagin), NULLs for skipped phases,
  and correct quality/speed values.
- Unit test `write_results_db` against a temp SQLite file: run twice, assert
  rows accumulate (append-only), schema is created, and columns match.
- Follows the repo's mixed unittest/pytest style; no live services needed.

## Known limitations (accepted)

- **Exact-engine quality is stored as measured, not synthesized.** The frontier
  chart's `quality_of` falls back to the `FLAT (fp32 IP, exact)` nDCG ceiling
  for an exact engine that has no quality row. The DB join deliberately does
  **not** replicate that fallback: if an exact baseline is timed in Phase 5b
  but its quality row is absent (e.g. it raised during `record`), the DB row
  keeps `ndcg_at_10` NULL rather than borrowing FLAT's number. A NULL is the
  honest value for a results store. This means the DB can under-report exact-
  engine quality relative to the chart in that skip case.
- **Unique model tags assumed.** Rows are keyed by (encoder tag, method). If
  two configured models share a `tag`, or two Phase 5b systems map to the same
  variant for one tag, the join collapses them last-write-wins with no warning.
  Tags are normally unique, matching the "one row per (run × model × method)"
  grain.
- **Exact-Fagin epsilon is 0.0, not NULL.** `_parse_fagin_name("Fagin TA
  (exact)")` yields epsilon 0.0, so exact-Fagin rows store `epsilon = 0.0`.
  Non-Fagin methods store NULL. Query accordingly (`method_family = 'Fagin'`
  to scope to Fagin rows).

## Out of scope

- Thread-scaling sweep timings (stay in the `.json` sidecar).
- Migrations / schema versioning.
- Any change to the Markdown report or existing `.json` sidecar contents.
