# Retrieval Experiment Results DB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append every `run_retrieval_experiment.py` invocation's per-method results (datetime, method, epsilon, agree@10/@K, nDCG@10, runtime) to a persistent SQLite database, additively alongside the existing Markdown + `.json` outputs.

**Architecture:** Add pure module-level helpers to `scripts/run_retrieval_experiment.py` that (a) classify a method name into a family, (b) join the three in-memory result structures (`metric_rows`, `agree_rows`, Phase 5b `head`) into one flat row per (encoder × method), reusing the existing name↔variant mapping used by `_frontier_chart_svg`, and (c) write those rows into a self-initializing SQLite table via stdlib `sqlite3`. Wire a `--results-db` CLI flag + `settings.results_db` default, and call the writer at the end of `main()` inside a try/except.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`, `uuid`, `datetime`; pytest/unittest for tests.

---

## Design reference (read before starting)

The three structures being joined (all already built in `main()`):

- `metric_rows`: list of `{"encoder": tag, "variant": label, "recall@10", "recall@K", "MRR@10", "nDCG@10", "n_queries"}`. Variants include simdq labels like `"asym b=2, +rescore"`, `"1-bit ham, no rescore"`, baseline names verbatim (`"FAISS HNSW (M=16, ef=128)"`, `"Fagin GTASD (eps=0.01, depth=0)"`), and `"FLAT (fp32 IP, exact)"`.
- `agree_rows`: list of `{"encoder", "variant", "agree@10", "agree@K"}` — same `variant` keys as `metric_rows` (both appended together in `run_quality`'s `record()`).
- `head`: list of per-source dicts `{"label", "D", "N", "queries", "systems": [{"name", "ms", "agree"}, ...]}`. There is one `head` entry per model, in the same order as `cfg["models"]`. Phase 5b `systems[].name` uses names like `"simdq asym b=2 (+rescore)"`, `"simdq 1-bit ham (no rescore)"`, `"FAISS HNSW (...)"`, `"Fagin GTASD (...)"`, `"Milvus FLAT (exact)"`.

Existing mapping logic to mirror lives in `_frontier_chart_svg` (`scripts/run_retrieval_experiment.py:1007`): `family_of(name)` and `quality_of(tag, name)`. `_parse_fagin_name` (`:1451`) returns `(algo, epsilon)` or `None`. Note the Phase 5b simdq names are `"simdq asym ..."` / `"simdq 1-bit ..."` and map to quality variants `"asym b=2, {mode}"` / `"1-bit ham, {mode}"` where `mode` is `+rescore`/`no rescore`.

The DB join is keyed off the **quality `variant`** string as the canonical `method`, because that is the label shared by `metric_rows` and `agree_rows`. Phase 5b runtime is attached to that method by mapping each Phase 5b system name to its quality variant (the inverse of `quality_of`). Methods that appear only in Phase 5b but have no quality variant (none currently — every Phase 5b engine has a quality row) or only in quality (e.g. `asym b=2, no rescore` DOES appear in both; `FLAT (fp32 IP, exact)` appears in quality but not as a distinct Phase 5b system) get NULLs for the missing side.

---

## Task 1: Method-family classifier helper

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` (add helper near `_parse_fagin_name`, around line 1461)
- Test: `tests/test_results_db.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_results_db.py`:

```python
import sqlite3
import unittest
from pathlib import Path

from scripts.run_retrieval_experiment import (
    _method_family, _join_method_rows, write_results_db,
)


class TestMethodFamily(unittest.TestCase):
    def test_families(self):
        self.assertEqual(_method_family("Milvus FLAT (exact)"), "Milvus")
        self.assertEqual(_method_family("FAISS HNSW (M=16, ef=128)"), "FAISS")
        self.assertEqual(_method_family("Fagin GTASD (eps=0.01, depth=0)"), "Fagin")
        self.assertEqual(_method_family("asym b=2, +rescore"), "simdq")
        self.assertEqual(_method_family("1-bit ham, no rescore"), "simdq")
        self.assertEqual(_method_family("simdq asym b=2 (+rescore)"), "simdq")
        self.assertEqual(_method_family("FLAT (fp32 IP, exact)"), "FLAT")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate ndocu && python -m pytest tests/test_results_db.py::TestMethodFamily -v`
Expected: FAIL — `ImportError: cannot import name '_method_family'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/run_retrieval_experiment.py` after `_parse_fagin_name` (after line 1460):

```python
def _method_family(name: str) -> str:
    """Bucket a method/variant name into an engine family for the DB row.

    Handles both Phase 5b system names ('simdq asym b=2 (+rescore)',
    'Milvus FLAT (exact)') and quality variant labels ('asym b=2, +rescore',
    '1-bit ham, no rescore', 'FLAT (fp32 IP, exact)').
    """
    if name.startswith("Milvus"):
        return "Milvus"
    if name.startswith("FAISS"):
        return "FAISS"
    if name.startswith("Fagin"):
        return "Fagin"
    if name.startswith("FLAT"):
        return "FLAT"
    # simdq quality variants ('asym b=2, ...', '1-bit ham, ...') and Phase 5b
    # simdq system names ('simdq asym ...', 'simdq 1-bit ...')
    return "simdq"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda activate ndocu && python -m pytest tests/test_results_db.py::TestMethodFamily -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_retrieval_experiment.py tests/test_results_db.py
git commit -m "Add method-family classifier for retrieval-experiment results DB"
```

---

## Task 2: Join helper — flatten quality + agreement + speed into per-method rows

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` (add `_join_method_rows` after `_method_family`)
- Test: `tests/test_results_db.py` (add `TestJoinMethodRows`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_results_db.py`:

```python
class TestJoinMethodRows(unittest.TestCase):
    def _fixture(self):
        metric_rows = [
            {"encoder": "m1", "variant": "asym b=2, +rescore",
             "recall@10": 0.5, "recall@K": 0.6, "MRR@10": 0.4,
             "nDCG@10": 0.55, "n_queries": 100},
            {"encoder": "m1", "variant": "Fagin GTASD (eps=0.01, depth=0)",
             "recall@10": 0.7, "recall@K": 0.8, "MRR@10": 0.6,
             "nDCG@10": 0.75, "n_queries": 100},
            {"encoder": "m1", "variant": "FLAT (fp32 IP, exact)",
             "recall@10": 0.9, "recall@K": 0.95, "MRR@10": 0.85,
             "nDCG@10": 0.92, "n_queries": 100},
        ]
        agree_rows = [
            {"encoder": "m1", "variant": "asym b=2, +rescore",
             "agree@10": 0.98, "agree@K": 0.97},
            {"encoder": "m1", "variant": "Fagin GTASD (eps=0.01, depth=0)",
             "agree@10": 0.88, "agree@K": 0.80},
        ]
        head = [{
            "label": "m1", "D": 384, "N": 1000, "queries": "real",
            "systems": [
                {"name": "simdq asym b=2 (+rescore)", "ms": 2.0, "agree": 0.97},
                {"name": "Fagin GTASD (eps=0.01, depth=0)", "ms": 50.0,
                 "agree": 0.80},
            ],
        }]
        return metric_rows, agree_rows, head

    def test_join_matches_quality_and_speed(self):
        metric_rows, agree_rows, head = self._fixture()
        rows = _join_method_rows([{"tag": "m1"}], metric_rows, agree_rows, head)
        by_method = {r["method"]: r for r in rows}

        asym = by_method["asym b=2, +rescore"]
        self.assertEqual(asym["encoder"], "m1")
        self.assertEqual(asym["method_family"], "simdq")
        self.assertIsNone(asym["epsilon"])
        self.assertAlmostEqual(asym["ndcg_at_10"], 0.55)
        self.assertAlmostEqual(asym["agree_at_10"], 0.98)
        self.assertAlmostEqual(asym["agree_at_k"], 0.97)
        self.assertAlmostEqual(asym["runtime_ms_per_q"], 2.0)
        self.assertAlmostEqual(asym["queries_per_sec"], 500.0)

        fagin = by_method["Fagin GTASD (eps=0.01, depth=0)"]
        self.assertEqual(fagin["method_family"], "Fagin")
        self.assertAlmostEqual(fagin["epsilon"], 0.01)
        self.assertAlmostEqual(fagin["runtime_ms_per_q"], 50.0)

    def test_flat_has_quality_but_null_speed(self):
        metric_rows, agree_rows, head = self._fixture()
        rows = _join_method_rows([{"tag": "m1"}], metric_rows, agree_rows, head)
        flat = next(r for r in rows if r["method"] == "FLAT (fp32 IP, exact)")
        self.assertAlmostEqual(flat["ndcg_at_10"], 0.92)
        self.assertIsNone(flat["runtime_ms_per_q"])
        self.assertIsNone(flat["queries_per_sec"])
        # FLAT has no agreement row -> NULL agreement
        self.assertIsNone(flat["agree_at_10"])

    def test_speed_only_when_quality_skipped(self):
        _, _, head = self._fixture()
        rows = _join_method_rows([{"tag": "m1"}], [], [], head)
        asym = next(r for r in rows if r["method"] == "asym b=2, +rescore")
        self.assertIsNone(asym["ndcg_at_10"])
        self.assertAlmostEqual(asym["runtime_ms_per_q"], 2.0)
        self.assertEqual(asym["method_family"], "simdq")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate ndocu && python -m pytest tests/test_results_db.py::TestJoinMethodRows -v`
Expected: FAIL — `ImportError: cannot import name '_join_method_rows'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/run_retrieval_experiment.py` after `_method_family`:

```python
def _speed_name_to_variant(name: str) -> str:
    """Map a Phase 5b system name to the quality `variant` key it shares with
    metric_rows/agree_rows. simdq Phase 5b names ('simdq asym b=2 (+rescore)',
    'simdq 1-bit ham (no rescore)') fold onto the quality variants
    ('asym b=2, +rescore', '1-bit ham, no rescore'); every other engine uses
    its name verbatim as the variant.
    """
    if name.startswith("simdq "):
        mode = "+rescore" if "+rescore" in name else "no rescore"
        var = "asym b=2" if name.startswith("simdq asym") else "1-bit ham"
        return f"{var}, {mode}"
    return name


def _join_method_rows(models, metric_rows, agree_rows, head):
    """Join quality (metric_rows), agreement (agree_rows), and Phase 5b speed
    (head) into one flat dict per (encoder, method), keyed by the quality
    `variant` label. Missing phases leave their columns as None.

    Returns a list of dicts with keys matching the `runs` table columns
    (minus run_id/run_ts/hostname/dataset/top_k/workers, which the writer adds).
    """
    metric_by = {(r["encoder"], r["variant"]): r for r in metric_rows}
    agree_by = {(r["encoder"], r["variant"]): r for r in agree_rows}

    # speed: (encoder, variant) -> system dict, using the model tag as encoder.
    speed_by = {}
    for m, hrow in zip(models, head or []):
        tag = m["tag"]
        for s in hrow.get("systems", []):
            speed_by[(tag, _speed_name_to_variant(s["name"]))] = s

    # union of every (encoder, method) key seen in any phase
    keys = set(metric_by) | set(agree_by) | set(speed_by)
    rows = []
    for enc, method in sorted(keys):
        mm = metric_by.get((enc, method))
        ag = agree_by.get((enc, method))
        sp = speed_by.get((enc, method))
        parsed = _parse_fagin_name(method)
        epsilon = parsed[1] if parsed is not None else None
        ms = sp["ms"] if sp else None
        rows.append({
            "encoder": enc,
            "method": method,
            "method_family": _method_family(method),
            "epsilon": epsilon,
            "ndcg_at_10": mm["nDCG@10"] if mm else None,
            "recall_at_10": mm["recall@10"] if mm else None,
            "recall_at_k": mm["recall@K"] if mm else None,
            "mrr_at_10": mm["MRR@10"] if mm else None,
            "agree_at_10": ag["agree@10"] if ag else None,
            "agree_at_k": ag["agree@K"] if ag else None,
            "n_queries": mm["n_queries"] if mm else None,
            "runtime_ms_per_q": ms,
            "queries_per_sec": (1000.0 / ms) if ms else None,
        })
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda activate ndocu && python -m pytest tests/test_results_db.py::TestJoinMethodRows -v`
Expected: PASS (all 3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_retrieval_experiment.py tests/test_results_db.py
git commit -m "Join quality/agreement/speed into per-method rows for results DB"
```

---

## Task 3: SQLite writer — self-initializing, append-only

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` (add `_RUNS_SCHEMA` + `write_results_db`; add `import sqlite3`, `import uuid`, and `datetime` to the imports block near line 74)
- Test: `tests/test_results_db.py` (add `TestWriteResultsDb`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_results_db.py`:

```python
class TestWriteResultsDb(unittest.TestCase):
    def _args(self, tmp):
        cfg = {"dataset": {"name": "nq"}, "settings": {"top_k": 100,
               "workers": 16}, "models": [{"tag": "m1"}]}
        isa = {"hostname": "euler"}
        metric_rows = [{"encoder": "m1", "variant": "asym b=2, +rescore",
                        "recall@10": 0.5, "recall@K": 0.6, "MRR@10": 0.4,
                        "nDCG@10": 0.55, "n_queries": 100}]
        agree_rows = [{"encoder": "m1", "variant": "asym b=2, +rescore",
                       "agree@10": 0.98, "agree@K": 0.97}]
        head = [{"label": "m1", "D": 384, "N": 1000, "queries": "real",
                 "systems": [{"name": "simdq asym b=2 (+rescore)",
                              "ms": 2.0, "agree": 0.97}]}]
        return cfg, isa, metric_rows, agree_rows, head

    def test_creates_and_appends(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "results.db"
            cfg, isa, mr, ar, head = self._args(d)
            n1 = write_results_db(db, cfg, isa, "run-a", "2026-07-15T10:00:00",
                                  mr, ar, head)
            n2 = write_results_db(db, cfg, isa, "run-b", "2026-07-15T11:00:00",
                                  mr, ar, head)
            self.assertEqual(n1, 1)
            self.assertEqual(n2, 1)
            con = sqlite3.connect(db)
            try:
                total = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                row = con.execute(
                    "SELECT run_id, run_ts, hostname, dataset, top_k, workers, "
                    "encoder, method, method_family, ndcg_at_10, agree_at_10, "
                    "agree_at_k, runtime_ms_per_q, queries_per_sec "
                    "FROM runs WHERE run_id='run-a'").fetchone()
            finally:
                con.close()
            self.assertEqual(total, 2)   # appended, not overwritten
            self.assertEqual(row[0], "run-a")
            self.assertEqual(row[2], "euler")
            self.assertEqual(row[3], "nq")
            self.assertEqual(row[4], 100)
            self.assertEqual(row[7], "asym b=2, +rescore")
            self.assertEqual(row[8], "simdq")
            self.assertAlmostEqual(row[12], 2.0)
            self.assertAlmostEqual(row[13], 500.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate ndocu && python -m pytest tests/test_results_db.py::TestWriteResultsDb -v`
Expected: FAIL — `ImportError: cannot import name 'write_results_db'`.

- [ ] **Step 3: Write minimal implementation**

First, extend the imports. The current block (`scripts/run_retrieval_experiment.py:64-75`) has `import csv`, `import json`, etc. and `from datetime import date`. Change the datetime import and add sqlite3/uuid:

Find:
```python
from datetime import date
```
Replace with:
```python
from datetime import date, datetime
import sqlite3
import uuid
```

Then add, just before `def write_results_db` (place near the other report helpers, e.g. after `_join_method_rows`):

```python
_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    run_ts            TEXT NOT NULL,
    hostname          TEXT,
    dataset           TEXT,
    top_k             INTEGER,
    workers           INTEGER,
    encoder           TEXT,
    method            TEXT NOT NULL,
    method_family     TEXT,
    epsilon           REAL,
    ndcg_at_10        REAL,
    recall_at_10      REAL,
    recall_at_k       REAL,
    mrr_at_10         REAL,
    agree_at_10       REAL,
    agree_at_k        REAL,
    n_queries         INTEGER,
    runtime_ms_per_q  REAL,
    queries_per_sec   REAL
)
"""

_RUNS_COLUMNS = [
    "run_id", "run_ts", "hostname", "dataset", "top_k", "workers",
    "encoder", "method", "method_family", "epsilon",
    "ndcg_at_10", "recall_at_10", "recall_at_k", "mrr_at_10",
    "agree_at_10", "agree_at_k", "n_queries",
    "runtime_ms_per_q", "queries_per_sec",
]


def write_results_db(db_path, cfg, isa, run_id, run_ts,
                     metric_rows, agree_rows, head):
    """Append one row per (model x method) to the persistent SQLite `runs`
    table at db_path. Self-initializing (CREATE TABLE IF NOT EXISTS),
    append-only. Returns the number of rows written.
    """
    st = cfg["settings"]
    ds = cfg["dataset"]
    base = {
        "run_id": run_id,
        "run_ts": run_ts,
        "hostname": isa.get("hostname"),
        "dataset": ds.get("name"),
        "top_k": int(st["top_k"]),
        "workers": int(st.get("workers", 0)) or None,
    }
    joined = _join_method_rows(cfg["models"], metric_rows, agree_rows, head)
    rows = [tuple((base | r).get(c) for c in _RUNS_COLUMNS) for r in joined]

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.execute(_RUNS_SCHEMA)
        placeholders = ", ".join("?" for _ in _RUNS_COLUMNS)
        con.executemany(
            f"INSERT INTO runs ({', '.join(_RUNS_COLUMNS)}) "
            f"VALUES ({placeholders})", rows)
        con.commit()
    finally:
        con.close()
    return len(rows)
```

Note: `base | r` merges the shared per-run fields with the per-method fields; per-method keys never collide with `base` keys.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda activate ndocu && python -m pytest tests/test_results_db.py::TestWriteResultsDb -v`
Expected: PASS.

- [ ] **Step 5: Run the whole new test file**

Run: `conda activate ndocu && python -m pytest tests/test_results_db.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_retrieval_experiment.py tests/test_results_db.py
git commit -m "Write per-method retrieval results to persistent SQLite store"
```

---

## Task 4: Wire the CLI flag, default setting, and main() hook

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` — `DEFAULT_SETTINGS` (line 107), `_CLI_SETTINGS_MAP` (line 163), argparse in `main()` (near line 1821), and the tail of `main()` (after line 1920).

- [ ] **Step 1: Add the default setting**

In `DEFAULT_SETTINGS` (`scripts/run_retrieval_experiment.py:107`), add after the `"index_root"` entry (line 130):

```python
    "results_db": "experiments/simdq/results.db",  # persistent per-method store
```

- [ ] **Step 2: Register the CLI override**

In `_CLI_SETTINGS_MAP` (`scripts/run_retrieval_experiment.py:163`), add an entry so `--results-db` flows through both the pre-Jinja override and the defaults merge:

```python
    ("results_db", "results_db"),
```

(Insert it in the list alongside `("index_root", "index_root")`.)

- [ ] **Step 3: Add the argparse flag**

In `main()`, next to `--index-root` (`scripts/run_retrieval_experiment.py:1772`), add:

```python
    ap.add_argument("--results-db", dest="results_db", default=None,
                    help="persistent SQLite file to append per-method results "
                         "to (default: experiments/simdq/results.db). Set to "
                         "an empty string to disable.")
```

- [ ] **Step 4: Generate run_id/run_ts early in main()**

Immediately after `cfg = load_config(args)` and the `ds, st, models = ...` unpack (`scripts/run_retrieval_experiment.py:1826-1827`), add:

```python
    run_id = uuid.uuid4().hex
    run_ts = datetime.now().isoformat(timespec="seconds")
```

- [ ] **Step 5: Call the writer at the end of main()**

After the `.json` sidecar block that ends with `print(f"# raw    -> {json_path}")` (`scripts/run_retrieval_experiment.py:1920`), add:

```python
    db_setting = st.get("results_db")
    if db_setting:
        db_path = _resolve(db_setting)
        try:
            n = write_results_db(db_path, cfg, isa, run_id, run_ts,
                                 metric_rows, agree_rows, head or [])
            print(f"# results-db -> {db_path}  (+{n} rows)")
        except Exception as e:
            print(f"# results-db skipped: {e}", file=sys.stderr)
```

- [ ] **Step 6: Verify the script still imports and parses args**

Run: `conda activate ndocu && python scripts/run_retrieval_experiment.py --help`
Expected: help text prints, including the new `--results-db` option; no import errors.

- [ ] **Step 7: Smoke-test the DB write path end-to-end (unit-level)**

Run: `conda activate ndocu && python -m pytest tests/test_results_db.py -v`
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_retrieval_experiment.py
git commit -m "Wire --results-db flag and hook results DB write into main()"
```

---

## Task 5: Full test suite sanity check

**Files:** none (verification only)

- [ ] **Step 1: Run the new tests plus a quick import check of neighbors**

Run: `conda activate ndocu && python -m pytest tests/test_results_db.py tests/test_simdq_index.py -q`
Expected: PASS (test_results_db fully; test_simdq_index confirms the shared module still imports cleanly).

- [ ] **Step 2: Confirm no stray DB file was committed**

Run: `git status --porcelain | grep -i results.db || echo "clean"`
Expected: `clean` (the runtime DB lives at `experiments/simdq/results.db` and should not be committed as part of this feature).

---

## Self-review notes

- **Spec coverage:** datetime (`run_ts`), method (`method`), epsilon (`epsilon` via `_parse_fagin_name`), agree@10 (`agree_at_10`), agree@100 (`agree_at_k`, == agree@100 at top_k=100), nDCG@10 (`ndcg_at_10`), runtime + timing (`runtime_ms_per_q`, `queries_per_sec`) — all covered by Tasks 2-3. SQLite default + override — Task 4. Self-initializing/append-only — Task 3. Robust to skipped phases — Task 2 tests. try/except so DB failure never loses outputs — Task 4 Step 5.
- **Placeholders:** none — every code step is complete.
- **Type consistency:** `_join_method_rows(models, metric_rows, agree_rows, head)`, `_method_family(name)`, `_speed_name_to_variant(name)`, `write_results_db(db_path, cfg, isa, run_id, run_ts, metric_rows, agree_rows, head)`, and `_RUNS_COLUMNS`/`_RUNS_SCHEMA` names are used identically across tasks and match the test imports.
- The `dict | dict` merge (`base | r`) requires Python 3.9+; the project targets 3.10+, so it is safe.
```
