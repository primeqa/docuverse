# `--latest` Results Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only `--latest` action to `scripts/run_retrieval_experiment.py` that prints (and optionally writes as CSV) the newest result per method type from the SQLite results DB, showing a few representative epsilons for Fagin schedules.

**Architecture:** Pure helpers do the work — `_method_type` buckets a method into a type, `_select_representative_epsilons` picks epsilons, `build_latest_summary` reduces raw DB rows to sorted display rows, `render_latest_table`/`write_latest_csv` format them, and `query_latest_rows`/`run_latest_report` wire it to the DB and stdout. `main()` short-circuits on `--latest` before loading any config.

**Tech Stack:** Python 3.10+, stdlib `sqlite3` + `csv`; pytest/unittest.

---

## Design reference (read before starting)

The results DB `runs` table (written by the existing `write_results_db`) has these columns:
`run_id, run_ts, hostname, dataset, top_k, workers, encoder, method, method_family, epsilon, ndcg_at_10, recall_at_10, recall_at_k, mrr_at_10, agree_at_10, agree_at_k, n_queries, runtime_ms_per_q, queries_per_sec` (plus an autoincrement `id`).

Existing helpers already in `scripts/run_retrieval_experiment.py`:
- `_parse_fagin_name(name)` → `(algo, epsilon)` for Fagin names (`algo` ∈ TA/TASD/GTA/GTASD; exact → epsilon 0.0), or `None` for non-Fagin.
- `_resolve(path)` → repo-relative-or-absolute `Path`.
- `DEFAULT_SETTINGS["results_db"]` = `"experiments/simdq/results.db"`.

All new code goes in `scripts/run_retrieval_experiment.py`. Place the new pure helpers immediately after `write_results_db` (which ends just before the `_FAGIN_ALGO_COLOR` constant). New tests go in a new file `tests/test_latest_summary.py`.

Column display order for the table and CSV (list of `(key, header)`):
```
("dataset", "dataset"), ("encoder", "encoder"), ("method_type", "method_type"),
("epsilon", "epsilon"), ("agree_at_10", "agree@10"), ("agree_at_k", "agree@K"),
("ndcg_at_10", "nDCG@10"), ("runtime_ms_per_q", "ms/q"),
("queries_per_sec", "q/s"), ("run_ts", "run_ts")
```
This is the single source of column order; Task 4 defines it as `_LATEST_COLUMNS` and Tasks 4-5 reuse it.

---

## Task 1: `_method_type` — bucket a method into its display type

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` (add after `write_results_db`)
- Test: `tests/test_latest_summary.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_latest_summary.py`:

```python
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.run_retrieval_experiment import _method_type


class TestMethodType(unittest.TestCase):
    def test_fagin_uses_algo(self):
        self.assertEqual(
            _method_type("Fagin GTASD (eps=0.01, depth=0)", "Fagin"),
            "Fagin GTASD")
        self.assertEqual(
            _method_type("Fagin TA (exact)", "Fagin"), "Fagin TA")

    def test_baselines_strip_paren_suffix(self):
        self.assertEqual(
            _method_type("FAISS HNSW (M=16, ef=128)", "FAISS"), "FAISS HNSW")
        self.assertEqual(
            _method_type("Milvus FLAT (exact)", "Milvus"), "Milvus FLAT")
        self.assertEqual(
            _method_type("FLAT (fp32 IP, exact)", "FLAT"), "FLAT")

    def test_simdq_labels_pass_through(self):
        self.assertEqual(
            _method_type("asym b=2, +rescore", "simdq"), "asym b=2, +rescore")
        self.assertEqual(
            _method_type("1-bit ham, no rescore", "simdq"),
            "1-bit ham, no rescore")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py -v`
Expected: FAIL — `ImportError: cannot import name '_method_type'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/run_retrieval_experiment.py` immediately after the `write_results_db` function:

```python
def _method_type(method: str, family: str) -> str:
    """Bucket a stored method into its display 'method type'. Fagin rows keep
    their schedule (Fagin TA / TASD / GTA / GTASD) so epsilon-swept rows of the
    same schedule group together; every other engine drops its parenthesized
    parameter suffix ('FAISS HNSW (M=16, ef=128)' -> 'FAISS HNSW'). simdq
    labels have no ' (' suffix and pass through unchanged.
    """
    if family == "Fagin":
        parsed = _parse_fagin_name(method)
        if parsed is not None:
            return f"Fagin {parsed[0]}"
    return method.split(" (")[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py -v`
Expected: PASS.

- [ ] **Step 5: Commit (targeted add only — never `git add -A`)**

```bash
cd /ssd5/raduf/sandbox/docuverse && git add scripts/run_retrieval_experiment.py tests/test_latest_summary.py && git commit -m "Add _method_type bucketing for --latest summary"
```

---

## Task 2: `_select_representative_epsilons` — pick exact + low/mid/high

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` (add after `_method_type`)
- Test: `tests/test_latest_summary.py` (add `TestSelectEpsilons`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_latest_summary.py` — update the import line at the top from
`from scripts.run_retrieval_experiment import _method_type`
to
`from scripts.run_retrieval_experiment import (_method_type, _select_representative_epsilons)`
and append:

```python
class TestSelectEpsilons(unittest.TestCase):
    def test_exact_low_mid_high(self):
        # exact(0) + smallest nonzero + median nonzero + largest nonzero
        got = _select_representative_epsilons([0.0, 0.001, 0.005, 0.01, 0.05])
        self.assertEqual(got, [0.0, 0.001, 0.005, 0.05])

    def test_all_when_few(self):
        self.assertEqual(_select_representative_epsilons([0.01]), [0.01])
        self.assertEqual(
            _select_representative_epsilons([0.0, 0.01]), [0.0, 0.01])
        self.assertEqual(
            _select_representative_epsilons([0.0, 0.001, 0.01, 0.05]),
            [0.0, 0.001, 0.01, 0.05])

    def test_dedup_and_sort_input_order_irrelevant(self):
        got = _select_representative_epsilons([0.05, 0.0, 0.01, 0.005, 0.001])
        self.assertEqual(got, [0.0, 0.001, 0.005, 0.05])

    def test_no_exact(self):
        # >4 values, no zero present: low/mid/high of the nonzero values.
        # nonzero=[0.001,0.005,0.01,0.02,0.05]; low=0.001,
        # mid=nonzero[(5-1)//2]=nonzero[2]=0.01, high=0.05
        got = _select_representative_epsilons(
            [0.001, 0.005, 0.01, 0.02, 0.05])
        self.assertEqual(got, [0.001, 0.01, 0.05])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py::TestSelectEpsilons -v`
Expected: FAIL — ImportError on `_select_representative_epsilons`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/run_retrieval_experiment.py` after `_method_type`:

```python
def _select_representative_epsilons(values) -> list:
    """From the distinct epsilon values present for one Fagin type, pick a few
    representative ones spanning the trade-off: exact (0.0) if present, plus the
    smallest, median, and largest nonzero values. Deduplicated, sorted
    ascending, at most 4. If <=4 distinct values exist, all are returned.
    """
    uniq = sorted(set(float(v) for v in values))
    if len(uniq) <= 4:
        return uniq
    picked = []
    if uniq[0] == 0.0:
        picked.append(0.0)
    nonzero = [v for v in uniq if v != 0.0]
    if nonzero:
        low = nonzero[0]
        high = nonzero[-1]
        # lower-median so a 4-nonzero list picks index 1 (e.g. 0.005 from
        # [0.001, 0.005, 0.01, 0.05])
        mid = nonzero[(len(nonzero) - 1) // 2]
        for v in (low, mid, high):
            if v not in picked:
                picked.append(v)
    return sorted(picked)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py -v`
Expected: PASS (TestMethodType + TestSelectEpsilons). Sanity: for
`[0,0.001,0.005,0.01,0.05]` → picked `[0]`, then low=0.001,
mid=`nonzero[(4-1)//2]`=`nonzero[1]`=0.005, high=0.05 → sorted
`[0.0, 0.001, 0.005, 0.05]`.

- [ ] **Step 5: Commit (targeted add)**

```bash
cd /ssd5/raduf/sandbox/docuverse && git add scripts/run_retrieval_experiment.py tests/test_latest_summary.py && git commit -m "Add representative-epsilon selection for --latest summary"
```

---

## Task 3: `build_latest_summary` — reduce raw rows to sorted display rows

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` (add after `_select_representative_epsilons`)
- Test: `tests/test_latest_summary.py` (add `TestBuildLatestSummary`)

- [ ] **Step 1: Write the failing test**

Update the import at the top of `tests/test_latest_summary.py` to:
`from scripts.run_retrieval_experiment import (_method_type, _select_representative_epsilons, build_latest_summary)`
and append:

```python
def _row(method, family, epsilon, run_ts, ndcg, ms, ridx,
         dataset="nq", encoder="m1"):
    return {"id": ridx, "run_id": f"r{ridx}", "run_ts": run_ts,
            "hostname": "h", "dataset": dataset, "top_k": 100, "workers": 16,
            "encoder": encoder, "method": method, "method_family": family,
            "epsilon": epsilon, "ndcg_at_10": ndcg, "recall_at_10": None,
            "recall_at_k": None, "mrr_at_10": None, "agree_at_10": 0.9,
            "agree_at_k": 0.8, "n_queries": 100, "runtime_ms_per_q": ms,
            "queries_per_sec": (1000.0 / ms) if ms else None}


class TestBuildLatestSummary(unittest.TestCase):
    def test_latest_wins_per_method(self):
        rows = [
            _row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                 "2026-07-10T09:00:00", 0.70, 1.0, 1),
            _row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                 "2026-07-14T09:00:00", 0.72, 0.9, 2),  # newer
        ]
        out = build_latest_summary(rows)
        faiss = [r for r in out if r["method_type"] == "FAISS HNSW"]
        self.assertEqual(len(faiss), 1)
        self.assertAlmostEqual(faiss[0]["ndcg_at_10"], 0.72)
        self.assertEqual(faiss[0]["run_ts"], "2026-07-14T09:00:00")

    def test_fagin_representative_epsilons(self):
        rows = []
        ridx = 1
        for eps in (0.0, 0.001, 0.005, 0.01, 0.05):
            name = ("Fagin GTASD (exact)" if eps == 0.0
                    else f"Fagin GTASD (eps={eps:g}, depth=0)")
            rows.append(_row(name, "Fagin", eps, "2026-07-14T09:00:00",
                             0.9 - eps, 10.0, ridx))
            ridx += 1
        out = build_latest_summary(rows)
        gtasd = [r for r in out if r["method_type"] == "Fagin GTASD"]
        eps_shown = [r["epsilon"] for r in gtasd]
        self.assertEqual(eps_shown, [0.0, 0.001, 0.005, 0.05])

    def test_sorted_by_family_type_epsilon(self):
        rows = [
            _row("Fagin GTASD (eps=0.01, depth=0)", "Fagin", 0.01,
                 "2026-07-14T09:00:00", 0.8, 10.0, 1),
            _row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                 "2026-07-14T09:00:00", 0.72, 0.9, 2),
        ]
        out = build_latest_summary(rows)
        # FAISS sorts before Fagin (family alpha), each carries method_family
        self.assertEqual(out[0]["method_family"], "FAISS")
        self.assertEqual(out[-1]["method_family"], "Fagin")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py::TestBuildLatestSummary -v`
Expected: FAIL — ImportError on `build_latest_summary`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/run_retrieval_experiment.py` after `_select_representative_epsilons`:

```python
def _row_is_newer(a: dict, b: dict) -> bool:
    """True if row a is newer than row b (by run_ts, tiebreak on id)."""
    ka = (a.get("run_ts") or "", a.get("id") or 0)
    kb = (b.get("run_ts") or "", b.get("id") or 0)
    return ka > kb


def build_latest_summary(rows) -> list:
    """Reduce raw `runs` rows to the sorted display rows for the --latest
    table: the newest row per method string, bucketed into method types, with
    only representative epsilons kept for Fagin types. Each display row carries
    method_family + method_type alongside the metric columns.
    """
    # 1) newest row per exact method string
    latest_by_method: dict = {}
    for r in rows:
        m = r["method"]
        cur = latest_by_method.get(m)
        if cur is None or _row_is_newer(r, cur):
            latest_by_method[m] = r

    # 2) group those into method types
    by_type: dict = {}
    for r in latest_by_method.values():
        mt = _method_type(r["method"], r["method_family"])
        by_type.setdefault(mt, []).append(r)

    # 3) per type: non-epsilon -> latest single row; epsilon -> representatives
    display: list = []
    for mt, group in by_type.items():
        has_eps = any(g.get("epsilon") is not None for g in group)
        if not has_eps:
            best = group[0]
            for g in group[1:]:
                if _row_is_newer(g, best):
                    best = g
            display.append(_display_row(best, mt))
            continue
        eps_values = [float(g["epsilon"]) for g in group
                      if g.get("epsilon") is not None]
        keep = set(_select_representative_epsilons(eps_values))
        # newest row per (type, epsilon), only for kept epsilons
        best_by_eps: dict = {}
        for g in group:
            e = g.get("epsilon")
            if e is None or float(e) not in keep:
                continue
            e = float(e)
            cur = best_by_eps.get(e)
            if cur is None or _row_is_newer(g, cur):
                best_by_eps[e] = g
        for e in sorted(best_by_eps):
            display.append(_display_row(best_by_eps[e], mt))

    # 4) sort by family, then type, then epsilon (None first)
    display.sort(key=lambda d: (d["method_family"], d["method_type"],
                                -1.0 if d["epsilon"] is None
                                else float(d["epsilon"])))
    return display


def _display_row(r: dict, method_type: str) -> dict:
    """Project a raw row into a display row for the --latest table."""
    return {
        "dataset": r.get("dataset"),
        "encoder": r.get("encoder"),
        "method_family": r.get("method_family"),
        "method_type": method_type,
        "epsilon": (None if r.get("epsilon") is None
                    else float(r["epsilon"])),
        "agree_at_10": r.get("agree_at_10"),
        "agree_at_k": r.get("agree_at_k"),
        "ndcg_at_10": r.get("ndcg_at_10"),
        "runtime_ms_per_q": r.get("runtime_ms_per_q"),
        "queries_per_sec": r.get("queries_per_sec"),
        "run_ts": r.get("run_ts"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py -v`
Expected: PASS (all three classes).

- [ ] **Step 5: Commit (targeted add)**

```bash
cd /ssd5/raduf/sandbox/docuverse && git add scripts/run_retrieval_experiment.py tests/test_latest_summary.py && git commit -m "Reduce results rows to latest-per-method-type display rows"
```

---

## Task 4: Rendering — `_LATEST_COLUMNS`, `render_latest_table`, `write_latest_csv`

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` (add after `_display_row`)
- Test: `tests/test_latest_summary.py` (add `TestRenderAndCsv`)

- [ ] **Step 1: Write the failing test**

Update the import at the top of `tests/test_latest_summary.py` to add the new names:
`from scripts.run_retrieval_experiment import (_method_type, _select_representative_epsilons, build_latest_summary, render_latest_table, write_latest_csv, _LATEST_COLUMNS)`
and append:

```python
class TestRenderAndCsv(unittest.TestCase):
    def _display(self):
        return [
            {"dataset": "nq", "encoder": "m1", "method_family": "FAISS",
             "method_type": "FAISS HNSW", "epsilon": None,
             "agree_at_10": 0.95, "agree_at_k": 0.90, "ndcg_at_10": 0.72,
             "runtime_ms_per_q": 0.9, "queries_per_sec": 1111.1,
             "run_ts": "2026-07-14T09:00:00"},
            {"dataset": "nq", "encoder": "m1", "method_family": "Fagin",
             "method_type": "Fagin GTASD", "epsilon": 0.01,
             "agree_at_10": 0.88, "agree_at_k": 0.80, "ndcg_at_10": 0.75,
             "runtime_ms_per_q": 50.0, "queries_per_sec": 20.0,
             "run_ts": "2026-07-14T09:00:00"},
        ]

    def test_table_has_headers_and_values(self):
        txt = render_latest_table(self._display())
        self.assertIn("method_type", txt)
        self.assertIn("FAISS HNSW", txt)
        self.assertIn("Fagin GTASD", txt)
        self.assertIn("nDCG@10", txt)
        # epsilon blank for FAISS row, present for Fagin
        self.assertIn("0.01", txt)

    def test_empty_renders_message(self):
        self.assertIn("no rows", render_latest_table([]).lower())

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.csv"
            write_latest_csv(self._display(), p)
            with p.open(newline="") as f:
                got = list(csv.reader(f))
        header = [h for _, h in _LATEST_COLUMNS]
        self.assertEqual(got[0], header)
        self.assertEqual(len(got), 3)  # header + 2 rows
        # FAISS row epsilon cell is empty string
        self.assertEqual(got[1][header.index("epsilon")], "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py::TestRenderAndCsv -v`
Expected: FAIL — ImportError on the new names.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/run_retrieval_experiment.py` after `_display_row`:

```python
# (column key, header) — single source of column order for table + CSV
_LATEST_COLUMNS = [
    ("dataset", "dataset"), ("encoder", "encoder"),
    ("method_type", "method_type"), ("epsilon", "epsilon"),
    ("agree_at_10", "agree@10"), ("agree_at_k", "agree@K"),
    ("ndcg_at_10", "nDCG@10"), ("runtime_ms_per_q", "ms/q"),
    ("queries_per_sec", "q/s"), ("run_ts", "run_ts"),
]


def _fmt_cell(key: str, value) -> str:
    """Format one display value for the table/CSV. None -> ''. Floats use
    compact fixed precision per column; epsilon uses :g."""
    if value is None:
        return ""
    if key == "epsilon":
        return f"{float(value):g}"
    if key in ("agree_at_10", "agree_at_k", "ndcg_at_10"):
        return f"{float(value):.4f}"
    if key in ("runtime_ms_per_q", "queries_per_sec"):
        return f"{float(value):.2f}"
    return str(value)


def render_latest_table(display_rows) -> str:
    """Aligned monospace table of the --latest display rows."""
    if not display_rows:
        return "no rows to show"
    headers = [h for _, h in _LATEST_COLUMNS]
    cells = [[_fmt_cell(k, r.get(k)) for k, _ in _LATEST_COLUMNS]
             for r in display_rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    def fmt_line(vals):
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(vals))
    lines = [fmt_line(headers), fmt_line(["-" * w for w in widths])]
    lines += [fmt_line(row) for row in cells]
    return "\n".join(lines)


def write_latest_csv(display_rows, path) -> None:
    """Write the --latest display rows as CSV to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([h for _, h in _LATEST_COLUMNS])
        for r in display_rows:
            w.writerow([_fmt_cell(k, r.get(k)) for k, _ in _LATEST_COLUMNS])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py -v`
Expected: PASS (all classes).

- [ ] **Step 5: Commit (targeted add)**

```bash
cd /ssd5/raduf/sandbox/docuverse && git add scripts/run_retrieval_experiment.py tests/test_latest_summary.py && git commit -m "Render --latest summary as aligned table and CSV"
```

---

## Task 5: DB read + orchestrator — `query_latest_rows`, `run_latest_report`

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` (add after `write_latest_csv`)
- Test: `tests/test_latest_summary.py` (add `TestRunLatestReport`)

- [ ] **Step 1: Write the failing test**

Update the top import of `tests/test_latest_summary.py` to also import the DB writer and the new functions:
`from scripts.run_retrieval_experiment import (_method_type, _select_representative_epsilons, build_latest_summary, render_latest_table, write_latest_csv, _LATEST_COLUMNS, query_latest_rows, run_latest_report, write_results_db)`
and append:

```python
class TestRunLatestReport(unittest.TestCase):
    def _write_two_runs(self, db):
        cfg = {"dataset": {"name": "nq"},
               "settings": {"top_k": 100, "workers": 16},
               "models": [{"tag": "m1"}]}
        isa = {"hostname": "euler"}
        # older run: FAISS HNSW ndcg 0.70
        mr_old = [{"encoder": "m1", "variant": "FAISS HNSW (M=16, ef=128)",
                   "recall@10": 0.5, "recall@K": 0.6, "MRR@10": 0.4,
                   "nDCG@10": 0.70, "n_queries": 100}]
        write_results_db(db, cfg, isa, "old", "2026-07-10T09:00:00",
                         mr_old, [], [])
        # newer run: FAISS HNSW ndcg 0.72
        mr_new = [{"encoder": "m1", "variant": "FAISS HNSW (M=16, ef=128)",
                   "recall@10": 0.5, "recall@K": 0.6, "MRR@10": 0.4,
                   "nDCG@10": 0.72, "n_queries": 100}]
        write_results_db(db, cfg, isa, "new", "2026-07-14T09:00:00",
                         mr_new, [], [])

    def test_end_to_end_latest_and_csv(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "results.db"
            csv_path = Path(d) / "latest.csv"
            self._write_two_runs(db)
            rows = query_latest_rows(db)
            self.assertTrue(any(r["method"].startswith("FAISS HNSW")
                                for r in rows))
            buf = io.StringIO()
            with redirect_stdout(buf):
                run_latest_report(db, csv_path)
            out = buf.getvalue()
            self.assertIn("FAISS HNSW", out)
            self.assertIn("0.7200", out)      # newer value shown
            self.assertNotIn("0.7000", out)    # older value dropped
            self.assertTrue(csv_path.exists())

    def test_missing_db_message(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "nope.db"
            buf = io.StringIO()
            with redirect_stdout(buf):
                run_latest_report(db, None)
            self.assertIn("no results DB", buf.getvalue())

    def test_empty_db_message(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "empty.db"
            con = sqlite3.connect(str(db))
            con.close()  # file exists, no `runs` table
            buf = io.StringIO()
            with redirect_stdout(buf):
                run_latest_report(db, None)
            self.assertIn("no rows", buf.getvalue().lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py::TestRunLatestReport -v`
Expected: FAIL — ImportError on `query_latest_rows`/`run_latest_report`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/run_retrieval_experiment.py` after `write_latest_csv`:

```python
def query_latest_rows(db_path) -> list:
    """Read every row from the `runs` table as a list of dicts. Returns [] if
    the DB file or the table is absent."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    try:
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute("SELECT * FROM runs")
        except sqlite3.OperationalError:
            return []   # no `runs` table
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def run_latest_report(db_path, csv_path) -> None:
    """Print the latest-per-method-type summary to stdout; if csv_path is set,
    also write it as CSV. Prints a friendly message when there is nothing to
    show."""
    db_path = Path(db_path)
    rows = query_latest_rows(db_path)
    if not db_path.exists():
        print(f"no results DB at {db_path}")
        return
    if not rows:
        print(f"results DB has no rows yet ({db_path})")
        return
    display = build_latest_summary(rows)
    print(render_latest_table(display))
    if csv_path is not None:
        write_latest_csv(display, csv_path)
        print(f"# latest-csv -> {Path(csv_path)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py -v`
Expected: PASS (all classes).

- [ ] **Step 5: Commit (targeted add)**

```bash
cd /ssd5/raduf/sandbox/docuverse && git add scripts/run_retrieval_experiment.py tests/test_latest_summary.py && git commit -m "Add query_latest_rows + run_latest_report orchestrator"
```

---

## Task 6: CLI wiring — `--latest` / `--latest-csv` flags + early main() short-circuit

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` — argparse block (near `--results-db`, ~line 1786-1790) and the top of `main()` (right after `args = ap.parse_args()`, ~line 1824-1826).

- [ ] **Step 1: Add the argparse flags**

In `main()`, find the `--results-db` argument (added by the earlier results-DB feature). Immediately after it, add:

```python
    ap.add_argument("--latest", action="store_true",
                    help="print a summary table of the latest result per "
                         "method type from the results DB and exit (no "
                         "experiment is run); representative epsilons are "
                         "shown for Fagin schedules")
    ap.add_argument("--latest-csv", dest="latest_csv", default=None,
                    help="with --latest, also write the summary table as CSV "
                         "to this path")
```

- [ ] **Step 2: Add the early short-circuit in main()**

Find, at the top of `main()`, the line `args = ap.parse_args()`. Immediately after it (before `cfg = load_config(args)`), add:

```python
    if args.latest:
        db = _resolve(args.results_db or DEFAULT_SETTINGS["results_db"])
        run_latest_report(
            db, _resolve(args.latest_csv) if args.latest_csv else None)
        return
```

- [ ] **Step 3: Verify help and the flag path**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python scripts/run_retrieval_experiment.py --help`
Expected: help prints; `--latest` and `--latest-csv` are listed; no errors.

- [ ] **Step 4: Verify `--latest` runs against a real (or absent) DB without a config**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python scripts/run_retrieval_experiment.py --latest --results-db /tmp/does_not_exist_latest.db`
Expected: prints `no results DB at /tmp/does_not_exist_latest.db` and exits 0 (no config error, no traceback). This proves the short-circuit happens before `load_config`.

- [ ] **Step 5: Run the full new test file + a neighbor for import sanity**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py tests/test_results_db.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit (targeted add)**

```bash
cd /ssd5/raduf/sandbox/docuverse && git add scripts/run_retrieval_experiment.py && git commit -m "Wire --latest / --latest-csv flags into main()"
```

---

## Task 7: Full-suite sanity check (verification only)

**Files:** none.

- [ ] **Step 1: Run the two new/related suites**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_latest_summary.py tests/test_results_db.py -q`
Expected: all PASS.

- [ ] **Step 2: Module import sanity**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -c "import scripts.run_retrieval_experiment"`
Expected: no error.

- [ ] **Step 3: Confirm no stray DB/CSV artifacts were committed**

Run: `cd /ssd5/raduf/sandbox/docuverse && git status --porcelain | grep -Ei 'results\.db|latest\.csv' || echo "clean"`
Expected: `clean`.

---

## Self-review notes

- **Spec coverage:** `--latest` action + early short-circuit (Task 6); screen table default (Task 4/5); `--latest-csv` (Task 4/6); latest-per-method-globally + type bucketing (Tasks 1, 3); representative epsilons exact+low/mid/high (Task 2, used in Task 3); columns incl. `agree@K` header and blank NULLs (Task 4); missing/empty DB messages (Task 5); reads `--results-db`/default path (Task 6). All covered.
- **Placeholder scan:** none — every code step is complete. Task 2 Step 4 explicitly reconciles the median index (`mid = nonzero[(len(nonzero) - 1) // 2]`); ensure that line is used in the committed implementation.
- **Type consistency:** `_method_type(method, family)`, `_select_representative_epsilons(values)`, `build_latest_summary(rows)`, `_display_row(r, method_type)`, `_row_is_newer(a, b)`, `_LATEST_COLUMNS`, `_fmt_cell(key, value)`, `render_latest_table(display_rows)`, `write_latest_csv(display_rows, path)`, `query_latest_rows(db_path)`, `run_latest_report(db_path, csv_path)` — names/signatures consistent across tasks and match the test imports. Display-row keys match `_LATEST_COLUMNS` keys.
- **DRY:** `_LATEST_COLUMNS` is the single column-order source for both table and CSV; `_fmt_cell` is shared by both renderers; `_parse_fagin_name` reused for algo + epsilon.
