# Optional Thread-Scaling Sweeps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the simdq thread-scaling sweeps (Phase 1 asym b=2 default+BLAS=1, Phase 6/8 Hamming SoA, and the `.sweeps.svg` chart) opt-in and default off in `scripts/run_retrieval_experiment.py`, via a new `run_thread_sweeps` setting and `--thread-sweeps` flag.

**Architecture:** Add a `run_thread_sweeps: False` default setting, a `--thread-sweeps` store_true CLI flag mapped through `_cli_overrides`, and tighten the sweep block's guard in `main()` to require both `run_simdq` and `run_thread_sweeps`. The report/JSON already handle `None` sweep results, so no downstream changes are needed.

**Tech Stack:** Python 3.10+, argparse; pytest/unittest.

---

## Design reference (read before starting)

Relevant existing code in `scripts/run_retrieval_experiment.py`:

- `DEFAULT_SETTINGS` dict (~line 107) holds run toggles like `run_milvus`, `run_faiss`, `run_simdq` (all `True`), plus `threads`, `speed_queries`, etc.
- `_cli_overrides(args)` (~line 175) turns argparse attrs into dotted-path overrides. It reads these `args` attributes: `dataset_name`, `queries_jsonl`, `corpus_file`, the `_CLI_SETTINGS_MAP` keys (`top_k`, `alpha`, `speed_queries`, `warmup`, `workers`, `milvus_uri`, `index_root`, `seed`, `results_db`), `threads`, `fagin_epsilon`, `fagin_schedule`, `no_milvus`, `no_faiss`, `no_fagin`, `no_simdq`, `skip_quality`, `skip_speed`, `out`. The `no_*`/`skip_*` flags append `settings.* = False` only when truthy.
- The speed phase in `main()` (~lines 2259–2273):

```python
    if st["run_speed"]:
        if st.get("run_simdq", True):
            common = dict(threads=st["threads"],
                          queries=int(st["speed_queries"]),
                          warmup=int(st["warmup"]), K=int(st["top_k"]),
                          alpha=int(st["alpha"]), seed=int(st["seed"]))
            p1_default = run_sweep_phase("Phase 1 — asym b=2", "asymmetric", 2,
                                         sources, query_specs, blas_threads=None,
                                         **common)
            p1_blas1 = run_sweep_phase("Phase 1 — asym b=2", "asymmetric", 2,
                                       sources, query_specs, blas_threads=1,
                                       **common)
            p68 = run_sweep_phase("Phase 6/8 — Hamming SoA", "hamming", None,
                                  sources, query_specs, blas_threads=None,
                                  **common)
        if (st["run_milvus"] or st["run_faiss"] or st.get("run_fagin", True)
                or st.get("run_simdq", True)):
            try:
                head = bench_head_to_head(sources, query_specs, st)
            except Exception as e:
                print(f"# Phase 5b skipped: {e}", file=sys.stderr)
```

- `p1_default = p1_blas1 = p68 = None` are set just before this block.
- `write_report` guards the thread-scaling section + `.sweeps.svg` write with `if p1_default is not None:`, and the JSON sidecar coalesces with `... or []`. So skipping the sweeps needs no report/JSON change.
- The argparse block in `main()` defines flags like `--no-simdq`, `--skip-speed` (~lines 2185–2200 region).

The `_cli_overrides` test must construct an `argparse.Namespace` carrying every attribute the function reads. Do this with an explicit helper in the test (see Task 1) — do NOT rely on the inline parser (it's built inside `main()` and not separately importable).

---

## Task 1: Default setting + `--thread-sweeps` flag + `_cli_overrides` mapping

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` — `DEFAULT_SETTINGS`, `_cli_overrides`, and the argparse block in `main()`.
- Test: `tests/test_thread_sweeps_flag.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_thread_sweeps_flag.py`:

```python
import argparse
import unittest

from scripts.run_retrieval_experiment import DEFAULT_SETTINGS, _cli_overrides


def _ns(**overrides):
    """Build an argparse.Namespace with every attribute _cli_overrides reads,
    defaulted to the 'not passed' value, then apply overrides."""
    base = dict(
        dataset_name=None, queries_jsonl=None, corpus_file=None,
        top_k=None, alpha=None, speed_queries=None, warmup=None,
        workers=None, milvus_uri=None, index_root=None, seed=None,
        results_db=None, threads=None, fagin_epsilon=None,
        fagin_schedule=None, no_milvus=False, no_faiss=False,
        no_fagin=False, no_simdq=False, skip_quality=False,
        skip_speed=False, out=None, thread_sweeps=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestThreadSweepsFlag(unittest.TestCase):
    def test_default_is_off(self):
        self.assertIs(DEFAULT_SETTINGS["run_thread_sweeps"], False)

    def test_flag_sets_override(self):
        ov = _cli_overrides(_ns(thread_sweeps=True))
        self.assertIs(ov["settings.run_thread_sweeps"], True)

    def test_absent_flag_no_override(self):
        ov = _cli_overrides(_ns(thread_sweeps=False))
        self.assertNotIn("settings.run_thread_sweeps", ov)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_thread_sweeps_flag.py -v`
Expected: FAIL — `KeyError: 'run_thread_sweeps'` (default missing) and/or the override assertions fail.

- [ ] **Step 3: Add the default setting**

In `scripts/run_retrieval_experiment.py`, in `DEFAULT_SETTINGS`, add the new key next to `run_simdq` (which reads `"run_simdq": True,` with a trailing comment). Insert after the `run_simdq` entry and its comment lines:

```python
    "run_thread_sweeps": False,  # Phase 1 / Phase 6-8 thread-scaling sweeps
                                 # (+ .sweeps.svg); opt-in via --thread-sweeps
```

- [ ] **Step 4: Map the flag in `_cli_overrides`**

In `_cli_overrides`, next to the other boolean-flag overrides (after the `if args.no_simdq:` block), add:

```python
    if args.thread_sweeps:
        ov["settings.run_thread_sweeps"] = True
```

- [ ] **Step 5: Add the argparse flag**

In `main()`'s argparse block, next to `--no-simdq`, add:

```python
    ap.add_argument("--thread-sweeps", dest="thread_sweeps",
                    action="store_true",
                    help="run the simdq thread-scaling sweeps (Phase 1 asym "
                         "b=2 default+BLAS=1, Phase 6/8 Hamming SoA) and emit "
                         "the .sweeps.svg chart; off by default")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_thread_sweeps_flag.py -v`
Expected: PASS (all 3 tests).

- [ ] **Step 7: Verify --help lists the flag and the module imports**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python scripts/run_retrieval_experiment.py --help`
Expected: help prints, `--thread-sweeps` listed, no errors.

- [ ] **Step 8: Commit (targeted add only — never `git add -A`)**

```bash
cd /ssd5/raduf/sandbox/docuverse && git add scripts/run_retrieval_experiment.py tests/test_thread_sweeps_flag.py && git commit -m "Add run_thread_sweeps setting + --thread-sweeps flag (default off)"
```

---

## Task 2: Gate the sweep block + update docstring/comments

**Files:**
- Modify: `scripts/run_retrieval_experiment.py` — the speed-phase sweep guard (~line 2260), the module docstring, and inline comments referencing the sweeps.

- [ ] **Step 1: Tighten the sweep guard**

In `main()`, change the sweep block's condition from:

```python
        if st.get("run_simdq", True):
```

to:

```python
        if st.get("run_simdq", True) and st.get("run_thread_sweeps", False):
```

(This is the `if` immediately inside `if st["run_speed"]:` that assigns `p1_default`/`p1_blas1`/`p68`. Do NOT change the second `if` that calls `bench_head_to_head`.)

- [ ] **Step 2: Update the module docstring**

Near the top of the file, the module docstring has a numbered step describing the speed phase (step "4." — it mentions "Phase 1 (asym b=2, default BLAS and BLAS=1), Phase 6/8 (hamming SoA) — reusing the sweep code in scripts/investigate_simdq_hardware.py"). Append a sentence to that bullet so it reads that these thread-scaling sweeps are opt-in:

Find the sentence ending `reusing the sweep code in scripts/investigate_simdq_hardware.py —` … within step 4 and add, at the end of that Phase 1 / 6-8 description (before the "and a Phase 5b head-to-head" continuation), the clause:

```
(these thread-scaling sweeps are opt-in via --thread-sweeps; off by default)
```

Keep the rest of step 4 (the Phase 5b head-to-head description) unchanged. If the exact insertion point is awkward, instead add a standalone line right after the Phase 6/8 clause; the goal is simply that the docstring states the sweeps default off.

- [ ] **Step 3: Update the `--no-simdq` help text note if it references sweeps**

The `--no-simdq` flag's help text mentions "quality variants, thread sweeps, and Phase 5b rows". Leave that accurate — `--no-simdq` still disables sweeps (via the `run_simdq` half of the new guard). No change required, but re-read it to confirm it isn't now misleading; if it implies sweeps run by default, adjust only if clearly wrong. (Expected: no change needed.)

- [ ] **Step 4: Verify behavior — sweeps off by default, on with flag**

Confirm the guard logic by inspecting that with defaults `run_thread_sweeps` is False so the block is skipped. Run the import + help sanity:

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -c "import scripts.run_retrieval_experiment as R; print('run_thread_sweeps default =', R.DEFAULT_SETTINGS['run_thread_sweeps'])"`
Expected: prints `run_thread_sweeps default = False`.

- [ ] **Step 5: Run the flag tests + a neighbor suite for import sanity**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_thread_sweeps_flag.py tests/test_latest_summary.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit (targeted add)**

```bash
cd /ssd5/raduf/sandbox/docuverse && git add scripts/run_retrieval_experiment.py && git commit -m "Gate simdq thread-scaling sweeps behind run_thread_sweeps (default off)"
```

---

## Task 3: Full-suite sanity check (verification only)

**Files:** none.

- [ ] **Step 1: Run the new + related suites**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -m pytest tests/test_thread_sweeps_flag.py tests/test_latest_summary.py tests/test_results_db.py -q`
Expected: all PASS.

- [ ] **Step 2: Module import sanity**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python -c "import scripts.run_retrieval_experiment"`
Expected: no error.

- [ ] **Step 3: Confirm the two flags coexist cleanly in help**

Run: `cd /ssd5/raduf/sandbox/docuverse && conda run -n ndocu python scripts/run_retrieval_experiment.py --help | grep -E 'thread-sweeps|no-simdq'`
Expected: both flags listed.

---

## Self-review notes

- **Spec coverage:** new `run_thread_sweeps: False` default (Task 1 Step 3); `--thread-sweeps` flag (Task 1 Step 5); `_cli_overrides` mapping applied only when passed (Task 1 Step 4, tested Step 1); tightened guard requiring both `run_simdq` and `run_thread_sweeps` (Task 2 Step 1); no report/JSON changes (relies on existing `None` guards — noted, not modified); docstring/comment update (Task 2 Step 2); tests for default + override presence/absence + help (Task 1 + Task 3). All covered.
- **Placeholder scan:** none — every code step shows exact code. Task 2 Step 2's insertion is described with a fallback for the exact anchor, but the required outcome (docstring states sweeps default off) is explicit.
- **Type consistency:** setting key `run_thread_sweeps`, argparse `dest="thread_sweeps"`, override key `settings.run_thread_sweeps`, and guard `st.get("run_thread_sweeps", False)` are consistent across tasks and match the test. The test's `_ns` helper sets `thread_sweeps` (the argparse dest) to mirror the real Namespace.
- **Ordering:** Task 1 adds the argparse flag and the `_cli_overrides` read together in the same commit, so no real run hits an `AttributeError` on `args.thread_sweeps`.
