import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.run_retrieval_experiment import (_method_type, _select_representative_epsilons, build_latest_summary, render_latest_table, write_latest_csv, _LATEST_COLUMNS, query_latest_rows, run_latest_report, write_results_db)


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

    def test_empty_input(self):
        self.assertEqual(build_latest_summary([]), [])

    def test_display_rows_carry_all_quality_metrics(self):
        rows = [_row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                     "2026-07-14T09:00:00", 0.72, 0.9, 1)]
        # _row sets recall_at_10/recall_at_k/mrr_at_10 to None by default;
        # override them so we can assert they propagate to the display row
        rows[0]["recall_at_10"] = 0.65
        rows[0]["recall_at_k"] = 0.85
        rows[0]["mrr_at_10"] = 0.55
        out = build_latest_summary(rows)
        self.assertAlmostEqual(out[0]["recall_at_10"], 0.65)
        self.assertAlmostEqual(out[0]["recall_at_k"], 0.85)
        self.assertAlmostEqual(out[0]["mrr_at_10"], 0.55)

    def test_all_encoders_kept(self):
        # two models sharing method names must BOTH appear, not collapse to
        # whichever ran most recently
        rows = [
            _row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                 "2026-07-10T09:00:00", 0.70, 0.9, 1, encoder="m97"),
            _row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                 "2026-07-14T09:00:00", 0.75, 0.8, 2, encoder="m311"),
        ]
        out = build_latest_summary(rows)
        faiss = [r for r in out if r["method_type"] == "FAISS HNSW"]
        self.assertEqual(len(faiss), 2)
        self.assertEqual({r["encoder"] for r in faiss}, {"m97", "m311"})

    def test_latest_wins_per_encoder(self):
        # latest-wins is scoped per encoder, not global
        rows = [
            _row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                 "2026-07-10T09:00:00", 0.70, 0.9, 1, encoder="m97"),
            _row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                 "2026-07-14T09:00:00", 0.71, 0.9, 2, encoder="m97"),  # newer
            _row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                 "2026-07-12T09:00:00", 0.90, 0.8, 3, encoder="m311"),
        ]
        out = build_latest_summary(rows)
        by_enc = {r["encoder"]: r for r in out
                  if r["method_type"] == "FAISS HNSW"}
        self.assertAlmostEqual(by_enc["m97"]["ndcg_at_10"], 0.71)
        self.assertAlmostEqual(by_enc["m311"]["ndcg_at_10"], 0.90)

    def test_epsilon_selection_per_encoder(self):
        # each encoder gets its own representative-epsilon selection
        rows = []
        ridx = 1
        for enc in ("m97", "m311"):
            for eps in (0.0, 0.001, 0.005, 0.01, 0.05):
                name = ("Fagin GTASD (exact)" if eps == 0.0
                        else f"Fagin GTASD (eps={eps:g}, depth=0)")
                rows.append(_row(name, "Fagin", eps, "2026-07-14T09:00:00",
                                 0.9 - eps, 10.0, ridx, encoder=enc))
                ridx += 1
        out = build_latest_summary(rows)
        for enc in ("m97", "m311"):
            eps_shown = [r["epsilon"] for r in out
                         if r["method_type"] == "Fagin GTASD"
                         and r["encoder"] == enc]
            self.assertEqual(eps_shown, [0.0, 0.001, 0.005, 0.05])

    def test_sorted_groups_by_encoder(self):
        rows = [
            _row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                 "2026-07-14T09:00:00", 0.72, 0.9, 1, encoder="m311"),
            _row("FAISS HNSW (M=16, ef=128)", "FAISS", None,
                 "2026-07-14T09:00:00", 0.70, 0.9, 2, encoder="m97"),
        ]
        out = build_latest_summary(rows)
        # rows are grouped by encoder (sorted), so all of one encoder's rows
        # are contiguous
        encoders = [r["encoder"] for r in out]
        self.assertEqual(encoders, sorted(encoders))


class TestRenderAndCsv(unittest.TestCase):
    def _display(self):
        return [
            {"dataset": "nq", "encoder": "m1", "method_family": "FAISS",
             "method_type": "FAISS HNSW", "epsilon": None,
             "recall_at_10": 0.65, "recall_at_k": 0.85, "mrr_at_10": 0.55,
             "agree_at_10": 0.95, "agree_at_k": 0.90, "ndcg_at_10": 0.72,
             "runtime_ms_per_q": 0.9, "queries_per_sec": 1111.1,
             "run_ts": "2026-07-14T09:00:00"},
            {"dataset": "nq", "encoder": "m1", "method_family": "Fagin",
             "method_type": "Fagin GTASD", "epsilon": 0.01,
             "recall_at_10": 0.60, "recall_at_k": 0.80, "mrr_at_10": 0.50,
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
        # all computed quality metrics are surfaced
        self.assertIn("R@10", txt)
        self.assertIn("R@K", txt)
        self.assertIn("MRR@10", txt)
        self.assertIn("0.6500", txt)   # recall_at_10 value rendered
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
