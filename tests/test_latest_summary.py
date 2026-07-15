import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.run_retrieval_experiment import (_method_type, _select_representative_epsilons, build_latest_summary)


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
