import sqlite3
import unittest
from pathlib import Path

from scripts.run_retrieval_experiment import _method_family, _join_method_rows


class TestMethodFamily(unittest.TestCase):
    def test_families(self):
        self.assertEqual(_method_family("Milvus FLAT (exact)"), "Milvus")
        self.assertEqual(_method_family("FAISS HNSW (M=16, ef=128)"), "FAISS")
        self.assertEqual(_method_family("Fagin GTASD (eps=0.01, depth=0)"), "Fagin")
        self.assertEqual(_method_family("asym b=2, +rescore"), "simdq")
        self.assertEqual(_method_family("1-bit ham, no rescore"), "simdq")
        self.assertEqual(_method_family("simdq asym b=2 (+rescore)"), "simdq")
        self.assertEqual(_method_family("FLAT (fp32 IP, exact)"), "FLAT")


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
