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
