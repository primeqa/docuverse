import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.run_retrieval_experiment import (_method_type, _select_representative_epsilons)


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
