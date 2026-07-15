import sqlite3
import unittest
from pathlib import Path

from scripts.run_retrieval_experiment import _method_family


class TestMethodFamily(unittest.TestCase):
    def test_families(self):
        self.assertEqual(_method_family("Milvus FLAT (exact)"), "Milvus")
        self.assertEqual(_method_family("FAISS HNSW (M=16, ef=128)"), "FAISS")
        self.assertEqual(_method_family("Fagin GTASD (eps=0.01, depth=0)"), "Fagin")
        self.assertEqual(_method_family("asym b=2, +rescore"), "simdq")
        self.assertEqual(_method_family("1-bit ham, no rescore"), "simdq")
        self.assertEqual(_method_family("simdq asym b=2 (+rescore)"), "simdq")
        self.assertEqual(_method_family("FLAT (fp32 IP, exact)"), "FLAT")
