"""Pytest configuration shared by all tests under tests/.

Custom CLI options must be registered in conftest.py — pytest_addoption
hooks in regular test modules are silently ignored.
"""
from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--update-baseline",
        action="store_true",
        default=False,
        help=(
            "Re-run T5 (test_simdq_recall.py) and overwrite "
            "tests/fixtures/simdq_scifact_baseline.json with measured "
            "NDCG@10 / Recall@100 values. One-shot operation; commit the "
            "regenerated JSON together with the kernel/encoder change "
            "that justified the move."
        ),
    )
