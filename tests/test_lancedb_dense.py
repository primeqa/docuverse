"""Tests for LanceDBDenseEngine — the refactored dense LanceDB engine."""
from __future__ import annotations

import pytest


def test_lancedb_dense_class_importable():
    """The new class lives at docuverse.engines.retrieval.lancedb.LanceDBDenseEngine."""
    from docuverse.engines.retrieval.lancedb import LanceDBDenseEngine
    from docuverse.engines.retrieval.lancedb import LanceDBEngine
    assert issubclass(LanceDBDenseEngine, LanceDBEngine)


def test_lancedb_engine_back_compat_alias():
    """``from .lancedb_engine import LanceDBEngine`` still works for back-compat."""
    from docuverse.engines.retrieval.lancedb.lancedb_engine import LanceDBEngine as Old
    from docuverse.engines.retrieval.lancedb import LanceDBDenseEngine
    # The old name now resolves to the dense subclass.
    assert Old is LanceDBDenseEngine
