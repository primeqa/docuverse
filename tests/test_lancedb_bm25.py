"""Tests for LanceDBBM25Engine — FTS-only retrieval over the shared `text` column."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("lancedb")


@pytest.fixture
def stub_config(tmp_path):
    """Minimal config sufficient to construct a BM25 engine."""
    cfg = SimpleNamespace(
        db_engine="lancedb-bm25",
        index_name="t_bm25",
        server=str(tmp_path),
        project_dir=str(tmp_path),
        top_k=10,
        ingestion_batch_size=8,
        bulk_batch=8,
        max_text_size=1024,
        data_template=SimpleNamespace(extra_fields=[]),
        duplicate_removal=None,
        rouge_duplicate_threshold=0.95,
        verbose=False,
        index_params=None,
    )
    cfg.__dict__["model_name"] = None
    return cfg


def test_bm25_class_importable():
    from docuverse.engines.retrieval.lancedb import LanceDBBM25Engine
    # Must be a subclass of the real (non-aliased) LanceDBEngine base.
    from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine as RealBase
    assert issubclass(LanceDBBM25Engine, RealBase)


def test_bm25_extra_schema_fields_is_empty(stub_config, monkeypatch):
    """BM25 indexes the existing text column; it adds no PyArrow fields."""
    from docuverse.engines.retrieval.lancedb import LanceDBBM25Engine

    # Bypass create_update_index since we're not really ingesting yet.
    eng = LanceDBBM25Engine(stub_config)
    assert eng.extra_schema_fields() == []


def test_bm25_encode_data_is_no_op(stub_config):
    from docuverse.engines.retrieval.lancedb import LanceDBBM25Engine
    eng = LanceDBBM25Engine(stub_config)
    out = eng.encode_data(["hello world", "foo bar"])
    # encode_data must return one entry per text, but BM25 has nothing to write.
    assert len(out) == 2
    assert all(v is None for v in out)


def test_bm25_search_uses_fts(stub_config, monkeypatch):
    """End-to-end: ingest 3 docs, query, get a hit."""
    from docuverse.engines.retrieval.lancedb import LanceDBBM25Engine
    from docuverse.engines.search_queries import SearchQueries

    eng = LanceDBBM25Engine(stub_config)
    eng.create_index()
    eng.table.add([
        {"id": "a", "text": "alpha brown fox jumps", "title": ""},
        {"id": "b", "text": "beta lazy dog sleeps", "title": ""},
        {"id": "c", "text": "gamma quick fox runs", "title": ""},
    ])
    eng.build_indexes()

    from docuverse.engines.data_template import default_query_template
    q = SearchQueries.Query(template=default_query_template, id="q1", text="fox", relevant=[], answers=[])
    res = eng.search(q)
    ids = [p["id"] for p in res.retrieved_passages] if hasattr(res, "retrieved_passages") \
        else [p["id"] for p in res]
    assert "a" in ids or "c" in ids  # one of the fox docs ranks
