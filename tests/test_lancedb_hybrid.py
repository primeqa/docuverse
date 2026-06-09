"""Tests for LanceDBHybridEngine — composition, schema merging, validation."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

pytest.importorskip("lancedb")


class _FakeDenseModel:
    embedding_dim = 8

    def encode(self, texts, **kwargs):
        return [np.full(8, float(i + 1) / 10.0, dtype=np.float32) for i, _ in enumerate(texts)]


class _FakeSparseModel:
    embedding_dim = 50

    def encode(self, texts, **kwargs):
        out = []
        for i, t in enumerate(texts):
            ids = [(i * 7 + k) % 50 for k in range(3)]
            vals = [1.0, 0.5, 0.25]
            out.append(csr_matrix((vals, ([0] * 3, ids)), shape=(1, 50)))
        return out


def _hybrid_config(tmp_path, combination="rrf", with_sparse=True, with_bm25=True):
    models = {
        "dense": {
            "weight": 0.7,
            "top_k": 10,
            "db_engine": "lancedb_dense",
            "embeddings_name": "dense_vec",
            "model_name": "fake-dense",
        }
    }
    if with_bm25:
        models["bm25"] = {
            "weight": 0.2,
            "top_k": 10,
            "db_engine": "lancedb_bm25",
        }
    if with_sparse:
        models["splade"] = {
            "weight": 0.1,
            "top_k": 10,
            "db_engine": "lancedb_sparse",
            "embeddings_name": "splade_vec",
            "model_name": "fake-sparse",
        }
    cfg = SimpleNamespace(
        db_engine="lancedb-hybrid",
        index_name="t_hybrid",
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
        hybrid={
            "combination": combination,
            "rrf_k": 60,
            "models": models,
        },
        hybrid_submodules=None,
    )
    cfg.__dict__["model_name"] = None
    return cfg


@pytest.fixture
def patched_models(monkeypatch):
    """Patch dense and sparse encoders to deterministic fakes everywhere."""
    from docuverse.engines.retrieval.lancedb import lancedb_dense as ld
    from docuverse.engines.retrieval.lancedb import lancedb_sparse as ls
    monkeypatch.setattr(ld, "DenseEmbeddingFunction", lambda *a, **k: _FakeDenseModel())
    monkeypatch.setattr(ls, "SparseEmbeddingFunction", lambda *a, **k: _FakeSparseModel())
    return monkeypatch


def test_hybrid_class_importable():
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine
    assert issubclass(LanceDBHybridEngine, LanceDBEngine)


def test_hybrid_invalid_combination_raises(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="bogus")
    with pytest.raises(ValueError, match="combination"):
        LanceDBHybridEngine(cfg)


def test_hybrid_weighted_missing_weight_raises(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="weighted")
    del cfg.hybrid["models"]["bm25"]["weight"]
    with pytest.raises(ValueError, match="weight"):
        LanceDBHybridEngine(cfg)


def test_hybrid_rejects_milvus_subengine(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path)
    cfg.hybrid["models"]["dense"]["db_engine"] = "milvus_dense"
    with pytest.raises(ValueError, match="lancedb"):
        LanceDBHybridEngine(cfg)


def test_hybrid_rejects_duplicate_embeddings_name(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path)
    cfg.hybrid["models"]["splade"]["embeddings_name"] = "dense_vec"  # collision
    with pytest.raises(ValueError, match="embeddings_name"):
        LanceDBHybridEngine(cfg)


def test_hybrid_merged_schema_contains_all_columns(tmp_path, patched_models):
    """Schema includes id/text/title + dense_vec + splade_vec (BM25 adds none)."""
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path)
    eng = LanceDBHybridEngine(cfg)
    schema = eng._build_schema()
    names = {f.name for f in schema}
    assert {"id", "text", "title", "dense_vec", "splade_vec"} <= names


def test_hybrid_ingest_writes_one_row_per_doc(tmp_path, patched_models):
    """Ingestion writes a single row per document with all sub-engine columns populated."""
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path)
    eng = LanceDBHybridEngine(cfg)
    eng.create_index()

    docs = [
        {"id": "a", "text": "alpha brown fox", "title": ""},
        {"id": "b", "text": "beta lazy dog", "title": ""},
        {"id": "c", "text": "gamma quick fox", "title": ""},
    ]
    records = eng._create_data(docs)
    assert len(records) == 3
    for r in records:
        assert "dense_vec" in r and "splade_vec" in r
        assert isinstance(r["dense_vec"], list) and len(r["dense_vec"]) == 8
        assert set(r["splade_vec"]) == {"indices", "values"}


def _ingest_and_search(eng, docs, query_text):
    from docuverse.engines.search_queries import SearchQueries
    from docuverse.engines.data_template import default_query_template
    eng.create_index()
    records = eng._create_data(docs)
    eng._insert_data(records)
    eng.build_indexes()
    q = SearchQueries.Query(template=default_query_template, id="q1", text=query_text,
                            relevant=[], answers=[])
    return eng.search(q)


def test_hybrid_search_dense_bm25_rrf_native_fast_path(tmp_path, patched_models):
    """Dense + BM25 + RRF should use LanceDB's native hybrid query."""
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="rrf", with_sparse=False, with_bm25=True)
    eng = LanceDBHybridEngine(cfg)
    docs = [
        {"id": "a", "text": "alpha brown fox jumps high", "title": ""},
        {"id": "b", "text": "beta lazy dog sleeps low", "title": ""},
        {"id": "c", "text": "gamma quick fox runs fast", "title": ""},
    ]
    res = _ingest_and_search(eng, docs, "fox")
    passages = res.retrieved_passages if hasattr(res, "retrieved_passages") else res
    ids = [p["id"] for p in passages]
    assert {"a", "c"} & set(ids)


def test_hybrid_search_three_engines_rrf(tmp_path, patched_models):
    """Dense + BM25 + sparse with RRF combination falls back to Python merge."""
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="rrf", with_sparse=True, with_bm25=True)
    eng = LanceDBHybridEngine(cfg)
    docs = [
        {"id": "a", "text": "alpha brown fox jumps high", "title": ""},
        {"id": "b", "text": "beta lazy dog sleeps low", "title": ""},
        {"id": "c", "text": "gamma quick fox runs fast", "title": ""},
    ]
    res = _ingest_and_search(eng, docs, "fox")
    passages = res.retrieved_passages if hasattr(res, "retrieved_passages") else res
    ids = [p["id"] for p in passages]
    assert len(ids) > 0
    assert all(p.get("score") is not None for p in passages)


def test_hybrid_search_three_engines_weighted(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="weighted",
                         with_sparse=True, with_bm25=True)
    eng = LanceDBHybridEngine(cfg)
    docs = [
        {"id": "a", "text": "alpha brown fox jumps high", "title": ""},
        {"id": "b", "text": "beta lazy dog sleeps low", "title": ""},
        {"id": "c", "text": "gamma quick fox runs fast", "title": ""},
    ]
    res = _ingest_and_search(eng, docs, "fox")
    passages = res.retrieved_passages if hasattr(res, "retrieved_passages") else res
    assert len(passages) > 0


def test_hybrid_rrf_math_unit():
    """RRF: rank-1 contribution = 1/(60+1); rank-2 = 1/(60+2)."""
    from docuverse.engines.retrieval.lancedb.lancedb_hybrid import _rrf_combine
    rankings = [
        [("a", 5.0), ("b", 4.0), ("c", 3.0)],
        [("c", 9.0), ("a", 7.0), ("b", 5.0)],
    ]
    out = _rrf_combine(rankings, k=60, top_k=3)
    ids = [item[0] for item in out]
    # 'a' appears at ranks 1 and 2; 'c' at ranks 3 and 1; 'b' at ranks 2 and 3.
    # Score(a)=1/61+1/62; Score(c)=1/63+1/61; Score(b)=1/62+1/63.
    # a > c > b
    assert ids == ["a", "c", "b"]


def test_hybrid_weighted_math_unit():
    """Weighted: min-max normalize each ranking then weighted-sum."""
    from docuverse.engines.retrieval.lancedb.lancedb_hybrid import _weighted_combine
    rankings = [
        [("a", 1.0), ("b", 0.0)],     # weight 0.7
        [("b", 4.0), ("a", 2.0)],     # weight 0.3
    ]
    out = _weighted_combine(rankings, weights=[0.7, 0.3], top_k=2)
    by_id = dict(out)
    # 'a' normalized ranking 1 = 1.0; ranking 2 = 0.0  -> 0.7*1.0 + 0.3*0.0 = 0.7
    # 'b' normalized ranking 1 = 0.0; ranking 2 = 1.0  -> 0.7*0.0 + 0.3*1.0 = 0.3
    assert abs(by_id["a"] - 0.7) < 1e-9
    assert abs(by_id["b"] - 0.3) < 1e-9


def test_hybrid_single_subengine_skips_merge(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="rrf",
                         with_sparse=False, with_bm25=False)
    eng = LanceDBHybridEngine(cfg)
    docs = [
        {"id": "a", "text": "alpha brown fox", "title": ""},
        {"id": "b", "text": "beta lazy dog", "title": ""},
    ]
    res = _ingest_and_search(eng, docs, "fox")
    passages = res.retrieved_passages if hasattr(res, "retrieved_passages") else res
    assert len(passages) > 0
