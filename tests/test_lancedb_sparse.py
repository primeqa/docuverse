"""Tests for LanceDBSparseEngine — sparse vectors as a struct column with Python scoring."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

pytest.importorskip("lancedb")


class _FakeSparseModel:
    """Deterministic fake sparse encoder."""
    embedding_dim = 100

    def encode(self, texts, **kwargs):
        out = []
        for t in texts:
            n = len(t)
            ids = [hash(t[:i]) % 100 for i in range(1, min(n, 4) + 1)]
            vals = [1.0 / (i + 1) for i in range(len(ids))]
            out.append(csr_matrix((vals, ([0] * len(ids), ids)), shape=(1, 100)))
        return out


@pytest.fixture
def stub_sparse_config(tmp_path):
    cfg = SimpleNamespace(
        db_engine="lancedb-sparse",
        index_name="t_sparse",
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
        embeddings_name="splade_sparse",
    )
    cfg.__dict__["model_name"] = "fake-sparse"
    return cfg


def test_sparse_class_importable():
    from docuverse.engines.retrieval.lancedb import LanceDBSparseEngine
    from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine
    assert issubclass(LanceDBSparseEngine, LanceDBEngine)


def test_sparse_struct_schema(stub_sparse_config, monkeypatch):
    """The sparse engine adds one struct column with indices + values."""
    from docuverse.engines.retrieval.lancedb import lancedb_sparse as ls
    monkeypatch.setattr(ls, "SparseEmbeddingFunction", lambda *a, **k: _FakeSparseModel())

    eng = ls.LanceDBSparseEngine(stub_sparse_config)
    fields = eng.extra_schema_fields()
    assert len(fields) == 1
    assert fields[0].name == "splade_sparse"
    # Struct of indices + values
    sub_names = {f.name for f in fields[0].type}
    assert sub_names == {"indices", "values"}


def test_sparse_encode_data_returns_struct_dicts(stub_sparse_config, monkeypatch):
    from docuverse.engines.retrieval.lancedb import lancedb_sparse as ls
    monkeypatch.setattr(ls, "SparseEmbeddingFunction", lambda *a, **k: _FakeSparseModel())
    eng = ls.LanceDBSparseEngine(stub_sparse_config)
    out = eng.encode_data(["hello", "world!"])
    assert len(out) == 2
    for entry in out:
        assert set(entry) == {"indices", "values"}
        assert len(entry["indices"]) == len(entry["values"])


def test_sparse_dot_product_scoring(stub_sparse_config, monkeypatch):
    """A query that overlaps with one document on key indices ranks it first."""
    from docuverse.engines.retrieval.lancedb import lancedb_sparse as ls
    from docuverse.engines.search_queries import SearchQueries
    from docuverse.engines.data_template import default_query_template
    monkeypatch.setattr(ls, "SparseEmbeddingFunction", lambda *a, **k: _FakeSparseModel())

    eng = ls.LanceDBSparseEngine(stub_sparse_config)
    eng.create_index()
    eng.table.add([
        {"id": "a", "text": "x", "title": "",
         "splade_sparse": {"indices": [1, 2, 3], "values": [1.0, 0.5, 0.25]}},
        {"id": "b", "text": "y", "title": "",
         "splade_sparse": {"indices": [10, 11], "values": [1.0, 1.0]}},
        {"id": "c", "text": "z", "title": "",
         "splade_sparse": {"indices": [2, 3, 4], "values": [0.5, 0.5, 0.5]}},
    ])
    eng.build_indexes()  # no-op for sparse

    q = SearchQueries.Query(template=default_query_template, id="q1", text="...", relevant=[], answers=[])
    # Inject a known query sparse vector by monkeypatching encode_query.
    monkeypatch.setattr(eng, "encode_query", lambda question, tm=None:
        csr_matrix(([1.0, 1.0], ([0, 0], [2, 3])), shape=(1, 100)))

    res = eng.search(q)
    passages = res.retrieved_passages if hasattr(res, "retrieved_passages") else res
    ids_in_order = [p["id"] for p in passages]
    # 'a' has 0.5 + 0.25 = 0.75; 'c' has 0.5 + 0.5 = 1.0; 'b' has 0.
    assert ids_in_order[0] == "c"
    assert "a" in ids_in_order
