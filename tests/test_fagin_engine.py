"""End-to-end FaginThresholdEngine tests with a real (tiny) encoder.

Mirrors tests/test_simdq_engine.py: skipped when sentence_transformers or
the native extension is unavailable."""
import os
import tempfile

import numpy as np
import pytest

from docuverse.engines.data_template import default_query_template
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_engine_config_params import RetrievalArguments
from docuverse.utils.retrievers import create_retrieval_engine

pytest.importorskip("docuverse.engines.retrieval.simdq._simdq_native")

TINY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d


class FakeCorpus:
    """Minimal corpus shim satisfying SearchCorpus's interface (len + slice)."""

    def __init__(self, docs):
        self._docs = docs

    def __len__(self):
        return len(self._docs)

    def __getitem__(self, index):
        return self._docs[index]


def _make_config(td: str) -> RetrievalArguments:
    cfg = RetrievalArguments()
    cfg.db_engine = "fagin"
    cfg.model_name = TINY_MODEL
    cfg.index_name = "test_fagin_e2e"
    cfg.project_dir = td
    cfg.top_k = 5
    cfg.ingestion_batch_size = 16
    cfg.max_text_size = 256
    return cfg


def _planted_corpus_docs():
    return [
        {"id": f"d{i}", "text": t, "title": ""}
        for i, t in enumerate([
            "the quick brown fox jumps over the lazy dog",
            "machine learning algorithms are statistical pattern matchers",
            "the cat sat on the warm mat in the morning sun",
            "neural networks approximate functions through layered transforms",
            "lions roar at dawn in the African savannah",
            "python is a high-level interpreted programming language",
            "transformers attend to all tokens via dot-product attention",
            "rabbits live in burrows and eat grass and clover",
            "BERT and GPT differ in their pretraining objectives",
            "elephants have the longest gestation of any mammal",
            "convolutional networks share weights across spatial positions",
            "the wolf is a social predator that hunts in packs",
        ])
    ] + [
        {"id": f"pad{j}", "text": f"padding doc {j}", "title": ""}
        for j in range(4)
    ]


@pytest.fixture
def fake_corpus():
    return FakeCorpus(_planted_corpus_docs())


def test_dispatch_and_round_trip(fake_corpus):
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        assert type(eng).__name__ == "FaginThresholdEngine"
        ok = eng.ingest(fake_corpus, update=False)
        assert ok is True
        meta_path = os.path.join(td, "fagin_data", "test_fagin_e2e", "meta.json")
        assert os.path.exists(meta_path)

        q = SearchQueries.Query(template=default_query_template, id="q1",
                                text="neural networks deep learning")
        res = eng.search(q)
        ids = [p["id"] for p in res.retrieved_passages]
        relevant = {"d1", "d3", "d6", "d8", "d10"}
        assert relevant.intersection(ids), \
            f"none of {relevant} in top-{cfg.top_k}: {ids}"
        # exact TA over 16 docs with K=5 must return exactly 5 results
        assert len(ids) == 5
        # stats accumulated
        assert eng.ta_stats["queries"] == 1
        assert eng.ta_stats["random_accesses"] >= 5


def test_search_matches_brute_force_over_encoded_corpus(fake_corpus):
    """The engine's exact mode must return exactly the brute-force top-k
    over the same encoded vectors — the core TA correctness claim, e2e."""
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        eng.ingest(fake_corpus, update=False)

        q = SearchQueries.Query(template=default_query_template, id="q1",
                                text="attention mechanism in transformers")
        res = eng.search(q)
        got_ids = [p["id"] for p in res.retrieved_passages]

        # Brute force over the persisted fp32 matrix (same vectors TA used).
        eng._ensure_loaded()
        Y = np.asarray(eng.index.Y, dtype=np.float64)
        qv = eng._query_emb_cache.get(id(q))
        if qv is None:
            eng._ensure_model()
            qv = np.asarray(eng.model.encode([q.text], show_progress_bar=False,
                                             prompt_name="query")[0])
        s = Y @ np.asarray(qv, dtype=np.float64)
        brute = [eng.id_map[i] for i in np.argsort(-s, kind="stable")[:cfg.top_k]]
        # Compare score-sets (tie-safe): every returned id must score >= kth
        kth = np.sort(s)[::-1][cfg.top_k - 1]
        idx_of = {d: i for i, d in enumerate(eng.id_map)}
        assert all(s[idx_of[d]] >= kth - 1e-5 for d in got_ids), \
            f"TA returned {got_ids}, brute force top-{cfg.top_k} is {brute}"


def test_search_all_parallel_matches_sequential(fake_corpus):
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        eng.ingest(fake_corpus, update=False)

        queries = [
            SearchQueries.Query(template=default_query_template, id=f"q{i}", text=t)
            for i, t in enumerate([
                "neural networks deep learning",
                "wild animals in the savannah",
                "interpreted programming languages",
                "attention mechanism in transformers",
                "small furry mammals",
            ])
        ]
        seq = [[p["id"] for p in eng.search(q).retrieved_passages] for q in queries]
        for nt in (1, 4):
            res = eng.search_all(queries, num_threads=nt)
            got = [[p["id"] for p in r.retrieved_passages] for r in res]
            assert got == seq, f"search_all(num_threads={nt}) diverged"


def test_delete_then_reingest(fake_corpus):
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        eng.ingest(fake_corpus, update=False)
        assert eng.has_index(cfg.index_name)
        eng.delete_index(cfg.index_name)
        assert not eng.has_index(cfg.index_name)
        eng2 = create_retrieval_engine(cfg)
        eng2.ingest(fake_corpus, update=False)
        assert eng2.has_index(cfg.index_name)


def test_info_reports_ta_stats(fake_corpus):
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        eng.ingest(fake_corpus, update=False)
        q = SearchQueries.Query(template=default_query_template, id="q1",
                                text="wolves hunting")
        eng.search(q)
        info = eng.info()
        assert info["retriever_type"] == "FaginThresholdEngine"
        assert info["ta_stats"]["queries"] == 1
        assert info["ta_stats"]["depth"] >= 1
        assert "index_meta" in info
