"""End-to-end SimdqEngine tests with a real (tiny) encoder."""
import os
import tempfile

import numpy as np
import pytest

from docuverse.engines.data_template import default_query_template
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_engine_config_params import RetrievalArguments
from docuverse.utils.retrievers import create_retrieval_engine


# A tiny multilingual sentence-transformer model is small and fast on CPU.
# If it's unavailable in the test environment we skip rather than fail.
TINY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d


class FakeCorpus:
    """Minimal corpus shim that satisfies SearchCorpus's interface (len + slice)."""

    def __init__(self, docs):
        self._docs = docs

    def __len__(self):
        return len(self._docs)

    def __getitem__(self, index):
        return self._docs[index]


def _make_config(td: str, family: str = "asymmetric") -> RetrievalArguments:
    """Return a RetrievalArguments instance (not a raw dict) to avoid a
    double-__post_init__ cycle that would fail on the SparseConfig field."""
    cfg = RetrievalArguments()
    cfg.db_engine = "simdq"
    cfg.model_name = TINY_MODEL
    cfg.index_name = "test_simdq_e2e"
    cfg.project_dir = td
    cfg.simdq_family = family
    cfg.simdq_b = 2
    cfg.simdq_d = None
    cfg.simdq_projection = "identity"
    cfg.simdq_store_floats = True
    cfg.simdq_rescore_alpha = 10
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
        for j in range(4)        # ensure at least 16 to exercise multiple batches
    ]


@pytest.fixture
def fake_corpus():
    return FakeCorpus(_planted_corpus_docs())


@pytest.mark.parametrize("family", ["asymmetric", "hamming"])
def test_dispatch_and_round_trip(family, fake_corpus):
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td, family=family)
        eng = create_retrieval_engine(cfg)
        assert type(eng).__name__ == "SimdqEngine"
        ok = eng.ingest(fake_corpus, update=False)
        assert ok is True
        # files on disk
        meta_path = os.path.join(td, "simdq_data", "test_simdq_e2e", "meta.json")
        assert os.path.exists(meta_path)

        # search for "neural networks" should rank the neural-net or transformer doc highly
        q = SearchQueries.Query(template=default_query_template, id="q1",
                                text="neural networks deep learning")
        res = eng.search(q)
        ids = [p["id"] for p in res.retrieved_passages]
        # planted-relevant docs (indices 1, 3, 6, 8, 10 in the corpus) — we expect
        # at least one of them in the top 5.
        relevant = {"d1", "d3", "d6", "d8", "d10"}
        assert relevant.intersection(ids), \
            f"none of {relevant} in top-{cfg.top_k}: {ids}"


def test_dispatch_raises_for_bad_family():
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        cfg.simdq_family = "not_a_family"
        eng = create_retrieval_engine(cfg)
        with pytest.raises(ValueError, match="family must be"):
            eng.ingest(FakeCorpus(_planted_corpus_docs()), update=False)
