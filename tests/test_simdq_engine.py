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


def test_search_all_parallel_matches_sequential(fake_corpus):
    """search_all (batch encode + threaded scan) must equal per-query search."""
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
        # Ground truth: one search() call per query.
        seq = [[p["id"] for p in eng.search(q).retrieved_passages] for q in queries]

        # Threaded batch path must produce identical rankings.
        for nt in (1, 4):
            res = eng.search_all(queries, num_threads=nt)
            got = [[p["id"] for p in r.retrieved_passages] for r in res]
            assert got == seq, f"search_all(num_threads={nt}) diverged: {got} != {seq}"


def test_dispatch_raises_for_bad_family():
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        cfg.simdq_family = "not_a_family"
        eng = create_retrieval_engine(cfg)
        with pytest.raises(ValueError, match="family must be"):
            eng.ingest(FakeCorpus(_planted_corpus_docs()), update=False)


# ---------------------------------------------------------------------------
# T3 gap-fill: factory dispatch over recipes, delete+reingest, missing encoder
# ---------------------------------------------------------------------------

# Recipes mirror scripts/bench_simdq_beir.py. R4/R5 (random_orthogonal at d=D/2)
# need a real-D encoder; they're covered end-to-end in T10 (recall regression).
RECIPE_CONFIGS = [
    ("R0", {"simdq_family": "hamming",    "simdq_b": 1,
            "simdq_projection": "identity", "simdq_d": None,
            "simdq_store_floats": True,  "simdq_rescore_alpha": 10}),
    ("R1", {"simdq_family": "asymmetric", "simdq_b": 1,
            "simdq_projection": "identity", "simdq_d": None,
            "simdq_store_floats": False, "simdq_rescore_alpha": 1}),
    ("R2", {"simdq_family": "asymmetric", "simdq_b": 1,
            "simdq_projection": "identity", "simdq_d": None,
            "simdq_store_floats": True,  "simdq_rescore_alpha": 10}),
    ("R3", {"simdq_family": "asymmetric", "simdq_b": 2,
            "simdq_projection": "identity", "simdq_d": None,
            "simdq_store_floats": False, "simdq_rescore_alpha": 1}),
]


@pytest.mark.parametrize(
    "recipe_id, overrides", RECIPE_CONFIGS, ids=[r[0] for r in RECIPE_CONFIGS]
)
def test_factory_dispatch_per_recipe(recipe_id, overrides):
    """create_retrieval_engine resolves to a SimdqEngine for every recipe."""
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        eng = create_retrieval_engine(cfg)
        assert type(eng).__name__ == "SimdqEngine"


def test_delete_then_reingest(fake_corpus):
    """delete_index then re-ingest must rebuild a working index."""
    pytest.importorskip("sentence_transformers")
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        eng = create_retrieval_engine(cfg)
        eng.ingest(fake_corpus, update=False)
        assert eng.has_index(cfg.index_name)
        eng.delete_index(cfg.index_name)
        assert not eng.has_index(cfg.index_name)
        # Second ingest should rebuild cleanly.
        eng2 = create_retrieval_engine(cfg)
        eng2.ingest(fake_corpus, update=False)
        assert eng2.has_index(cfg.index_name)


def test_missing_encoder_raises_clean_error(fake_corpus):
    """A non-existent HF model id must produce a clean error from a known
    family (OSError/RuntimeError/ValueError), not an opaque AttributeError
    from inside DenseEmbeddingFunction's lazy attrs."""
    pytest.importorskip("sentence_transformers")
    bad_id = "this/definitely-does-not-exist-on-hf-12345"
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(td)
        cfg.model_name = bad_id
        with pytest.raises((OSError, RuntimeError, ValueError)) as ei:
            eng = create_retrieval_engine(cfg)
            # Loader is lazy; ingest forces the encoder download.
            eng.ingest(fake_corpus, update=False)
        msg = str(ei.value).lower()
        assert (bad_id in str(ei.value)
                or "not a valid" in msg
                or "not found" in msg
                or "is not a local folder" in msg
                or "couldn't connect" in msg
                or "huggingface" in msg), \
            f"missing-encoder error message not informative: {ei.value!r}"
