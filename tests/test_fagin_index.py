"""FaginIndex build/persist/search tests."""
import numpy as np
import pytest

pytest.importorskip("docuverse.engines.retrieval.simdq._simdq_native")

from docuverse.engines.retrieval.fagin.fagin_index import FaginIndex


def _rand(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d)).astype(np.float32),
            rng.standard_normal(d).astype(np.float32))


def test_build_shapes_and_dtypes():
    Y, _ = _rand()
    ix = FaginIndex.build(Y, encoder_id="test-enc")
    assert ix.n_vectors == 200 and ix.dim == 16
    assert ix.Y.shape == (200, 16) and ix.Y.dtype == np.float32
    assert ix.order.shape == (16, 200) and ix.order.dtype == np.int32
    assert ix.vals.shape == (16, 200) and ix.vals.dtype == np.float32
    # each dimension's vals must be descending
    assert np.all(np.diff(ix.vals, axis=1) <= 0)
    # order must be a permutation per dimension
    for j in range(16):
        assert sorted(ix.order[j].tolist()) == list(range(200))


def test_search_matches_brute_force():
    Y, q = _rand()
    ix = FaginIndex.build(Y)
    idxs, scores, stats = ix.search(q, K=10)
    s = Y.astype(np.float64) @ q.astype(np.float64)
    desc = np.sort(s)[::-1]
    np.testing.assert_allclose(np.sort(scores)[::-1], desc[:10], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(scores, s[idxs], rtol=1e-4, atol=1e-5)
    assert stats["depth"] >= 1


def test_save_load_roundtrip(tmp_path):
    Y, q = _rand()
    ix = FaginIndex.build(Y, encoder_id="enc-x")
    ix.save(tmp_path / "idx")
    lx = FaginIndex.load(tmp_path / "idx")
    assert lx.n_vectors == ix.n_vectors and lx.dim == ix.dim
    assert lx.encoder_id == "enc-x"
    i1, s1, _ = ix.search(q, K=8)
    i2, s2, _ = lx.search(q, K=8)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(s1, s2, rtol=1e-6)


def test_save_overwrites_existing(tmp_path):
    Y, q = _rand()
    ix = FaginIndex.build(Y)
    ix.save(tmp_path / "idx")
    ix.save(tmp_path / "idx")            # second save must not fail
    lx = FaginIndex.load(tmp_path / "idx")
    assert lx.n_vectors == 200


def test_build_rejects_bad_input():
    with pytest.raises(ValueError, match="2-D"):
        FaginIndex.build(np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError, match="N must be >= 1"):
        FaginIndex.build(np.zeros((0, 4), dtype=np.float32))


def test_search_rejects_bad_q():
    Y, _ = _rand()
    ix = FaginIndex.build(Y)
    with pytest.raises(ValueError, match="shape"):
        ix.search(np.zeros(3, dtype=np.float32), K=5)


def test_retrieval_arguments_have_fagin_fields():
    from docuverse.engines.search_engine_config_params import RetrievalArguments
    cfg = RetrievalArguments()
    assert cfg.fagin_batch_rows == 64
    assert cfg.fagin_epsilon == 0.0
    assert cfg.fagin_max_depth == 0
    assert cfg.fagin_num_threads == 0
