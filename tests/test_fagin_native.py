"""Kernel-level tests for _simdq_native.fagin_search (Fagin's Threshold
Algorithm, exact top-K inner product over per-dimension sorted lists)."""
import numpy as np
import pytest

_native = pytest.importorskip("docuverse.engines.retrieval.simdq._simdq_native")


def _search(Y, q, K, batch=8, epsilon=0.0, max_depth=0, num_threads=1):
    """Build sorted lists in NumPy (reference layout) and call the kernel."""
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    perm = np.argsort(-Y, axis=0, kind="stable")           # (N, D), descending
    order = np.ascontiguousarray(perm.T, dtype=np.int32)   # (D, N)
    vals = np.ascontiguousarray(
        np.take_along_axis(Y, perm, axis=0).T, dtype=np.float32)  # (D, N)
    scores_b, idx_b, stats = _native.fagin_search(
        Y, order, vals, q, int(K), int(batch), float(epsilon),
        int(max_depth), int(num_threads))
    scores = np.frombuffer(scores_b, dtype=np.float32).copy()
    idxs = np.frombuffer(idx_b, dtype=np.int64).copy()
    return idxs, scores, stats


def _check_exact_topk(Y, q, idxs, scores, K):
    """Exactness contract, robust to ties and fp32 accumulation order:
    every returned doc scores >= the true kth score, returned score values
    are the true scores of the returned ids, and the returned score multiset
    matches the brute-force top-K score multiset."""
    s = Y.astype(np.float64) @ q.astype(np.float64)
    n_expect = min(K, len(s))
    desc = np.sort(s)[::-1]
    kth = desc[n_expect - 1]
    valid = idxs[:n_expect]
    assert np.all(valid >= 0)
    assert np.all(idxs[n_expect:] == -1)
    assert len(np.unique(valid)) == n_expect
    assert np.all(s[valid] >= kth - 1e-4)
    np.testing.assert_allclose(scores[:n_expect], s[valid], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.sort(scores[:n_expect])[::-1], desc[:n_expect],
                               rtol=1e-4, atol=1e-5)
    assert np.all(np.diff(scores[:n_expect]) <= 1e-6)  # descending


def test_exact_topk_matches_brute_force():
    rng = np.random.default_rng(42)
    Y = rng.standard_normal((500, 32))
    q = rng.standard_normal(32)
    idxs, scores, stats = _search(Y, q, K=10)
    _check_exact_topk(Y, q, idxs, scores, 10)
    assert stats["depth"] >= 1
    assert stats["random_accesses"] <= 500


def test_negative_and_zero_query_weights():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((400, 24))
    q = rng.standard_normal(24)
    q[::3] = -np.abs(q[::3])       # force negatives (bottom-up scans)
    q[1::5] = 0.0                  # force skipped dims
    idxs, scores, _ = _search(Y, q, K=10)
    _check_exact_topk(Y, q, idxs, scores, 10)


def test_all_negative_query_and_negative_documents():
    rng = np.random.default_rng(1)
    Y = -np.abs(rng.standard_normal((300, 16)))
    q = -np.abs(rng.standard_normal(16))
    idxs, scores, _ = _search(Y, q, K=7)
    _check_exact_topk(Y, q, idxs, scores, 7)


def test_d_not_multiple_of_8_and_odd_batch():
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((50, 7))
    q = rng.standard_normal(7)
    idxs, scores, _ = _search(Y, q, K=5, batch=3)
    _check_exact_topk(Y, q, idxs, scores, 5)


def test_k_equals_n_and_k_greater_than_n():
    rng = np.random.default_rng(3)
    Y = rng.standard_normal((20, 8))
    q = rng.standard_normal(8)
    idxs, scores, _ = _search(Y, q, K=20)
    _check_exact_topk(Y, q, idxs, scores, 20)
    idxs, scores, _ = _search(Y, q, K=32)     # K > N: 20 valid + 12 padding
    _check_exact_topk(Y, q, idxs, scores, 32)


def test_single_document():
    Y = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    q = np.array([0.5, 1.0, -1.0], dtype=np.float32)
    idxs, scores, _ = _search(Y, q, K=1)
    assert idxs[0] == 0
    np.testing.assert_allclose(scores[0], -4.5, rtol=1e-6)


def test_duplicate_rows_ties():
    rng = np.random.default_rng(4)
    row = rng.standard_normal(12)
    Y = np.vstack([row] * 10 + [rng.standard_normal((30, 12))])
    q = rng.standard_normal(12)
    idxs, scores, _ = _search(Y, q, K=15)
    _check_exact_topk(Y, q, idxs, scores, 15)


def test_all_zero_query():
    rng = np.random.default_rng(5)
    Y = rng.standard_normal((40, 6))
    q = np.zeros(6)
    idxs, scores, stats = _search(Y, q, K=5)
    assert np.all(scores[:5] == 0.0)
    assert sorted(idxs[:5].tolist()) == [0, 1, 2, 3, 4]
    assert stats["sorted_accesses"] == 0 and stats["exhausted"] == 1


def test_epsilon_stops_earlier_and_is_bounded():
    rng = np.random.default_rng(6)
    Y = rng.standard_normal((2000, 48))
    q = rng.standard_normal(48)
    _, exact_scores, exact_stats = _search(Y, q, K=10, epsilon=0.0)
    idxs, scores, stats = _search(Y, q, K=10, epsilon=5.0)
    assert stats["depth"] <= exact_stats["depth"]
    s = Y.astype(np.float64) @ q.astype(np.float64)
    brute_kth = np.sort(s)[::-1][9]
    # Guarantee: returned kth is within epsilon of the true kth.
    assert scores[9] >= brute_kth - 5.0 - 1e-4
    # Returned scores are still the true scores of the returned ids.
    np.testing.assert_allclose(scores[:10], s[idxs[:10]], rtol=1e-4, atol=1e-5)


def test_max_depth_caps_depth():
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((1000, 32))
    q = rng.standard_normal(32)
    idxs, scores, stats = _search(Y, q, K=10, batch=8, max_depth=20)
    assert stats["depth"] <= 20
    valid = idxs[idxs >= 0]
    assert len(np.unique(valid)) == len(valid)
    assert np.all(np.diff(scores[:len(valid)]) <= 1e-6)


def test_stats_consistency():
    rng = np.random.default_rng(8)
    Y = rng.standard_normal((600, 20))
    q = rng.standard_normal(20)
    q[3] = 0.0
    _, _, stats = _search(Y, q, K=10, batch=16)
    n_active = int(np.count_nonzero(q))
    assert stats["sorted_accesses"] == n_active * stats["depth"]
    assert stats["random_accesses"] <= min(600, stats["sorted_accesses"])
    assert stats["rounds"] >= 1
    assert stats["depth"] <= 600


def test_num_threads_invariance():
    rng = np.random.default_rng(9)
    Y = rng.standard_normal((800, 40))
    q = rng.standard_normal(40)
    i1, s1, _ = _search(Y, q, K=10, num_threads=1)
    i4, s4, _ = _search(Y, q, K=10, num_threads=4)
    np.testing.assert_allclose(s1, s4, rtol=1e-6)
    _check_exact_topk(Y, q, i4, s4, 10)


def test_input_validation():
    rng = np.random.default_rng(10)
    Y = np.ascontiguousarray(rng.standard_normal((10, 4)), dtype=np.float32)
    perm = np.argsort(-Y, axis=0, kind="stable")
    order = np.ascontiguousarray(perm.T, dtype=np.int32)
    vals = np.ascontiguousarray(np.take_along_axis(Y, perm, axis=0).T,
                                dtype=np.float32)
    q = np.ascontiguousarray(rng.standard_normal(4), dtype=np.float32)
    with pytest.raises(ValueError):
        _native.fagin_search(Y, order, vals, q, 0, 8, 0.0, 0, 1)   # K < 1
    with pytest.raises(ValueError):
        _native.fagin_search(Y, order, vals, q, 5, 0, 0.0, 0, 1)   # batch < 1
    with pytest.raises(ValueError):
        _native.fagin_search(Y, order, vals, q[:3], 5, 8, 0.0, 0, 1)  # bad q shape
    with pytest.raises(ValueError):
        # .copy() keeps it C-contiguous so the shape check (not the buffer
        # contiguity check) is what rejects it
        _native.fagin_search(Y, order[:, :5].copy(), vals, q, 5, 8, 0.0, 0, 1)
    with pytest.raises(TypeError):
        _native.fagin_search(Y.astype(np.float64), order, vals, q, 5, 8, 0.0, 0, 1)
