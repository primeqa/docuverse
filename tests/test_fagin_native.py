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
