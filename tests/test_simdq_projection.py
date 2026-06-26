"""Tests for docuverse.engines.retrieval.simdq.projection."""
from __future__ import annotations

import numpy as np
import pytest

from docuverse.engines.retrieval.simdq.projection import (
    apply_projection,
    identity,
    learned_orthogonal,
    random_orthogonal,
)


def _anisotropic(n, d, seed):
    """Zero-mean but anisotropic + correlated data (the regime ITQ targets)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, :8] *= 10.0                      # variance concentrated in a few dims
    A = rng.standard_normal((d, d)).astype(np.float32)
    X = X @ A                             # correlate the dimensions
    return (X - X.mean(axis=0)).astype(np.float32)


def _quant_error(X, W):
    """Mean squared binary quantization error of sign(X W^T)."""
    Y = X @ W.T
    return float(np.mean((np.sign(Y) - Y) ** 2))


def test_identity_shape_and_values():
    W = identity(D=768)
    assert W.shape == (768, 768)
    assert W.dtype == np.float32
    # diagonal is 1, off-diagonal is 0
    assert np.allclose(W, np.eye(768, dtype=np.float32))


def test_random_orthogonal_shape_and_seed():
    W1 = random_orthogonal(D=768, d=384, seed=42)
    W2 = random_orthogonal(D=768, d=384, seed=42)
    W3 = random_orthogonal(D=768, d=384, seed=7)
    assert W1.shape == (384, 768)
    assert W1.dtype == np.float32
    # determinism for the same seed
    assert np.array_equal(W1, W2)
    # different seed -> different matrix
    assert not np.array_equal(W1, W3)


def test_random_orthogonal_rows_are_orthonormal():
    W = random_orthogonal(D=768, d=384, seed=123)
    # rows are unit-norm
    norms = np.linalg.norm(W, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # rows are pairwise orthogonal -> W @ W.T = I_d
    gram = W @ W.T
    assert np.allclose(gram, np.eye(384, dtype=np.float32), atol=1e-4)


def test_random_orthogonal_full_dim():
    # d == D path also works: W is 768x768 orthogonal
    W = random_orthogonal(D=768, d=768, seed=0)
    assert W.shape == (768, 768)
    gram = W @ W.T
    assert np.allclose(gram, np.eye(768, dtype=np.float32), atol=1e-4)


def test_apply_projection_identity():
    Y = np.arange(2 * 768, dtype=np.float32).reshape(2, 768)
    W = identity(768)
    Z = apply_projection(Y, W)
    assert Z.shape == (2, 768)
    assert np.array_equal(Y, Z)


def test_apply_projection_random_orthogonal_preserves_norm():
    rng = np.random.default_rng(11)
    Y = rng.normal(size=(4, 768)).astype(np.float32)
    W = random_orthogonal(D=768, d=768, seed=99)
    Z = apply_projection(Y, W)
    # full-rank orthogonal projection preserves norms
    yn = np.linalg.norm(Y, axis=1)
    zn = np.linalg.norm(Z, axis=1)
    assert np.allclose(yn, zn, atol=1e-4)


def test_random_orthogonal_rejects_invalid_d():
    with pytest.raises(ValueError):
        random_orthogonal(D=768, d=128, seed=0)  # not in {D, D/2}


# ---------------------------------------------------------------------------
# Parametrized multi-D tests (Task 6)
# ---------------------------------------------------------------------------

from docuverse.engines.retrieval.simdq.projection import (  # noqa: E402
    validate_D_d, SUPPORTED_D, SUPPORTED_d,
)


@pytest.mark.parametrize("D,d", [
    (384, 384), (384, 192),
    (768, 768), (768, 384),
    (1024, 1024), (1024, 512),
    (1536, 1536), (1536, 768),
])
def test_random_orthogonal_shape_and_orthonormal_rows(D, d):
    W = random_orthogonal(D, d, seed=42)
    assert W.shape == (d, D)
    assert W.dtype == np.float32
    # Rows are orthonormal: W @ W.T == I_d (within fp32 tolerance)
    gram = W @ W.T
    assert np.allclose(gram, np.eye(d, dtype=np.float32), atol=1e-4)


@pytest.mark.parametrize("D,d", [
    (384, 384), (768, 768),
])
def test_apply_projection_norms_preserved_at_d_eq_D(D, d):
    """For d == D, random_orthogonal preserves norms (square Q)."""
    W = random_orthogonal(D, d, seed=0)
    X = np.random.RandomState(1).randn(50, D).astype(np.float32)
    Y = apply_projection(X, W)
    n_x = np.linalg.norm(X, axis=1)
    n_y = np.linalg.norm(Y, axis=1)
    assert np.allclose(n_x, n_y, rtol=1e-4)


def test_validate_D_d_rejects_unsupported():
    with pytest.raises(ValueError, match="D must be one of"):
        validate_D_d(512, 512)
    with pytest.raises(ValueError, match="d must be"):
        validate_D_d(768, 256)
    with pytest.raises(ValueError, match="d must be"):
        validate_D_d(384, 768)


# ---------------------------------------------------------------------------
# T3 gap-fill: idempotence (W @ W.T == I_d) + cross-process seed determinism
# ---------------------------------------------------------------------------

import subprocess  # noqa: E402
import sys  # noqa: E402


@pytest.mark.parametrize("D, d", [
    (384, 384), (768, 768), (1024, 1024), (1536, 1536),
    (768, 384), (1024, 512), (1536, 768),
])
def test_random_orthogonal_idempotence(D, d):
    """W (d x D) from random_orthogonal must satisfy W @ W.T == I_d to 1e-5.

    This is the Johnson-Lindenstrauss-flavored property the spec relies on:
    rows of W are an orthonormal frame in R^D restricted to a d-dim subspace.
    """
    W = random_orthogonal(D=D, d=d, seed=42)
    assert W.shape == (d, D)
    gram = W @ W.T
    np.testing.assert_allclose(gram, np.eye(d, dtype=W.dtype), atol=1e-5)


def test_random_orthogonal_seed_deterministic_cross_process():
    """Same seed in two separate Python processes must produce byte-identical W.

    Catches RNG state leak through environment / module-level random state.
    """
    cmd = [
        sys.executable, "-c",
        "import sys; "
        "from docuverse.engines.retrieval.simdq.projection import random_orthogonal; "
        "W = random_orthogonal(D=768, d=384, seed=42); "
        "sys.stdout.buffer.write(W.tobytes())",
    ]
    out1 = subprocess.check_output(cmd)
    out2 = subprocess.check_output(cmd)
    assert out1 == out2, "random_orthogonal not deterministic across processes"
    # Also assert deterministic vs an in-process call.
    W_inproc = random_orthogonal(D=768, d=384, seed=42)
    assert out1 == W_inproc.tobytes()


# ---------------------------------------------------------------------------
# ITQ (learned_orthogonal)
# ---------------------------------------------------------------------------

class TestLearnedOrthogonal:
    def test_shape_and_orthonormal_rows_full_dim(self):
        X = _anisotropic(2000, 256, seed=0)
        W = learned_orthogonal(X, d=256, seed=42, n_iters=30)
        assert W.shape == (256, 256)
        assert W.dtype == np.float32
        # rows orthonormal: W Wᵀ = I_d
        assert np.allclose(W @ W.T, np.eye(256), atol=1e-3)

    def test_shape_and_orthonormal_rows_half_dim(self):
        X = _anisotropic(2000, 256, seed=1)
        W = learned_orthogonal(X, d=128, seed=42, n_iters=30)
        assert W.shape == (128, 256)
        assert np.allclose(W @ W.T, np.eye(128), atol=1e-3)

    def test_deterministic_for_same_seed(self):
        X = _anisotropic(1500, 256, seed=2)
        W1 = learned_orthogonal(X, d=256, seed=42, n_iters=20)
        W2 = learned_orthogonal(X, d=256, seed=42, n_iters=20)
        assert np.array_equal(W1, W2)

    def test_reduces_quantization_error_vs_random_init(self):
        # ITQ monotonically lowers the binary quant error; 50 iters must beat the
        # random rotation it started from (n_iters=0, same seed -> same init).
        X = _anisotropic(3000, 256, seed=3)
        W0 = learned_orthogonal(X, d=256, seed=42, n_iters=0)
        W50 = learned_orthogonal(X, d=256, seed=42, n_iters=50)
        assert _quant_error(X, W50) < _quant_error(X, W0)

    def test_beats_random_orthogonal_quant_error(self):
        # The whole point: a learned rotation is a better SimHash basis than a
        # data-independent random one on anisotropic data.
        X = _anisotropic(3000, 256, seed=4)
        W_itq = learned_orthogonal(X, d=256, seed=42, n_iters=50)
        W_rand = random_orthogonal(D=256, d=256, seed=42)
        assert _quant_error(X, W_itq) < _quant_error(X, W_rand)

    def test_rejects_bad_d(self):
        X = _anisotropic(500, 256, seed=5)
        with pytest.raises(ValueError):
            learned_orthogonal(X, d=200, seed=42)      # not D or D/2

    def test_rejects_too_few_samples(self):
        X = _anisotropic(100, 256, seed=6)             # N < d
        with pytest.raises(ValueError):
            learned_orthogonal(X, d=256, seed=42)

    def test_subsampling_caps_fit_rows(self):
        # max_fit_samples < N must still produce a valid orthonormal W.
        X = _anisotropic(4000, 128, seed=7)
        W = learned_orthogonal(X, d=128, seed=42, n_iters=20, max_fit_samples=1000)
        assert W.shape == (128, 128)
        assert np.allclose(W @ W.T, np.eye(128), atol=1e-3)
