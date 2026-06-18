"""Tests for docuverse.engines.retrieval.simdq.projection."""
from __future__ import annotations

import numpy as np
import pytest

from docuverse.engines.retrieval.simdq.projection import (
    apply_projection,
    identity,
    random_orthogonal,
)


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
