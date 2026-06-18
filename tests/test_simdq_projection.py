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
