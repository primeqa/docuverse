"""Tests for the simdq affine Standardizer (fit/apply, persistence, index wiring)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from docuverse.engines.retrieval.simdq import SimdqIndex
from docuverse.engines.retrieval.simdq.standardize import Standardizer
from docuverse.utils.isotropy import mean_offset, sign_imbalance, n_dead_dims


def _anisotropic(n: int, d: int, seed: int) -> np.ndarray:
    """Gaussian with a mean offset, an inflated block, and one dead dim."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, d)).astype(np.float32)
    X[:, :8] *= 12.0          # anisotropy
    X[:, 0] += 30.0           # rogue always-positive dim + global mean offset
    X[:, 5] = 0.7             # dead (constant) dim
    return X


def test_fit_apply_centers_and_standardizes():
    X = _anisotropic(4000, 768, seed=0)
    std = Standardizer.fit(X)
    Z = std.apply(X)

    # mean ~ 0 and unit variance on live dims
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-4)
    live = ~std.dead_mask
    assert np.allclose(Z.std(axis=0)[live], 1.0, atol=1e-3)

    # the cheap isotropy metrics recover
    assert mean_offset(Z) < 1e-3
    assert sign_imbalance(Z) < 0.10        # excludes the dead dim


def test_dead_dim_is_zeroed_not_saturated():
    X = _anisotropic(2000, 768, seed=1)
    std = Standardizer.fit(X)
    assert std.n_dead == 1
    assert std.dead_mask[5]
    Z = std.apply(X)
    # dead dim maps to exactly 0 (neutral), not a constant ±1 bit
    assert np.all(Z[:, 5] == 0.0)
    assert n_dead_dims(Z) == 1


def test_apply_single_vector():
    X = _anisotropic(1000, 768, seed=2)
    std = Standardizer.fit(X)
    one = std.apply(X[0])
    assert one.shape == (768,)
    assert np.array_equal(one, std.apply(X[:1])[0])


def test_save_load_round_trip(tmp_path: Path):
    X = _anisotropic(1000, 768, seed=3)
    std = Standardizer.fit(X)
    p = tmp_path / "standardize.npz"
    std.save(p)
    loaded = Standardizer.load(p)
    assert np.array_equal(loaded.mu, std.mu)
    assert np.array_equal(loaded.inv_sigma, std.inv_sigma)
    assert np.array_equal(loaded.dead_mask, std.dead_mask)
    assert np.array_equal(loaded.apply(X), std.apply(X))


def test_index_persists_and_applies_standardizer(tmp_path: Path):
    X = _anisotropic(1024, 768, seed=4)
    idx = SimdqIndex.build(vectors=X, b=2, projection="identity", standardize=True)
    out = tmp_path / "idx"
    idx.save(out)

    assert (out / "standardize.npz").exists()
    import json
    meta = json.loads((out / "meta.json").read_text())
    assert meta["standardize"] is True

    loaded = SimdqIndex.load(out)
    assert loaded.standardizer is not None
    assert np.array_equal(loaded.standardizer.mu, idx.standardizer.mu)

    # a query gets the same transform applied as the corpus did
    q = X[0]
    idxs, scores = loaded.search(q, K=5, K_prime=50)
    assert idxs[0] == 0          # a vector is its own nearest neighbour


def test_standardize_default_off_is_backward_compatible(tmp_path: Path):
    X = _anisotropic(512, 768, seed=5)
    idx = SimdqIndex.build(vectors=X, b=2, projection="identity")  # no standardize
    out = tmp_path / "idx"
    idx.save(out)
    assert not (out / "standardize.npz").exists()
    import json
    meta = json.loads((out / "meta.json").read_text())
    assert meta["standardize"] is False
    loaded = SimdqIndex.load(out)
    assert loaded.standardizer is None
