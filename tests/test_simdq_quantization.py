"""Round-trip + scale tests for simdq quantization wrappers."""
from __future__ import annotations

import numpy as np
import pytest

from docuverse.engines.retrieval.simdq.quantization import (
    fit_scales,
    pack,
    unpack_levels,
)


def _gaussian(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n, d)).astype(np.float32)


def test_fit_scales_shape_and_values():
    Y = _gaussian(64, 768, seed=0)
    s = fit_scales(Y)
    assert s.shape == (64,)
    assert s.dtype == np.float32
    # ||y||/sqrt(d), no zeros for a Gaussian sample
    expected = np.linalg.norm(Y, axis=1) / np.sqrt(768)
    assert np.allclose(s, expected, atol=1e-5)


def test_pack_b1_round_trip():
    Y = _gaussian(32, 768, seed=1)
    s = fit_scales(Y)
    codes = pack(Y, s, b=1)
    # codes is dim-major SoA: D rows of ceil(N/8) bytes
    assert codes.shape == (768 * ((32 + 7) // 8),)
    assert codes.dtype == np.uint8
    levels = unpack_levels(codes, N=32, D=768, b=1)
    # each entry is +1 or -1
    assert set(np.unique(levels).tolist()).issubset({-1, 1})
    # round-trip: sign of (Y / s[:, None]) matches levels
    Y_norm = Y / s[:, None]
    expected = np.where(Y_norm >= 0, 1, -1).astype(np.int8)
    assert np.array_equal(levels, expected)


def test_pack_b2_round_trip():
    Y = _gaussian(40, 768, seed=2) * 2.0  # widen distribution into b=2 levels
    s = fit_scales(Y)
    codes = pack(Y, s, b=2)
    assert codes.shape == (768 * ((40 + 3) // 4),)
    levels = unpack_levels(codes, N=40, D=768, b=2)
    assert set(np.unique(levels).tolist()).issubset({-3, -1, 1, 3})
    # Compare against the boundary rule used by simdq_pack_b2:
    # yn < -2 -> -3; yn < 0 -> -1; yn < 2 -> +1; else +3.
    Y_norm = Y / s[:, None]
    boundaries = np.full(Y_norm.shape, 3, dtype=np.int8)
    boundaries[Y_norm <  2.0] =  1
    boundaries[Y_norm <  0.0] = -1
    boundaries[Y_norm < -2.0] = -3
    assert np.array_equal(levels, boundaries)


def test_pack_b4_round_trip():
    Y = _gaussian(20, 768, seed=3) * 8.0
    s = fit_scales(Y)
    codes = pack(Y, s, b=4)
    assert codes.shape == (768 * ((20 + 1) // 2),)
    levels = unpack_levels(codes, N=20, D=768, b=4)
    # 16 levels, evenly spaced from -15 to +15, step 2
    assert set(np.unique(levels).tolist()).issubset(set(range(-15, 16, 2)))


def test_pack_rejects_bad_b():
    Y = _gaussian(8, 768, seed=4)
    s = fit_scales(Y)
    with pytest.raises(ValueError):
        pack(Y, s, b=3)


def test_pack_rejects_bad_dim():
    Y = _gaussian(8, 384, seed=5)  # D != 768 -> rejected in Plan 2
    with pytest.raises(ValueError):
        fit_scales(Y)
