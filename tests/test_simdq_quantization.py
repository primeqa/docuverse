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
    Y = _gaussian(8, 500, seed=5)  # 500 not in SUPPORTED_d -> rejected
    with pytest.raises(ValueError):
        fit_scales(Y)


# ---------------------------------------------------------------------------
# Parametrized multi-d tests (Task 6)
# ---------------------------------------------------------------------------

import docuverse.engines.retrieval.simdq.quantization as q  # noqa: E402
from docuverse.engines.retrieval.simdq.projection import SUPPORTED_d  # noqa: E402


@pytest.mark.parametrize("d", SUPPORTED_d)
@pytest.mark.parametrize("b", [1, 2, 4])
def test_pack_round_trip_at_each_d(d, b):
    rng = np.random.RandomState(d * 100 + b)
    Y = rng.randn(64, d).astype(np.float32)
    scales = q.fit_scales(Y)
    codes = q.pack(Y, scales, b=b)
    levels = q.unpack_levels(codes, N=64, D=d, b=b)
    # Reconstruct quantized Y and check signs match (asymmetric ranks robustly to
    # quantization at the b=2 level).
    s = scales.reshape(-1, 1).astype(np.float32)
    yhat = levels.astype(np.float32) * s / float(2 ** b - 1) * np.sqrt(d)
    # Spot-check: sign agreement on at least 70% of dims (a generous floor).
    sign_match = (np.sign(yhat) == np.sign(Y)).mean()
    assert sign_match > 0.70, f"sign agreement only {sign_match:.2%} at d={d} b={b}"


@pytest.mark.parametrize("d", [384, 768, 1024, 1536])
def test_pack_hamming_round_trip(d):
    rng = np.random.RandomState(d)
    Y = rng.randn(32, d).astype(np.float32)
    codes = q.pack_hamming(Y)
    assert codes.dtype == np.uint8
    assert codes.size == 32 * d // 8
    # Unpack one row and confirm the sign matches.
    words = d // 64
    code0 = codes[:words * 8].view(np.uint64)
    for w in range(words):
        for b in range(64):
            bit = bool((code0[w] >> np.uint64(b)) & np.uint64(1))
            assert bit == (Y[0, w * 64 + b] >= 0.0), \
                f"bit mismatch at w={w} b={b}"
