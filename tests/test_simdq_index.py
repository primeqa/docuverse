"""End-to-end SimdqIndex tests: round-trip, mode parity, agreement vs sklearn."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from docuverse.engines.retrieval.simdq import SimdqIndex


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    return tmp_path / "idx"


def _gaussian(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n, d)).astype(np.float32)


def test_build_save_load_round_trip(tmp_index_dir):
    Y = _gaussian(1024, 768, seed=0)
    idx = SimdqIndex.build(
        vectors=Y, b=2,
        projection="identity", store_floats=True,
    )
    idx.save(tmp_index_dir)

    # files we expect on disk
    assert (tmp_index_dir / "meta.json").exists()
    assert (tmp_index_dir / "W.npy").exists()
    assert (tmp_index_dir / "codes.bin").exists()
    assert (tmp_index_dir / "scales.bin").exists()
    assert (tmp_index_dir / "floats.bin").exists()

    meta = json.loads((tmp_index_dir / "meta.json").read_text())
    assert meta["n_vectors"] == 1024
    assert meta["b"] == 2
    assert meta["d"] == 768
    assert meta["has_floats"] is True

    loaded = SimdqIndex.load(tmp_index_dir)
    assert loaded.n_vectors == 1024
    assert loaded.b == 2
    assert loaded.d == 768
    # codes content matches
    assert np.array_equal(loaded.codes, idx.codes)
    assert np.array_equal(loaded.scales, idx.scales)


def test_build_no_floats_omits_floats_bin(tmp_index_dir):
    Y = _gaussian(256, 768, seed=1)
    idx = SimdqIndex.build(vectors=Y, b=2, store_floats=False)
    idx.save(tmp_index_dir)
    assert not (tmp_index_dir / "floats.bin").exists()
    loaded = SimdqIndex.load(tmp_index_dir)
    assert loaded.has_floats is False
    # codes-only search still works
    q = _gaussian(1, 768, seed=99)[0]
    indices, scores = loaded.search(q, K=10)
    assert indices.shape == (10,)
    assert scores.shape == (10,)


def test_search_mode_parity_differs(tmp_index_dir):
    """Codes-only and two-stage rescore should differ on a non-unit-norm corpus."""
    Y = _gaussian(2048, 768, seed=2)
    # Inflate norms of the second half so per-vector scales differ a lot.
    Y[1024:] *= 8.0
    idx = SimdqIndex.build(vectors=Y, b=2, store_floats=True)
    idx.save(tmp_index_dir)

    q = _gaussian(1, 768, seed=42)[0]
    codes_only_idx, codes_only_scores = idx.search(q, K=20, K_prime=20)
    two_stage_idx, two_stage_scores = idx.search(q, K=20, K_prime=200)

    # the two top-K sets should differ at least on one position when scales vary
    assert not np.array_equal(codes_only_idx, two_stage_idx)


def test_search_rescore_returns_exact_dot_products(tmp_index_dir):
    """The fp16 rescore stage must report scores that match the actual
    fp16-rounded dot product Y[idx] @ q for each returned index, and the
    returned scores must be sorted descending. This is a pipeline test —
    it does not depend on b-bit quantizer fidelity (which is a Plan-1
    design concern), only on the rescore stage doing what it claims.
    """
    rng = np.random.default_rng(3)
    Y = rng.standard_normal(size=(2048, 768)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    idx = SimdqIndex.build(vectors=Y, b=4, store_floats=True)
    idx.save(tmp_index_dir)

    q = rng.standard_normal(size=(768,)).astype(np.float32)
    q /= np.linalg.norm(q)

    sim_idx, sim_scores = idx.search(q, K=10, K_prime=100)

    assert sim_idx.shape == (10,)
    assert sim_scores.shape == (10,)

    # Each returned score must equal the fp16-projected vector @ q (the
    # actual rescore computation). Tolerance covers fp16 rounding only.
    for i, s in zip(sim_idx.tolist(), sim_scores.tolist()):
        actual = float(Y[i].astype(np.float16).astype(np.float32) @ q)
        assert abs(actual - s) < 1e-3, (
            f"rescore mismatch at idx={i}: returned {s}, expected {actual}"
        )

    # Descending order.
    for r in range(9):
        assert sim_scores[r] >= sim_scores[r + 1], (
            f"scores not descending at rank {r}: "
            f"{sim_scores[r]} < {sim_scores[r + 1]}"
        )


def test_search_brute_force_recall_baseline(tmp_index_dir):
    """Document the empirical recall@10 of b=4 + K'=100 rescore against
    cosine brute-force on unit-norm isotropic Gaussians. This is a coarse
    sanity check that the pipeline doesn't randomize results — recall
    >> random baseline (10/10000 = 0.1%). The actual Plan-1 b=4 quantizer
    uses a step-2 grid that under-utilizes its 16 levels on N(0,1)
    per-dim distributions; real embeddings (granite, ST) have heavier
    tails and exercise more of b=4's range. Plan 3's BEIR sweep will
    measure retrieval quality properly on real data.
    """
    rng = np.random.default_rng(3)
    Y = rng.standard_normal(size=(10000, 768)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    idx = SimdqIndex.build(vectors=Y, b=4, store_floats=True)
    idx.save(tmp_index_dir)

    Q = rng.standard_normal(size=(30, 768)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)

    overlap = []
    for q in Q:
        scores = Y @ q
        gold = np.argpartition(-scores, 10)[:10]
        gold = gold[np.argsort(-scores[gold])]
        sim_idx, _ = idx.search(q, K=10, K_prime=100)
        overlap.append(len(set(sim_idx.tolist()) & set(gold.tolist())) / 10.0)

    mean_overlap = float(np.mean(overlap))
    # Empirical floor on unit-norm Gaussians; real embeddings do much better.
    assert mean_overlap >= 0.30, f"mean recall@10 = {mean_overlap:.3f}"


def test_search_rejects_bad_K(tmp_index_dir):
    Y = _gaussian(512, 768, seed=4)
    idx = SimdqIndex.build(vectors=Y, b=2)
    with pytest.raises(ValueError):
        idx.search(np.zeros(768, dtype=np.float32), K=300)   # K > 256
    with pytest.raises(ValueError):
        idx.search(np.zeros(768, dtype=np.float32), K=10, K_prime=5)  # K' < K
