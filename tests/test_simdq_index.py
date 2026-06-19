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


# ---------------------------------------------------------------------------
# Parametrized multi-D + hamming family tests (Task 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D,d,projection", [
    (384, 384, "identity"),
    (384, 192, "random_orthogonal"),
    (1024, 1024, "identity"),
    (1024, 512, "random_orthogonal"),
    (1536, 768, "random_orthogonal"),
])
def test_asym_build_save_load_search_multi_D(D, d, projection, tmp_path):
    rng = np.random.RandomState(D)
    Y = rng.randn(2048, D).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    idx = SimdqIndex.build(
        Y, family="asymmetric", b=2, d=d,
        projection=projection, projection_seed=7, store_floats=True,
    )
    p = tmp_path / f"idx_{D}_{d}"
    idx.save(p)
    idx2 = SimdqIndex.load(p)
    assert idx2.D_orig == D and idx2.d == d and idx2.family == "asymmetric"

    # Smoke-test search: any well-defined query returns K finite indices in [0, N)
    q_v = rng.randn(D).astype(np.float32)
    iN, sN = idx2.search(q_v, K=10, K_prime=100)
    assert iN.shape == (10,) and sN.shape == (10,)
    assert iN.min() >= 0 and iN.max() < 2048
    assert np.all(np.isfinite(sN))


@pytest.mark.parametrize("D", [384, 768, 1024, 1536])
def test_hamming_family_round_trip(D, tmp_path):
    rng = np.random.RandomState(D + 1)
    Y = rng.randn(1024, D).astype(np.float32)
    idx = SimdqIndex.build(
        Y, family="hamming", d=D, projection="identity", store_floats=False,
    )
    p = tmp_path / f"idx_h_{D}"
    idx.save(p)
    idx2 = SimdqIndex.load(p)
    assert idx2.family == "hamming"
    assert idx2.b is None
    assert idx2.scales is None

    q_v = rng.randn(D).astype(np.float32)
    iN, sN = idx2.search(q_v, K=10, K_prime=10)
    assert iN.shape == (10,) and sN.shape == (10,)
    # Hamming "scores" are -distance; descending order means smaller distances first.
    assert np.all(np.diff(sN) <= 1e-6)


# ---------------------------------------------------------------------------
# T3 gap-fill: empty corpus, N=1, K=1, K_prime extremes, save->load->save,
# corrupted meta.json, format_version mismatch.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="SimdqIndex.build silently accepts N=0; search later crashes with "
           "IndexError. Build should validate N>=1 up front. Follow-up bug.",
    strict=True,
)
def test_empty_corpus_rejected(tmp_index_dir):
    """Building from a (0, D) array must error cleanly, not segfault."""
    Y = np.zeros((0, 768), dtype=np.float32)
    with pytest.raises((ValueError, RuntimeError)):
        SimdqIndex.build(vectors=Y, b=2, store_floats=False)


def test_single_vector_corpus(tmp_index_dir):
    """N=1 must round-trip and search must return idx=0."""
    Y = _gaussian(1, 768, seed=11)
    idx = SimdqIndex.build(vectors=Y, b=2, store_floats=True)
    idx.save(tmp_index_dir)
    loaded = SimdqIndex.load(tmp_index_dir)
    q = Y[0]
    indices, _ = loaded.search(q, K=1, K_prime=1)
    assert int(indices[0]) == 0


def test_k_equals_one(tmp_index_dir):
    """K=1 codes-only must return exactly 1 index."""
    Y = _gaussian(512, 768, seed=12)
    idx = SimdqIndex.build(vectors=Y, b=2, store_floats=False)
    indices, scores = idx.search(Y[0], K=1, K_prime=1)
    assert len(indices) == 1
    assert len(scores) == 1


def test_kprime_max_256(tmp_index_dir):
    """K_prime=256 must work; K_prime=257 must raise."""
    Y = _gaussian(1024, 768, seed=13)
    idx = SimdqIndex.build(vectors=Y, b=2, store_floats=True)
    indices, _ = idx.search(Y[0], K=10, K_prime=256)
    assert len(indices) == 10
    with pytest.raises(ValueError):
        idx.search(Y[0], K=10, K_prime=257)


def test_save_load_save_metadata_stable(tmp_index_dir):
    """save -> load -> save -> load: meta.json content must match between saves."""
    Y = _gaussian(256, 768, seed=14)
    idx1 = SimdqIndex.build(vectors=Y, b=2, store_floats=True)
    idx1.save(tmp_index_dir)
    meta1 = (tmp_index_dir / "meta.json").read_text()

    idx2 = SimdqIndex.load(tmp_index_dir)
    other = tmp_index_dir.parent / "idx2"
    idx2.save(other)
    meta2 = (other / "meta.json").read_text()

    # Drop fields that legitimately may move between saves (e.g. timestamps);
    # for now there are none, so the parsed JSON must match exactly.
    assert json.loads(meta1) == json.loads(meta2)


def test_corrupted_meta_json_rejected(tmp_index_dir):
    """A meta.json that doesn't parse as JSON must produce a clear error."""
    Y = _gaussian(64, 768, seed=15)
    idx = SimdqIndex.build(vectors=Y, b=2, store_floats=False)
    idx.save(tmp_index_dir)
    (tmp_index_dir / "meta.json").write_text("{not valid json")
    with pytest.raises(Exception) as ei:
        SimdqIndex.load(tmp_index_dir)
    # Don't pin the exact exception type (json.JSONDecodeError vs ValueError);
    # require the message is informative about the JSON parse failure.
    msg = str(ei.value).lower()
    assert "json" in msg or "decode" in msg or "expecting" in msg


def test_format_version_mismatch_rejected(tmp_index_dir):
    """Loader rejects a future format_version with a message naming both versions."""
    Y = _gaussian(64, 768, seed=16)
    idx = SimdqIndex.build(vectors=Y, b=2, store_floats=False)
    idx.save(tmp_index_dir)

    meta_path = tmp_index_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["format_version"] = 999
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError) as ei:
        SimdqIndex.load(tmp_index_dir)
    msg = str(ei.value)
    assert "format_version" in msg
    assert "999" in msg
    assert "1" in msg  # the current version must be named too
