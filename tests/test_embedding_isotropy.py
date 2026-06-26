"""
Check whether embedding vectors are approximately uniformly distributed on the
unit sphere — the empirical condition that makes sign(x) a useful SimHash code
without an explicit random-projection matrix W.

Four metrics are computed:

  sign_imbalance   max_d |P(x_d > 0) - 0.5|
                   0 = perfectly balanced per dimension.

  mean_offset      ||E[x]||_2 / sqrt(D)
                   0 = distribution centred at origin.

  cov_spectral_ratio   lambda_max / lambda_min of the sample covariance
                       1 = spherically symmetric; larger = anisotropic.

  effective_rank_ratio   exp(H(p)) / D,  p = s^2 / sum(s^2)
                         1 = all singular values equal; < 1 = concentrated.

The unit tests exercise each metric against known synthetic distributions.
The integration tests (marked `slow`) accept a numpy array of real embeddings
via the `embeddings` fixture and report whether the distribution is isotropic
enough for sign(x) to behave like a SimHash code.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def sign_imbalance(embeddings: np.ndarray) -> float:
    """Max |P(x_d > 0) - 0.5| over all dimensions d."""
    pos_frac = (embeddings > 0).mean(axis=0)
    return float(np.abs(pos_frac - 0.5).max())


def mean_offset(embeddings: np.ndarray) -> float:
    """||mean||_2 / sqrt(D): 0 for a zero-mean distribution."""
    D = embeddings.shape[1]
    return float(np.linalg.norm(embeddings.mean(axis=0)) / np.sqrt(D))


def cov_spectral_ratio(embeddings: np.ndarray) -> float:
    """lambda_max / lambda_min of the sample covariance matrix.

    1.0 = perfectly isotropic.  Covariance is computed on L2-normalised vectors
    so that scale differences don't mask directional anisotropy.
    """
    X = _l2_normalize(embeddings)
    # Use SVD of the centred matrix to get singular values efficiently.
    Xc = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    # singular values of Xc relate to eigenvalues of the covariance by s^2/N.
    lam = s ** 2
    return float(lam.max() / (lam.min() + 1e-12))


def effective_rank_ratio(embeddings: np.ndarray) -> float:
    """exp(H(p)) / D where p_i = s_i^2 / sum_j s_j^2.

    1.0 = all singular values equal (perfectly spread).
    Follows Roy & Vetterli (2007).
    """
    X = _l2_normalize(embeddings)
    Xc = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    D = len(s)
    p = s ** 2 / (s ** 2).sum()
    entropy = -float(np.sum(p * np.log(p + 1e-30)))
    return float(np.exp(entropy) / D)


def isotropy_report(embeddings: np.ndarray) -> dict[str, float]:
    """Return all four metrics as a dict."""
    return {
        "sign_imbalance":      sign_imbalance(embeddings),
        "mean_offset":         mean_offset(embeddings),
        "cov_spectral_ratio":  cov_spectral_ratio(embeddings),
        "effective_rank_ratio": effective_rank_ratio(embeddings),
    }


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_sphere(rng=None):
    """N=2000, D=64 vectors sampled uniformly from S^{D-1}."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2000, 64)).astype(np.float32)
    return _l2_normalize(X)


@pytest.fixture
def anisotropic(rng=None):
    """N=2000, D=64 vectors with variance concentrated in first 4 dims."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2000, 64)).astype(np.float32)
    # inflate first 4 dimensions by 20x
    X[:, :4] *= 20.0
    return _l2_normalize(X)


@pytest.fixture
def biased_mean(rng=None):
    """N=2000, D=64 vectors with a large mean offset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2000, 64)).astype(np.float32)
    X[:, 0] += 5.0          # strong bias in first dimension
    return _l2_normalize(X)


# ---------------------------------------------------------------------------
# Unit tests: metric correctness on known distributions
# ---------------------------------------------------------------------------

class TestSignImbalance:
    def test_uniform_sphere_is_balanced(self, uniform_sphere):
        # Each dimension should be ~50/50; allow 3 std devs ~ 3*sqrt(0.25/2000) ~ 0.034
        assert sign_imbalance(uniform_sphere) < 0.05

    def test_anisotropic_still_balanced(self, anisotropic):
        # L2-normalising a skewed distribution should not badly unbalance signs
        assert sign_imbalance(anisotropic) < 0.15

    def test_constant_vector_is_maximally_imbalanced(self):
        X = np.ones((100, 8), dtype=np.float32)
        assert sign_imbalance(X) == pytest.approx(0.5, abs=1e-6)


class TestMeanOffset:
    def test_uniform_sphere_near_zero(self, uniform_sphere):
        # E[x] = 0 for symmetric distribution; sample noise ~ 1/sqrt(N) ~ 0.022
        assert mean_offset(uniform_sphere) < 0.05

    def test_biased_mean_detected(self, biased_mean):
        assert mean_offset(biased_mean) > 0.05

    def test_zero_mean_exactly(self):
        # Antipodal pairs cancel exactly.
        v = np.eye(4, dtype=np.float32)
        X = np.vstack([v, -v])
        assert mean_offset(X) == pytest.approx(0.0, abs=1e-6)


class TestCovSpectralRatio:
    def test_uniform_sphere_near_one(self, uniform_sphere):
        # For N >> D the ratio converges to 1; allow generous tolerance.
        ratio = cov_spectral_ratio(uniform_sphere)
        assert ratio < 5.0, f"spectral ratio {ratio:.1f} too large for isotropic data"

    def test_anisotropic_has_large_ratio(self, anisotropic):
        ratio = cov_spectral_ratio(anisotropic)
        assert ratio > 5.0, f"spectral ratio {ratio:.1f} should be large for anisotropic data"

    def test_isotropic_synthetic_tight(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5000, 32)).astype(np.float32)
        X = _l2_normalize(X)
        assert cov_spectral_ratio(X) < 3.0


class TestEffectiveRankRatio:
    def test_uniform_sphere_near_one(self, uniform_sphere):
        ratio = effective_rank_ratio(uniform_sphere)
        assert ratio > 0.8, f"effective rank ratio {ratio:.3f} too low for isotropic data"

    def test_anisotropic_is_low(self, anisotropic):
        ratio = effective_rank_ratio(anisotropic)
        assert ratio < 0.5, f"effective rank ratio {ratio:.3f} should be low for anisotropic data"

    def test_rank_one_is_minimal(self):
        # All vectors are ±e_0 — rank-1 covariance, everything along one axis.
        v = np.zeros((1000, 32), dtype=np.float32)
        v[::2, 0] = 1.0
        v[1::2, 0] = -1.0
        assert effective_rank_ratio(v) < 0.1


# ---------------------------------------------------------------------------
# Integration test: plug in real embeddings
# ---------------------------------------------------------------------------
# Supply embeddings via --embeddings-npy on the command line or by overriding
# the `embeddings` fixture in conftest.py.
#
# Example:
#   pytest tests/test_embedding_isotropy.py::test_real_embeddings_are_isotropic \
#          --embeddings-npy /path/to/embeddings.npy -s
# ---------------------------------------------------------------------------

@pytest.fixture
def real_embeddings(request):
    path = request.config.getoption("--embeddings-npy", default=None)
    if path is None:
        pytest.skip("Pass --embeddings-npy /path/to/embeddings.npy to run this test.")
    X = np.load(path).astype(np.float32)
    assert X.ndim == 2, f"Expected 2-D array, got shape {X.shape}"
    return X


@pytest.mark.slow
def test_real_embeddings_are_isotropic(real_embeddings, capsys):
    X = real_embeddings
    N, D = X.shape
    report = isotropy_report(X)

    with capsys.disabled():
        print(f"\nIsotropy report  N={N}  D={D}")
        print(f"  sign_imbalance       {report['sign_imbalance']:.4f}   (< 0.05 = good)")
        print(f"  mean_offset          {report['mean_offset']:.4f}   (< 0.05 = good)")
        print(f"  cov_spectral_ratio   {report['cov_spectral_ratio']:.1f}     (< 5   = good)")
        print(f"  effective_rank_ratio {report['effective_rank_ratio']:.4f}   (> 0.8 = good)")

    # These thresholds are empirical; adjust based on your model and corpus size.
    assert report["sign_imbalance"] < 0.10, (
        f"sign_imbalance={report['sign_imbalance']:.4f}: "
        "some dimensions are heavily skewed — sign(x) bits will be uninformative"
    )
    assert report["mean_offset"] < 0.10, (
        f"mean_offset={report['mean_offset']:.4f}: "
        "embeddings have a large mean — subtract it before hashing"
    )
    assert report["cov_spectral_ratio"] < 20.0, (
        f"cov_spectral_ratio={report['cov_spectral_ratio']:.1f}: "
        "variance is concentrated in a few directions — sign(x) loses information"
    )
    assert report["effective_rank_ratio"] > 0.5, (
        f"effective_rank_ratio={report['effective_rank_ratio']:.4f}: "
        "effective dimensionality is less than half D — random W would help"
    )
