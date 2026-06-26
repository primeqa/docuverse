"""Isotropy checks for embedding distributions.

The metrics live in ``docuverse.utils.isotropy`` (imported here) so the same code
backs both these tests and the simdq encoder's tuning path. This file:

  * unit-tests the metrics against known synthetic distributions, and
  * (marked ``slow`` / ``diagnostic``) loads real embeddings via ``--embeddings-npy``
    and checks whether the encoder's *fitted* standardizing transform makes
    ``sign(x)`` a usable SimHash code.

Design note: the test measures, it does not invent a fix. The transform it gates
is the same ``Standardizer`` the simdq index fits and persists — see
``docuverse.engines.retrieval.simdq.standardize``. Anisotropy that standardization
cannot remove (a low ``effective_rank_ratio``) is *reported*, not gated, because
the remedy there is a random rotation ``W`` (``simdq_projection='random_orthogonal'``)
whose benefit the rotation-invariant spectral metrics cannot show directly.

Example:
  pytest tests/test_embedding_isotropy.py -m diagnostic \
         --embeddings-npy /path/to/embeddings.npy -s
"""
from __future__ import annotations

import numpy as np
import pytest

from docuverse.utils.isotropy import (
    cov_spectral_ratio,
    effective_rank_ratio,
    isotropy_report,
    l2_normalize,
    mean_offset,
    n_dead_dims,
    sign_imbalance,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_sphere():
    """N=2000, D=64 vectors sampled uniformly from S^{D-1}."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2000, 64)).astype(np.float32)
    return l2_normalize(X)


@pytest.fixture
def anisotropic():
    """N=2000, D=64 vectors with variance concentrated in first 4 dims."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2000, 64)).astype(np.float32)
    X[:, :4] *= 20.0          # inflate first 4 dimensions by 20x
    return l2_normalize(X)


@pytest.fixture
def biased_mean():
    """N=2000, D=64 vectors with a large mean offset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2000, 64)).astype(np.float32)
    X[:, 0] += 5.0            # strong bias in first dimension
    return l2_normalize(X)


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

    def test_constant_dim_is_reported_dead_not_skewed(self):
        # An all-constant matrix is degenerate, not "skewed": with drop_dead the
        # live-dim imbalance is 0 (no live dims) and the dims surface via
        # n_dead_dims; the raw max still reports the saturated 0.5.
        X = np.ones((100, 8), dtype=np.float32)
        assert sign_imbalance(X) == pytest.approx(0.0, abs=1e-6)
        assert sign_imbalance(X, drop_dead=False) == pytest.approx(0.5, abs=1e-6)
        assert n_dead_dims(X) == 8

    def test_single_dead_dim_does_not_saturate_metric(self):
        # One constant dim among many balanced ones must not pin the metric at 0.5
        # (the bug that hid the live-dim picture on real embeddings).
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4000, 64)).astype(np.float32)
        X[:, 7] = 3.0                      # one dead/constant dim
        assert n_dead_dims(X) == 1
        assert sign_imbalance(X) < 0.05    # live dims still balanced
        assert sign_imbalance(X, drop_dead=False) == pytest.approx(0.5, abs=1e-6)


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
        X = l2_normalize(X)
        assert cov_spectral_ratio(X) < 3.0

    def test_percentile_ratio_robust_to_near_zero_tail(self):
        # Real embeddings are near rank-deficient: a few trailing eigenvalues are
        # tiny but nonzero. A naive lambda_max/lambda_min explodes on that tail;
        # the inner-percentile ratio ignores it and stays bounded.
        rng = np.random.default_rng(1)
        X = rng.standard_normal((5000, 64)).astype(np.float32)
        X[:, 0] *= 1e-5                                   # one near-degenerate axis
        X = l2_normalize(X)
        s = np.linalg.svd(X - X.mean(0), full_matrices=False)[1]
        lam = s ** 2
        naive = lam.max() / (lam.min() + 1e-12)
        ratio = cov_spectral_ratio(X)
        assert naive > 1e6                               # the tail wrecks the naive ratio
        assert np.isfinite(ratio) and ratio < 10.0       # percentile ratio stays sane


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
# Integration tests: real embeddings + the encoder's fitted transform
# ---------------------------------------------------------------------------
# Supply embeddings via --embeddings-npy on the command line.
#
#   pytest tests/test_embedding_isotropy.py -m diagnostic \
#          --embeddings-npy /path/to/embeddings.npy -s
# ---------------------------------------------------------------------------

# What standardization is responsible for (the encoder-owned, cheap fix).
SIGN_IMBALANCE_GATE = 0.10
MEAN_OFFSET_GATE = 0.10
# Below this effective rank, standardization is not enough — recommend a random
# rotation W. Reported, not gated (rotation-invariant metrics can't show its gain).
EFFECTIVE_RANK_RECOMMEND_W = 0.80


def _fit_standardize(X: np.ndarray) -> np.ndarray:
    """Apply the same affine transform the simdq index fits at build time."""
    from docuverse.engines.retrieval.simdq.standardize import Standardizer
    return Standardizer.fit(X).apply(X)


def _print_report(title: str, report: dict[str, float], N: int, D: int) -> None:
    print(f"\n{title}  N={N}  D={D}")
    print(f"  sign_imbalance       {report['sign_imbalance']:.4f}   (< 0.10 = good)")
    print(f"  n_dead_dims          {int(report['n_dead_dims'])}")
    print(f"  mean_offset          {report['mean_offset']:.4f}   (< 0.10 = good)")
    print(f"  cov_spectral_ratio   {report['cov_spectral_ratio']:.2f}     (< 20  = good)")
    print(f"  effective_rank_ratio {report['effective_rank_ratio']:.4f}   (> 0.80 = good)")


@pytest.fixture
def real_embeddings(request):
    path = request.config.getoption("--embeddings-npy", default=None)
    if path is None:
        pytest.skip("Pass --embeddings-npy /path/to/embeddings.npy to run this test.")
    X = np.load(path).astype(np.float32)
    assert X.ndim == 2, f"Expected 2-D array, got shape {X.shape}"
    return X


@pytest.mark.slow
@pytest.mark.diagnostic
def test_real_embeddings_isotropy_report(real_embeddings, capsys):
    """Report-only: print raw vs. standardized metrics and a geometry hint.

    Never fails — a diagnostic to decide what preprocessing a given encoder needs
    before sign-hashing. Standardization fixes sign balance and centering; a low
    effective_rank_ratio that survives it means the dims stay correlated.

    The hint about a random rotation is a *geometry heuristic only*, not an NDCG
    result: effective_rank_ratio is rotation-invariant (it cannot measure whether
    W helps ranking), and the random_orthogonal recipes also reduce d=D/2, which
    trades quality for index size. Always confirm against the recall sweep
    (tests/test_simdq_recall.py) before choosing a projection.
    """
    X = real_embeddings
    N, D = X.shape
    raw = isotropy_report(X)
    std = isotropy_report(_fit_standardize(X))
    with capsys.disabled():
        _print_report("Isotropy report [raw]", raw, N, D)
        _print_report("Isotropy report [standardized]", std, N, D)
        if std["effective_rank_ratio"] >= EFFECTIVE_RANK_RECOMMEND_W:
            hint = "standardized embeddings look isotropic enough for sign(x)"
        else:
            hint = (
                f"dims stay correlated (effective rank "
                f"{std['effective_rank_ratio']:.2f} < {EFFECTIVE_RANK_RECOMMEND_W} "
                "after standardization). A random rotation MIGHT help sign(x) act "
                "as SimHash — but this is geometry, not NDCG: confirm with "
                "tests/test_simdq_recall.py (identity+standardize often wins)."
            )
        print(f"\n  geometry hint (not an NDCG verdict): {hint}")


@pytest.mark.slow
def test_fitted_transform_yields_balanced_bits(real_embeddings, capsys):
    """Strict gate on the encoder's fitted standardizer: after the transform the
    sign bits must be balanced and the embeddings centred.

    This is the property standardization *guarantees* and that directly makes
    sign(x) bits usable. Residual spectral anisotropy (needing W) is intentionally
    left to the diagnostic report, not gated here.
    """
    X = real_embeddings
    N, D = X.shape
    std = _fit_standardize(X)
    report = isotropy_report(std)

    with capsys.disabled():
        _print_report("Isotropy report [standardized]", report, N, D)

    assert report["sign_imbalance"] < SIGN_IMBALANCE_GATE, (
        f"sign_imbalance={report['sign_imbalance']:.4f} after standardization: "
        "some live dimension is still heavily sign-skewed — sign(x) bits lose info"
    )
    assert report["mean_offset"] < MEAN_OFFSET_GATE, (
        f"mean_offset={report['mean_offset']:.4f} after standardization: "
        "centering did not take — check the fitted transform"
    )


@pytest.mark.slow
@pytest.mark.diagnostic
def test_standardization_improves_isotropy(real_embeddings, capsys):
    """Quick check that the fitted transform recovers the cheap metrics.

    mean_offset must drop toward 0 (centering by construction) and sign_imbalance
    / cov_spectral_ratio must not worsen. effective_rank_ratio is reported but not
    gated: standardization rescales axes, it does not decorrelate them, so a low
    effective rank can persist and signals that a rotation W is still needed.
    """
    X = real_embeddings
    raw = isotropy_report(X)
    std = isotropy_report(_fit_standardize(X))

    with capsys.disabled():
        print("\nmetric                       raw   standardized")
        for k in raw:
            print(f"  {k:20s} {raw[k]:10.4f} {std[k]:12.4f}")

    assert std["mean_offset"] <= raw["mean_offset"] + 1e-6, (
        "centering should drive mean_offset toward 0"
    )
    assert std["sign_imbalance"] <= raw["sign_imbalance"] + 1e-6, (
        "standardization should not worsen sign balance"
    )
    assert std["cov_spectral_ratio"] <= raw["cov_spectral_ratio"] + 1e-6, (
        "equalising per-dim variance should not worsen the spectral ratio"
    )
