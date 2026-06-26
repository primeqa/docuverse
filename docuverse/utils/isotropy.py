"""Isotropy metrics for embedding distributions.

These quantify how close a set of embedding vectors is to being uniformly
distributed on the unit sphere — the empirical condition that makes ``sign(x)``
a useful SimHash code (e.g. for the simdq bit-hashing engine) without an
explicit random-projection matrix ``W``.

The metrics are pure measurement: they neither modify the embeddings nor decide
what to do about anisotropy. The *fix* (centering / standardization / a random
rotation) is a learned transform that belongs in the encoder/index build path —
see ``docuverse.engines.retrieval.simdq.standardize.Standardizer``.

Metrics
-------
sign_imbalance       max_d |P(x_d > 0) - 0.5| over *live* dims (0 = balanced).
                     Dead (zero-variance) dims are excluded — a constant dim is a
                     degenerate bit, reported separately via ``n_dead_dims``, not
                     a "skewed" one. Pass ``drop_dead=False`` for the raw max.
n_dead_dims          number of dimensions with ~zero variance (dead/constant).
mean_offset          ||E[x]||_2 / sqrt(D)        (0 = centred at origin).
cov_spectral_ratio   robust condition number of the covariance: a percentile
                     ratio (default 95th/5th eigenvalue). 1 = isotropic; larger =
                     anisotropic. A naive lambda_max/lambda_min is numerically
                     useless on real embeddings (near rank-deficient -> the ratio
                     explodes to 1e6-1e15 with no stable meaning).
effective_rank_ratio exp(H(p)) / D, p = s^2 / sum(s^2) (Roy & Vetterli 2007).
                     1 = all singular values equal; < 1 = concentrated.
"""
from __future__ import annotations

import numpy as np

DEAD_DIM_EPS = 1e-6     # relative to the most-varying dim; see dead_dim_mask


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def dead_dim_mask(embeddings: np.ndarray, eps: float = DEAD_DIM_EPS) -> np.ndarray:
    """Boolean mask (D,) marking near-zero-variance (dead/constant) dims.

    The threshold is *relative* to the most-varying dimension: a dim is dead when
    ``std < eps * std.max()`` (plus a tiny absolute floor for the all-dead case).
    The std is computed in float64 — a float32 ``np.std`` of a constant column
    reports spurious accumulation noise (~ value * N * 2^-23) that grows with N
    and magnitude, which a relative cutoff would otherwise miss.
    """
    std = embeddings.astype(np.float64, copy=False).std(axis=0)
    threshold = eps * float(std.max()) + 1e-12
    return std < threshold


def n_dead_dims(embeddings: np.ndarray, eps: float = DEAD_DIM_EPS) -> int:
    """Count of dimensions that are effectively constant across the corpus.

    A dead dim produces the same sign bit for every vector — a wasted bit that
    also saturates a naive ``max``-based ``sign_imbalance``.
    """
    return int(dead_dim_mask(embeddings, eps).sum())


def sign_imbalance(
    embeddings: np.ndarray, drop_dead: bool = True, eps: float = DEAD_DIM_EPS
) -> float:
    """Max |P(x_d > 0) - 0.5| over dimensions d.

    With ``drop_dead=True`` (default) zero-variance dims are excluded so a single
    dead dimension cannot pin the metric at 0.5 and hide the live-dim picture;
    those dims are surfaced by :func:`n_dead_dims` instead. If *every* dim is
    dead the result is 0.0 (there is no informative dim to be imbalanced).
    """
    pos_frac = (embeddings > 0).mean(axis=0)
    imb = np.abs(pos_frac - 0.5)
    if drop_dead:
        live = ~dead_dim_mask(embeddings, eps)
        if live.any():
            return float(imb[live].max())
        return 0.0
    return float(imb.max())


def mean_offset(embeddings: np.ndarray) -> float:
    """||mean||_2 / sqrt(D): 0 for a zero-mean distribution."""
    D = embeddings.shape[1]
    return float(np.linalg.norm(embeddings.mean(axis=0)) / np.sqrt(D))


def _centered_singular_values(embeddings: np.ndarray) -> np.ndarray:
    """Singular values of the L2-normalised, mean-centred matrix.

    L2-normalising first so scale differences don't mask directional anisotropy.
    """
    X = l2_normalize(embeddings)
    Xc = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    return s


def cov_spectral_ratio(
    embeddings: np.ndarray, lo: float = 5.0, hi: float = 95.0
) -> float:
    """Robust condition number of the covariance: percentile_hi / percentile_lo
    of its eigenvalues (default 95th / 5th).

    1.0 = isotropic; larger = anisotropic. Using inner percentiles ignores the
    numerically unstable smallest-eigenvalue tail, so the number is comparable
    across models (unlike a raw lambda_max/lambda_min, which on real embeddings
    just measures noise in the floating-point floor).
    """
    lam = _centered_singular_values(embeddings) ** 2
    return float(np.percentile(lam, hi) / (np.percentile(lam, lo) + 1e-12))


def effective_rank_ratio(embeddings: np.ndarray) -> float:
    """exp(H(p)) / D where p_i = s_i^2 / sum_j s_j^2 (Roy & Vetterli 2007).

    1.0 = all singular values equal (perfectly spread); < 1 = concentrated.
    """
    s = _centered_singular_values(embeddings)
    D = len(s)
    p = s ** 2 / (s ** 2).sum()
    entropy = -float(np.sum(p * np.log(p + 1e-30)))
    return float(np.exp(entropy) / D)


def isotropy_report(embeddings: np.ndarray) -> dict[str, float]:
    """All metrics as a dict, suitable for printing or asserting against."""
    return {
        "sign_imbalance":       sign_imbalance(embeddings),
        "n_dead_dims":          float(n_dead_dims(embeddings)),
        "mean_offset":          mean_offset(embeddings),
        "cov_spectral_ratio":   cov_spectral_ratio(embeddings),
        "effective_rank_ratio": effective_rank_ratio(embeddings),
    }
