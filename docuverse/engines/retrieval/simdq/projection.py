"""Projection matrices for simdq.

Spec section 6 ("Projection") allows three flavors:

  * identity: W = I_D  (no projection; quantize the encoder output as-is).
  * random_orthogonal: W = Q^T where Q comes from QR of an iid Gaussian
    matrix.  Rows of W are orthonormal, so applying W to a vector preserves
    inner products (when d == D) or projects onto a random d-dim subspace
    (when d == D/2).
  * learned_orthogonal: an orthonormal-row W fit on the corpus by ITQ
    (Gong & Lazebnik 2011) to minimise binary quantization error — a data-aware
    drop-in for random_orthogonal.  See :func:`learned_orthogonal`.  Best paired
    with the affine standardizer (standardize=True).

The "halve dims, double bits" comparison in the spec uses random_orthogonal
with d = D/2; the recipe sweep in Plan 3 will exercise both.

Supported encoder dims D are {384, 768, 1024, 1536}; reduced dims d must be
either D or D/2.  The family axis (asymmetric vs hamming) is handled in
simdq_index.py, not here.
"""
from __future__ import annotations

import numpy as np

SUPPORTED_D = (384, 768, 1024, 1536)
SUPPORTED_d = (192, 384, 512, 768, 1024, 1536)


def validate_D_d(D: int, d: int) -> None:
    if D not in SUPPORTED_D:
        raise ValueError(
            f"simdq: D must be one of {SUPPORTED_D}; got {D}"
        )
    if d not in SUPPORTED_d:
        raise ValueError(
            f"simdq: d must be one of {SUPPORTED_d}; got {d}"
        )
    if d not in (D, D // 2):
        raise ValueError(
            f"simdq: for D={D}, d must be {D} or {D // 2}; got {d}"
        )


def identity(D: int) -> np.ndarray:
    """Return I_D as a (D, D) float32 matrix."""
    return np.eye(D, dtype=np.float32)


def random_orthogonal(D: int, d: int, seed: int) -> np.ndarray:
    """Return W in R^{d x D} with orthonormal rows.

    Build via QR of a (D, d) standard-normal matrix; take the first d columns
    of Q (always orthonormal, even when d < D), transpose.

    Raises:
        ValueError: if d not in {D, D/2}.  v1 supports those two cases only.
    """
    if d not in (D, D // 2):
        raise ValueError(
            f"random_orthogonal: d must be one of {{D, D//2}} = "
            f"{{{D}, {D // 2}}}, got d={d}"
        )
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(size=(D, d)).astype(np.float32)
    Q, _ = np.linalg.qr(G)              # Q in R^{D x d}, columns orthonormal
    return Q.T.astype(np.float32, copy=False)


def learned_orthogonal(
    X: np.ndarray,
    d: int,
    seed: int = 42,
    n_iters: int = 50,
    max_fit_samples: int = 50000,
) -> np.ndarray:
    """Learn a (d, D) orthonormal-row projection via ITQ (Gong & Lazebnik 2011).

    ITQ minimizes the binary quantization error ``||sign(Z R) - Z R||^2`` over an
    orthogonal rotation ``R``, after a PCA reduction to ``d`` dims. It directly
    targets the information that ``sign(x)`` throws away, so it tends to beat a
    random rotation as a SimHash basis. Returns ``W = (P R)^T`` where ``P`` are
    the top-``d`` PCA axes and ``R`` is the learned rotation; ``W`` has
    orthonormal rows (``W Wᵀ = I_d``), so the float rescore tier stays faithful.

    ``X`` is assumed (approximately) zero-mean — pair with ``standardize=True``,
    which both centres and equalises per-dim variance. For ``d == D`` this is a
    full rotation; for ``d == D/2`` it also reduces dimension. The fit uses up to
    ``max_fit_samples`` rows (sampled) to bound build time; the resulting ``W``
    is applied to the whole corpus.

    Raises:
        ValueError: if ``d`` is not in ``{D, D/2}`` or fewer than ``d`` rows are
            available to fit.
    """
    if X.ndim != 2:
        raise ValueError(f"learned_orthogonal: X must be 2-D; got {X.shape}")
    X = X.astype(np.float32, copy=False)
    N, D = X.shape
    if d not in (D, D // 2):
        raise ValueError(
            f"learned_orthogonal: d must be one of {{D, D//2}} = "
            f"{{{D}, {D // 2}}}, got d={d}"
        )
    rng = np.random.default_rng(seed)

    Xfit = X
    if N > max_fit_samples:
        sel = rng.choice(N, size=max_fit_samples, replace=False)
        Xfit = X[sel]
    if Xfit.shape[0] < d:
        raise ValueError(
            f"learned_orthogonal: need at least d={d} samples to fit a "
            f"{d}-dim rotation; got {Xfit.shape[0]}"
        )

    # PCA to d via SVD of the (assumed zero-mean) data: top-d right singular
    # vectors are the principal axes. Orthonormal columns -> W keeps orthonormal
    # rows and the map stays purely linear (consistent corpus/query).
    _, _, Vt = np.linalg.svd(Xfit, full_matrices=False)
    P = np.ascontiguousarray(Vt[:d].T, dtype=np.float32)        # (D, d)
    Z = Xfit @ P                                                # (M, d)

    # ITQ: alternate sign-binarization and the orthogonal Procrustes update
    #   min_R ||B - Z R||^2  ->  R = V Uᵀ where U Σ Vᵀ = SVD(Bᵀ Z).
    R = np.linalg.qr(rng.standard_normal((d, d)))[0].astype(np.float32)
    for _ in range(n_iters):
        B = np.where(Z @ R >= 0.0, np.float32(1.0), np.float32(-1.0))
        U, _, Vt_r = np.linalg.svd(B.T @ Z)
        R = np.ascontiguousarray(Vt_r.T @ U.T, dtype=np.float32)

    return np.ascontiguousarray((P @ R).T, dtype=np.float32)    # (d, D)


def apply_projection(Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute Y @ W.T, returning a (N, d) float32 array.

    Y is (N, D) float32, W is (d, D) float32. Returns a contiguous fp32
    matrix — required by the C extension's compute_scales / pack_b{1,2,4}
    entry points.
    """
    if Y.dtype != np.float32:
        Y = Y.astype(np.float32, copy=False)
    if W.dtype != np.float32:
        W = W.astype(np.float32, copy=False)
    Z = Y @ W.T
    return np.ascontiguousarray(Z, dtype=np.float32)
