"""Projection matrices for simdq.

Spec section 6 ("Projection") allows two flavors:

  * identity: W = I_D  (no projection; quantize the encoder output as-is).
  * random_orthogonal: W = Q^T where Q comes from QR of an iid Gaussian
    matrix.  Rows of W are orthonormal, so applying W to a vector preserves
    inner products (when d == D) or projects onto a random d-dim subspace
    (when d == D/2).

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
