"""Affine pre-projection standardizer for simdq.

Real transformer embeddings are anisotropic and carry rogue / dead dimensions
(see ``docuverse.utils.isotropy``), which makes naked ``sign(x)`` a poor SimHash
code. The cheap fix is to **center and per-dimension standardize** the encoder
output before the linear projection ``W`` and bit-quantization:

    x' = (x - mu) / sigma           (dead dims, sigma ~ 0, are zeroed)

This is an *affine* map: the mean offset ``mu`` cannot be folded into the linear
``W`` (which only sees ``W @ x``), so the fitted ``(mu, sigma)`` are stored with
the index and applied identically at build time (to the corpus) and search time
(to the query). Skipping it on one side would break the dot-product geometry.

Standardization fixes per-bit sign balance and the mean offset; it does **not**
decorrelate dimensions, so a low ``effective_rank_ratio`` can persist and is the
signal that a random rotation ``W`` (``simdq_projection='random_orthogonal'``)
is still needed on top. The two are complementary, not alternatives.

This transform is opt-in (``simdq_standardize``) and gated by the recall
regression (T5) — whether it helps a given encoder is an empirical, per-model
tuning decision, not a default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from docuverse.utils.isotropy import DEAD_DIM_EPS, dead_dim_mask


@dataclass
class Standardizer:
    """Fitted per-dimension affine standardizer: ``x -> (x - mu) * inv_sigma``.

    ``inv_sigma`` is 0 on dead dims, so those map to exactly 0 (neutral in both
    the sign code and the dot-product rescore) rather than a saturated bit.
    """
    mu: np.ndarray          # (D,) fp32
    inv_sigma: np.ndarray   # (D,) fp32; 0 on dead dims
    dead_mask: np.ndarray   # (D,) bool

    @classmethod
    def fit(cls, vectors: np.ndarray, eps: float = DEAD_DIM_EPS) -> "Standardizer":
        if vectors.ndim != 2:
            raise ValueError(f"Standardizer.fit: expected 2-D, got {vectors.shape}")
        X = vectors.astype(np.float32, copy=False)
        # Compute moments in float64: the float32 mean/std of large-N columns
        # accumulates error proportional to magnitude (a constant column can
        # report std ~ 1e-5), which would both mis-fit sigma and hide dead dims.
        X64 = X.astype(np.float64)
        mu = X64.mean(axis=0)
        sigma = X64.std(axis=0)
        dead = dead_dim_mask(X, eps)
        # avoid 1/0 warnings on dead dims; their inv_sigma is forced to 0 anyway
        safe_sigma = np.where(dead, 1.0, sigma)
        inv_sigma = np.where(dead, 0.0, 1.0 / safe_sigma).astype(np.float32)
        return cls(
            mu=mu.astype(np.float32, copy=False),
            inv_sigma=inv_sigma,
            dead_mask=dead,
        )

    @property
    def n_dead(self) -> int:
        return int(self.dead_mask.sum())

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Standardize a (N, D) batch or a single (D,) vector. Returns fp32."""
        X = vectors.astype(np.float32, copy=False)
        out = (X - self.mu) * self.inv_sigma
        return np.ascontiguousarray(out, dtype=np.float32)

    # ----- persistence -----

    def save(self, path: Path) -> None:
        """Persist as a single .npz next to the index."""
        np.savez(
            path,
            mu=self.mu,
            inv_sigma=self.inv_sigma,
            dead_mask=self.dead_mask,
        )

    @classmethod
    def load(cls, path: Path) -> "Standardizer":
        z = np.load(path)
        return cls(
            mu=z["mu"].astype(np.float32, copy=False),
            inv_sigma=z["inv_sigma"].astype(np.float32, copy=False),
            dead_mask=z["dead_mask"].astype(bool, copy=False),
        )
