"""FaginIndex — per-dimension sorted lists + fp32 matrix for Fagin's
Threshold Algorithm (see docs/superpowers/specs/2026-07-06-fagin-threshold-
engine-design.md).

On-disk layout:

    <index_path>/
    ├── meta.json    # {format_version, n_vectors, dim, encoder_id}
    ├── Y.npy        # (N, D) fp32 row-major — random-access side
    ├── order.bin    # (D, N) int32 — per-dim argsort of Y[:, j], descending
    └── vals.bin     # (D, N) fp32  — Y[order[j], j] (sorted values)

All three arrays are mmap-loaded at search time. The native kernel
(_simdq_native.fagin_search) does the per-query TA scan.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from docuverse.engines.retrieval.simdq import _simdq_native as _native

FORMAT_VERSION = 1


@dataclass
class FaginIndex:
    Y: np.ndarray          # (N, D) fp32, C-contiguous
    order: np.ndarray      # (D, N) int32
    vals: np.ndarray       # (D, N) fp32
    n_vectors: int
    dim: int
    encoder_id: Optional[str] = None
    max_norm_sq: float = 1.0   # max squared row norm; ball radius^2 for the
                               # norm-aware (GTA/GTASD) halting bound

    # ----- build -----

    @classmethod
    def build(cls, vectors: np.ndarray,
              encoder_id: Optional[str] = None) -> "FaginIndex":
        if vectors.ndim != 2:
            raise ValueError(f"fagin build: vectors must be 2-D; got shape "
                             f"{vectors.shape}")
        if vectors.shape[0] == 0:
            raise ValueError("fagin build: N must be >= 1; got an empty corpus")
        Y = np.ascontiguousarray(vectors, dtype=np.float32)
        N, D = Y.shape
        # Row indices sorting each column descending; stable for ties.
        perm = np.argsort(-Y, axis=0, kind="stable")               # (N, D)
        vals = np.take_along_axis(Y, perm, axis=0)                 # (N, D)
        order = np.ascontiguousarray(perm.T, dtype=np.int32)       # (D, N)
        vals = np.ascontiguousarray(vals.T, dtype=np.float32)      # (D, N)
        # Max squared row norm: the ball radius^2 used by the norm-aware
        # (GTA/GTASD) halting bound. Exact for any corpus (== 1 for unit-norm).
        max_norm_sq = float(np.einsum("ij,ij->i", Y, Y).max()) if N else 1.0
        return cls(Y=Y, order=order, vals=vals, n_vectors=int(N), dim=int(D),
                   encoder_id=encoder_id, max_norm_sq=max_norm_sq)

    # ----- persist -----

    def save(self, path: os.PathLike) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        with open(tmp / "meta.json", "w") as f:
            json.dump({"format_version": FORMAT_VERSION,
                       "n_vectors": self.n_vectors,
                       "dim": self.dim,
                       "encoder_id": self.encoder_id,
                       "max_norm_sq": self.max_norm_sq}, f)
        np.save(tmp / "Y.npy", self.Y)
        np.ascontiguousarray(self.order).tofile(tmp / "order.bin")
        np.ascontiguousarray(self.vals).tofile(tmp / "vals.bin")
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: os.PathLike) -> "FaginIndex":
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"fagin load: unsupported format_version "
                             f"{meta.get('format_version')}")
        N, D = int(meta["n_vectors"]), int(meta["dim"])
        Y = np.load(path / "Y.npy", mmap_mode="r")
        order = np.memmap(path / "order.bin", dtype=np.int32, mode="r",
                          shape=(D, N))
        vals = np.memmap(path / "vals.bin", dtype=np.float32, mode="r",
                         shape=(D, N))
        return cls(Y=Y, order=order, vals=vals, n_vectors=N, dim=D,
                   encoder_id=meta.get("encoder_id"),
                   max_norm_sq=float(meta.get("max_norm_sq", 1.0)))

    # ----- search -----

    SCHEDULES = {"lockstep": 0, "steepest": 1,
                 "lockstep_norm": 2, "steepest_norm": 3}

    def search(self, q: np.ndarray, K: int, batch: int = 64,
               epsilon: float = 0.0, max_depth: int = 0,
               num_threads: int = 0,
               schedule: str = "lockstep") -> Tuple[np.ndarray, np.ndarray, dict]:
        """Run TA for one query. Returns (idxs int64[K], scores fp32[K],
        stats dict); unfilled slots (K > N) have idx -1.

        schedule: "lockstep" (round-robin sorted access, weight-blind) or
        "steepest" (advance the dim with the largest marginal threshold drop;
        exact at epsilon=0, fewer random accesses at epsilon>0). The
        "lockstep_norm"/"steepest_norm" variants (printed as GTA/GTASD) use the
        norm-aware water-filling halting bound: because the corpus vectors are
        L2-normalized, the classic box threshold sum_j q_j t_j overestimates
        badly, and the norm-constrained bound (also respecting ||x||_2 = 1) is
        provably tighter while staying exact at epsilon=0 — fewer sorted and
        random accesses. See research/2026-07-13-fagin-norm-aware-threshold."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"fagin search: q must have shape ({self.dim},); "
                             f"got {q.shape}")
        if schedule not in self.SCHEDULES:
            raise ValueError(f"fagin search: unknown schedule {schedule!r}; "
                             f"expected one of {sorted(self.SCHEDULES)}")
        scores_b, idx_b, stats = _native.fagin_search(
            self.Y, self.order, self.vals, q,
            int(K), int(batch), float(epsilon), int(max_depth),
            int(num_threads), self.SCHEDULES[schedule], float(self.max_norm_sq))
        scores = np.frombuffer(scores_b, dtype=np.float32)
        idxs = np.frombuffer(idx_b, dtype=np.int64)
        return idxs, scores, stats
