"""SimdqIndex — build/save/load/search a simdq index at D=768.

On-disk layout (matches spec section 7):

    <index_path>/
    ├── meta.json
    ├── W.npy           # (d, D) fp32 projection matrix
    ├── codes.bin       # SoA-packed codes, length = D * ceil(N / (8/b))
    ├── scales.bin      # (N,) fp16 per-vector scales (omitted if quantization='global')
    └── floats.bin      # optional: (N, d) fp16, mmap'd at search

Per-vector scale is applied **post-scan in Python** (multiply each top-K'
raw score by scales[idx] before sorting). For unit-norm corpora the
ranking is unchanged; for variable-norm corpora the codes-only mode is
approximate and users should pass K_prime > K to enable two-stage
rescore against floats.bin.

D is fixed at 768 in Plan 2. Plan 3 widens the kernel template + this
class to the rest of {384, 1024, 1536}.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from docuverse.engines.retrieval.simdq import _simdq_native as _native
from docuverse.engines.retrieval.simdq import projection as _projection
from docuverse.engines.retrieval.simdq import quantization as _quant

D_FIXED = _native.D
FORMAT_VERSION = 1


def _row_bytes(N: int, b: int) -> int:
    if b == 1: return (N + 7) // 8
    if b == 2: return (N + 3) // 4
    if b == 4: return (N + 1) // 2
    raise ValueError(f"bad b={b}")


@dataclass
class SimdqIndex:
    """In-memory + on-disk simdq index.

    Build via SimdqIndex.build(...); save / load via the class methods.
    Direct construction is supported but the .build entry point is the
    documented one.
    """
    codes: np.ndarray            # uint8, length D * row_bytes(N, b)
    scales: np.ndarray           # fp16 (N,)
    W: np.ndarray                # fp32 (d, D)  -- d == D_FIXED in Plan 2
    n_vectors: int
    b: int
    d: int
    projection_name: str
    projection_seed: int
    has_floats: bool
    floats_mmap: Optional[np.ndarray] = None   # fp16 (N, d), mmap'd if has_floats
    encoder_id: Optional[str] = None

    # ----- build -----

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        b: int = 2,
        d: Optional[int] = None,
        projection: str = "identity",
        projection_seed: int = 42,
        store_floats: bool = True,
        encoder_id: Optional[str] = None,
    ) -> "SimdqIndex":
        if vectors.ndim != 2 or vectors.shape[1] != D_FIXED:
            raise ValueError(
                f"simdq build: vectors must have shape (N, {D_FIXED}); "
                f"got {vectors.shape}"
            )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        N = vectors.shape[0]
        if d is None:
            d = D_FIXED
        if d != D_FIXED:
            # Plan 2 supports d == D only (kernel template is fixed at 768).
            # The projection step still works for d = D/2, but the kernel
            # cannot scan it; raise instead of silently producing a malformed
            # index.
            raise ValueError(
                f"simdq build: Plan 2 supports d == {D_FIXED} only; "
                f"got d={d}. d=D/2 will land with Plan 3."
            )
        if projection == "identity":
            W = _projection.identity(D_FIXED)
        elif projection == "random_orthogonal":
            W = _projection.random_orthogonal(D_FIXED, d, projection_seed)
        else:
            raise ValueError(
                f"simdq build: projection must be 'identity' or 'random_orthogonal'; "
                f"got {projection!r}"
            )

        Y = _projection.apply_projection(vectors, W)              # (N, d)
        scales_fp32 = _quant.fit_scales(Y)
        codes = _quant.pack(Y, scales_fp32, b=b)
        scales_fp16 = scales_fp32.astype(np.float16)
        floats_mmap = Y.astype(np.float16) if store_floats else None

        return cls(
            codes=codes, scales=scales_fp16, W=W,
            n_vectors=N, b=b, d=d,
            projection_name=projection, projection_seed=projection_seed,
            has_floats=store_floats, floats_mmap=floats_mmap,
            encoder_id=encoder_id,
        )

    # ----- save / load -----

    def save(self, path: os.PathLike) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        if tmp.exists():
            for child in tmp.iterdir():
                child.unlink()
            tmp.rmdir()
        tmp.mkdir(parents=True)

        # codes / scales
        (tmp / "codes.bin").write_bytes(self.codes.tobytes())
        (tmp / "scales.bin").write_bytes(self.scales.tobytes())
        np.save(tmp / "W.npy", self.W)
        if self.has_floats:
            assert self.floats_mmap is not None
            (tmp / "floats.bin").write_bytes(
                np.ascontiguousarray(self.floats_mmap).tobytes()
            )

        meta = {
            "format_version": FORMAT_VERSION,
            "n_vectors": int(self.n_vectors),
            "D_orig": D_FIXED,
            "d": int(self.d),
            "b": int(self.b),
            "projection": self.projection_name,
            "projection_seed": int(self.projection_seed),
            "quantization": "per_vector",
            "code_layout": "soa",
            "code_bytes_per_vector": int(self.codes.size // self.d),
            "has_floats": bool(self.has_floats),
            "float_dtype": "float16" if self.has_floats else None,
            "encoder_id": self.encoder_id,
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))

        # atomic-ish rename
        if path.exists():
            for child in path.iterdir():
                child.unlink()
            path.rmdir()
        tmp.rename(path)

    @classmethod
    def load(cls, path: os.PathLike) -> "SimdqIndex":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        if meta["format_version"] != FORMAT_VERSION:
            raise ValueError(
                f"simdq load: format_version mismatch — file has "
                f"{meta['format_version']}, this code understands "
                f"{FORMAT_VERSION}. Rebuild the index."
            )
        N = meta["n_vectors"]
        b = meta["b"]
        d = meta["d"]
        codes = np.fromfile(path / "codes.bin", dtype=np.uint8)
        scales = np.fromfile(path / "scales.bin", dtype=np.float16)
        if scales.shape[0] != N:
            raise ValueError(
                f"simdq load: scales.bin has {scales.shape[0]} entries; "
                f"meta says n_vectors={N}"
            )
        W = np.load(path / "W.npy")
        floats_mmap = None
        if meta.get("has_floats"):
            floats_mmap = np.memmap(
                path / "floats.bin", dtype=np.float16, mode="r",
                shape=(N, d),
            )
        return cls(
            codes=codes, scales=scales, W=W,
            n_vectors=N, b=b, d=d,
            projection_name=meta["projection"],
            projection_seed=meta["projection_seed"],
            has_floats=bool(meta.get("has_floats", False)),
            floats_mmap=floats_mmap,
            encoder_id=meta.get("encoder_id"),
        )

    # ----- search -----

    def search(
        self,
        q: np.ndarray,
        K: int = 10,
        K_prime: Optional[int] = None,
        num_threads: int = 0,
    ):
        """Top-K search.

        Args:
            q: (D,) fp32 query (pre-projection).
            K: top-K to return.
            K_prime: number of candidates the kernel returns; if > K and
                has_floats is True, an fp16 rescore stage selects the
                final top-K. If None, K_prime = K (codes-only mode).
            num_threads: 0 -> OpenMP default.

        Returns:
            (indices, scores): both shape (K,). indices int64, scores
            fp32, descending order.
        """
        if K <= 0 or K > 256:
            raise ValueError(f"simdq search: K must be in [1, 256]; got {K}")
        if K_prime is None:
            K_prime = K
        if K_prime < K:
            raise ValueError(
                f"simdq search: K_prime ({K_prime}) must be >= K ({K})"
            )
        if K_prime > 256:
            raise ValueError(
                f"simdq search: K_prime must be <= 256; got {K_prime}"
            )

        if q.dtype != np.float32:
            q = q.astype(np.float32, copy=False)
        if q.shape != (D_FIXED,):
            raise ValueError(
                f"simdq search: q must have shape ({D_FIXED},); got {q.shape}"
            )

        # project the query (asymmetric: query stays float)
        q_proj = (self.W @ q).astype(np.float32, copy=False)
        q_proj = np.ascontiguousarray(q_proj)

        # call the kernel for K' candidates
        if self.b == 1:
            scan = _native.scan_b1
        elif self.b == 2:
            scan = _native.scan_b2
        elif self.b == 4:
            scan = _native.scan_b4
        else:
            raise ValueError(f"simdq search: bad b={self.b}")
        scores_buf, idx_buf = scan(self.codes, self.n_vectors, q_proj, K_prime, num_threads)
        raw_scores = np.frombuffer(scores_buf, dtype=np.float32).copy()
        idxs = np.frombuffer(idx_buf, dtype=np.int64).copy()

        # apply per-vector scale (fp16 -> fp32 for the multiply)
        scales = self.scales[idxs].astype(np.float32)
        scaled_scores = raw_scores * scales

        if self.has_floats and K_prime > K:
            # fp16 rescore stage: full fp32 dot product against the projected
            # fp16 vectors for the K' candidates.
            assert self.floats_mmap is not None
            cand_floats = np.array(self.floats_mmap[idxs], dtype=np.float32)
            rescore = cand_floats @ q_proj                            # (K',)
            order = np.argsort(-rescore)[:K]
            return idxs[order], rescore[order].astype(np.float32)

        # codes-only or no floats: re-sort by scaled score, return top K
        order = np.argsort(-scaled_scores)[:K]
        return idxs[order], scaled_scores[order]
