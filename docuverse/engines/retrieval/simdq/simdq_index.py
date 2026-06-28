"""SimdqIndex — build/save/load/search a simdq index.

On-disk layout (matches spec section 7):

    <index_path>/
    ├── meta.json
    ├── W.npy           # (d, D_orig) fp32 projection matrix
    ├── standardize.npz # optional: fitted (mu, inv_sigma, dead_mask) affine
    ├── codes.bin       # SoA (asymmetric) or AoS (hamming) packed codes
    ├── scales.bin      # (N,) fp16 per-vector scales (asymmetric only)
    └── floats.bin      # optional: (N, d) fp16, mmap'd at search

Supported encoder dims D ∈ {384, 768, 1024, 1536}; reduced dims d ∈ {D, D/2}.

Two family variants are supported:
  * "asymmetric": existing b-bit (b ∈ {1, 2, 4}) SoA codes with per-vector
    fp16 scales. Query stays float; scores are dot-product approximations.
  * "hamming": 1-bit AoS Hamming codes. No scales, no b parameter. Query
    is sign-quantized at search time; scores are negative Hamming distances.

Per-vector scale is applied **post-scan in Python** (multiply each top-K'
raw score by scales[idx] before sorting). For unit-norm corpora the
ranking is unchanged; for variable-norm corpora the codes-only mode is
approximate and users should pass K_prime > K to enable two-stage
rescore against floats.bin.
"""
from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:  # threadpoolctl is optional
    _threadpool_limits = None


@contextlib.contextmanager
def _single_threaded_blas():
    """Keep BLAS single-threaded for the tiny per-query projection GEMV.

    The projection ``W @ q`` is ~1 MFLOP, but at D=768 OpenBLAS parallelizes it
    and spawns its own thread pool. Under the engine's ``parallel_process`` (one
    query per thread) those pools oversubscribe the cores and dominate runtime —
    the root cause of the 768-dim "slowness" (see
    docs/simdq_768_investigation.md). A no-op if threadpoolctl is unavailable.
    """
    if _threadpool_limits is None:
        yield
    else:
        with _threadpool_limits(limits=1, user_api="blas"):
            yield

from docuverse.engines.retrieval.simdq import _simdq_native as _native
from docuverse.engines.retrieval.simdq import projection as _projection
from docuverse.engines.retrieval.simdq import quantization as _quant
from docuverse.engines.retrieval.simdq.projection import (
    SUPPORTED_D, SUPPORTED_d, validate_D_d,
)
from docuverse.engines.retrieval.simdq.standardize import Standardizer

FORMAT_VERSION = 1

# Per-bit weights for packing a 64-dim sign block into a uint64 word (bit b ->
# value 1<<b), used by the Hamming query quantizer.
_HAMMING_BIT_WEIGHTS = np.uint64(1) << np.arange(64, dtype=np.uint64)


@dataclass
class SimdqIndex:
    """In-memory + on-disk simdq index.

    family="asymmetric" stores per-bit-quantized codes + per-vector scales.
    family="hamming"    stores 1-bit AoS Hamming codes; no scales, no b.
    """
    family: str                                 # "asymmetric" | "hamming"
    codes: np.ndarray                           # uint8
    W: np.ndarray                               # fp32 (d, D)
    n_vectors: int
    D_orig: int                                 # encoder dim
    d: int                                      # reduced dim
    b: Optional[int] = None                     # 1/2/4 if asym; None if hamming
    scales: Optional[np.ndarray] = None         # fp16 (N,) if asym; None if hamming
    projection_name: str = "identity"
    projection_seed: int = 42
    has_floats: bool = False
    floats_mmap: Optional[np.ndarray] = None
    encoder_id: Optional[str] = None
    standardizer: Optional[Standardizer] = None    # affine pre-projection fix

    # ----- build -----

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        family: str = "asymmetric",
        b: Optional[int] = 2,
        d: Optional[int] = None,
        projection: str = "identity",
        projection_seed: int = 42,
        store_floats: bool = True,
        encoder_id: Optional[str] = None,
        standardize: bool = False,
        itq_iters: int = 50,
    ) -> "SimdqIndex":
        if vectors.ndim != 2:
            raise ValueError(f"simdq build: vectors must be 2-D; got {vectors.shape}")
        if vectors.shape[0] == 0:
            raise ValueError(
                f"simdq build: N must be >= 1; got an empty corpus with shape "
                f"{vectors.shape}"
            )
        D = int(vectors.shape[1])
        if D not in SUPPORTED_D:
            raise ValueError(f"simdq build: D must be one of {SUPPORTED_D}; got {D}")
        if d is None:
            d = D
        validate_D_d(D, d)

        if family not in ("asymmetric", "hamming"):
            raise ValueError(f"simdq build: family must be 'asymmetric' or 'hamming'; got {family!r}")
        if family == "asymmetric" and b not in (1, 2, 4):
            raise ValueError(f"simdq build: asymmetric requires b in (1, 2, 4); got {b}")
        if family == "hamming" and (d % 64) != 0:
            raise ValueError(f"simdq build: hamming requires d divisible by 64; got d={d}")

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        N = int(vectors.shape[0])

        # Optional affine standardization, fit on the corpus and applied (here
        # and at search time) *before* the linear projection W.
        standardizer = Standardizer.fit(vectors) if standardize else None
        if standardizer is not None:
            vectors = standardizer.apply(vectors)

        if projection == "identity" and d == D:
            W = _projection.identity(D)
        elif projection == "random_orthogonal":
            W = _projection.random_orthogonal(D, d, projection_seed)
        elif projection == "learned_orthogonal":
            # Data-aware ITQ rotation, fit on the (standardized) corpus.
            W = _projection.learned_orthogonal(
                vectors, d, seed=projection_seed, n_iters=itq_iters,
            )
        elif projection == "identity" and d != D:
            raise ValueError(
                "simdq build: projection='identity' requires d == D; "
                f"got D={D}, d={d}. Use projection='random_orthogonal' "
                "for d=D/2 reductions."
            )
        else:
            raise ValueError(
                "simdq build: projection must be 'identity', 'random_orthogonal', "
                f"or 'learned_orthogonal'; got {projection!r}"
            )
        Y = _projection.apply_projection(vectors, W)              # (N, d)

        if family == "asymmetric":
            scales_fp32 = _quant.fit_scales(Y)
            codes = _quant.pack(Y, scales_fp32, b=b)
            scales_fp16 = scales_fp32.astype(np.float16)
            stored_b = b
        else:
            codes = _quant.pack_hamming(Y)
            scales_fp16 = None
            stored_b = None

        floats_mmap = Y.astype(np.float16) if store_floats else None

        return cls(
            family=family,
            codes=codes, scales=scales_fp16, W=W,
            n_vectors=N, D_orig=D, d=d, b=stored_b,
            projection_name=projection, projection_seed=projection_seed,
            has_floats=store_floats, floats_mmap=floats_mmap,
            encoder_id=encoder_id, standardizer=standardizer,
        )

    # ----- save / load -----

    def save(self, path: os.PathLike) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        if tmp.exists():
            for child in tmp.iterdir(): child.unlink()
            tmp.rmdir()
        tmp.mkdir(parents=True)

        (tmp / "codes.bin").write_bytes(self.codes.tobytes())
        if self.scales is not None:
            (tmp / "scales.bin").write_bytes(self.scales.tobytes())
        np.save(tmp / "W.npy", self.W)
        if self.standardizer is not None:
            self.standardizer.save(tmp / "standardize.npz")
        if self.has_floats:
            assert self.floats_mmap is not None
            (tmp / "floats.bin").write_bytes(
                np.ascontiguousarray(self.floats_mmap).tobytes()
            )

        meta = {
            "format_version": FORMAT_VERSION,
            "n_vectors": int(self.n_vectors),
            "D_orig": int(self.D_orig),
            "d": int(self.d),
            "family": self.family,
            "b": (None if self.b is None else int(self.b)),
            "projection": self.projection_name,
            "projection_seed": int(self.projection_seed),
            "quantization": ("none" if self.family == "hamming" else "per_vector"),
            "code_layout": ("aos" if self.family == "hamming" else "soa"),
            "code_bytes_per_vector": int(self.codes.size // self.n_vectors)
                                     if self.family == "hamming"
                                     else int(self.codes.size // self.d),
            "has_floats": bool(self.has_floats),
            "float_dtype": "float16" if self.has_floats else None,
            "encoder_id": self.encoder_id,
            "standardize": self.standardizer is not None,
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))

        if path.exists():
            for child in path.iterdir(): child.unlink()
            path.rmdir()
        tmp.rename(path)

    @classmethod
    def load(cls, path: os.PathLike) -> "SimdqIndex":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        if meta["format_version"] != FORMAT_VERSION:
            raise ValueError(
                f"simdq load: format_version mismatch (file={meta['format_version']}, "
                f"code={FORMAT_VERSION}). Rebuild the index."
            )
        N = meta["n_vectors"]
        d = meta["d"]
        family = meta.get("family", "asymmetric")    # legacy: pre-Plan-3 indexes are asym
        b = meta.get("b")

        codes = np.fromfile(path / "codes.bin", dtype=np.uint8)
        scales = None
        if family == "asymmetric":
            scales = np.fromfile(path / "scales.bin", dtype=np.float16)
            if scales.shape[0] != N:
                raise ValueError(
                    f"simdq load: scales.bin has {scales.shape[0]} entries; "
                    f"meta says n_vectors={N}"
                )
        W = np.load(path / "W.npy")
        standardizer = None
        if meta.get("standardize"):
            standardizer = Standardizer.load(path / "standardize.npz")
        floats_mmap = None
        if meta.get("has_floats"):
            floats_mmap = np.memmap(
                path / "floats.bin", dtype=np.float16, mode="r",
                shape=(N, d),
            )
        return cls(
            family=family,
            codes=codes, scales=scales, W=W,
            n_vectors=N, D_orig=meta.get("D_orig", d), d=d, b=b,
            projection_name=meta["projection"],
            projection_seed=meta["projection_seed"],
            has_floats=bool(meta.get("has_floats", False)),
            floats_mmap=floats_mmap,
            encoder_id=meta.get("encoder_id"),
            standardizer=standardizer,
        )

    # ----- search -----

    def search(self, q: np.ndarray, K: int = 10, K_prime: Optional[int] = None,
               num_threads: int = 0):
        if K <= 0 or K > 256:
            raise ValueError(f"simdq search: K must be in [1, 256]; got {K}")
        if K_prime is None: K_prime = K
        if K_prime < K or K_prime > 256:
            raise ValueError(
                f"simdq search: need K <= K_prime <= 256; got K={K}, K'={K_prime}"
            )
        if q.dtype != np.float32:
            q = q.astype(np.float32, copy=False)
        if q.shape != (self.D_orig,):
            raise ValueError(
                f"simdq search: q must have shape ({self.D_orig},); got {q.shape}"
            )

        if self.standardizer is not None:
            q = self.standardizer.apply(q)
        q_proj = self._project_query(q)

        if self.family == "asymmetric":
            return self._search_asym(q_proj, K, K_prime, num_threads)
        return self._search_hamming(q_proj, K, K_prime, num_threads)

    def _project_query(self, q: np.ndarray) -> np.ndarray:
        """Apply the projection W to a query, cheaply.

        Identity projection (the default, and the common case) is a no-op, so we
        skip the (D, D) matmul entirely — both to save the work and, more
        importantly, to avoid OpenBLAS spawning a parallel thread pool for it
        (see _single_threaded_blas / docs/simdq_768_investigation.md). The real
        projection path keeps the matmul but pins BLAS to one thread.
        """
        if self.projection_name == "identity" and self.d == self.D_orig:
            return np.ascontiguousarray(q, dtype=np.float32)
        with _single_threaded_blas():
            q_proj = self.W @ q
        return np.ascontiguousarray(q_proj.astype(np.float32, copy=False))

    def _search_asym(self, q_proj, K, K_prime, num_threads):
        scan = {1: _native.scan_b1, 2: _native.scan_b2, 4: _native.scan_b4}[self.b]
        scores_buf, idx_buf = scan(self.codes, self.n_vectors, self.d,
                                   q_proj, K_prime, num_threads)
        raw_scores = np.frombuffer(scores_buf, dtype=np.float32).copy()
        idxs       = np.frombuffer(idx_buf, dtype=np.int64).copy()
        scales = self.scales[idxs].astype(np.float32)
        scaled = raw_scores * scales
        if self.has_floats and K_prime > K:
            cand_floats = np.array(self.floats_mmap[idxs], dtype=np.float32)
            rescore = cand_floats @ q_proj
            order = np.argsort(-rescore)[:K]
            return idxs[order], rescore[order].astype(np.float32)
        order = np.argsort(-scaled)[:K]
        return idxs[order], scaled[order]

    def _search_hamming(self, q_proj, K, K_prime, num_threads):
        # Quantize the projected query to 1-bit codes (same packing as pack_hamming):
        # bit b of word w is set iff q_proj[w*64 + b] >= 0. Vectorized — the old
        # per-bit Python loop was ~49× slower and GIL-bound under the threaded
        # search path (see docs/simdq_768_investigation.md §6).
        words = self.d // 64
        signs = (q_proj >= 0.0).reshape(words, 64).astype(np.uint64)
        q_bits = (signs * _HAMMING_BIT_WEIGHTS).sum(axis=1).astype(np.uint64)
        q_bytes = q_bits.tobytes()                                # 8*words bytes

        dist_buf, idx_buf = _native.scan_hamming(
            self.codes, self.n_vectors, self.d, q_bytes, K_prime, num_threads,
        )
        dists = np.frombuffer(dist_buf, dtype=np.int64).copy()
        idxs  = np.frombuffer(idx_buf,  dtype=np.int64).copy()

        if self.has_floats and K_prime > K:
            cand_floats = np.array(self.floats_mmap[idxs], dtype=np.float32)
            rescore = cand_floats @ q_proj
            order = np.argsort(-rescore)[:K]
            return idxs[order], rescore[order].astype(np.float32)
        # Codes-only Hamming: smaller distance = better. Convert to a
        # similarity-like score (negative distance) so callers always sort
        # descending.
        order = np.argsort(dists)[:K]
        return idxs[order], (-dists[order]).astype(np.float32)
