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


def _fit_kmeans(Y: np.ndarray, nlist: int, seed: int):
    """k-means on Y -> (centroids (nlist, d) fp32, assignment (N,) int64).

    Uses faiss if available (fast), else sklearn. Only needed at build time.
    """
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    N, d = Y.shape
    nlist = int(min(nlist, N))
    try:
        import faiss
        km = faiss.Kmeans(d, nlist, niter=20, seed=int(seed), verbose=False)
        km.train(Y)
        centroids = np.ascontiguousarray(km.centroids.reshape(nlist, d), dtype=np.float32)
        assign = km.index.search(Y, 1)[1].ravel().astype(np.int64)
    except Exception:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=nlist, random_state=int(seed), n_init=3)
        assign = km.fit_predict(Y).astype(np.int64)
        centroids = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
    return centroids, assign


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
    # Optional IVF outer index (hamming only): cluster the corpus, store codes
    # reordered so each cluster is contiguous, and at search scan only the
    # nprobe nearest clusters instead of all N codes (see _search_hamming_ivf).
    ivf_centroids: Optional[np.ndarray] = None      # (nlist, d) fp32
    ivf_perm: Optional[np.ndarray] = None           # (N,) int64: reordered pos -> original id
    ivf_offsets: Optional[np.ndarray] = None        # (nlist+1,) int64 cluster boundaries
    ivf_nprobe: int = 1                             # default clusters to probe at search

    @property
    def has_ivf(self) -> bool:
        return self.ivf_centroids is not None

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
        ivf_nlist: Optional[int] = None,
        ivf_nprobe: int = 1,
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

        # Optional IVF outer index (hamming only). Cluster the corpus, then
        # reorder the AoS codes so each cluster occupies a contiguous range;
        # search scans only the nprobe nearest clusters. floats stay in original
        # order — ivf_perm maps a reordered position back to the original id.
        ivf_centroids = ivf_perm = ivf_offsets = None
        if ivf_nlist:
            if family != "hamming":
                raise ValueError("simdq build: ivf_nlist is only supported for family='hamming'")
            ivf_centroids, assign = _fit_kmeans(Y, ivf_nlist, projection_seed)
            nlist = ivf_centroids.shape[0]
            ivf_perm = np.argsort(assign, kind="stable").astype(np.int64)
            counts = np.bincount(assign, minlength=nlist)
            ivf_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            bpv = (d // 64) * 8                          # bytes per vector (AoS)
            codes = np.ascontiguousarray(
                codes.reshape(N, bpv)[ivf_perm].reshape(-1))

        floats_mmap = Y.astype(np.float16) if store_floats else None

        return cls(
            family=family,
            codes=codes, scales=scales_fp16, W=W,
            n_vectors=N, D_orig=D, d=d, b=stored_b,
            projection_name=projection, projection_seed=projection_seed,
            has_floats=store_floats, floats_mmap=floats_mmap,
            encoder_id=encoder_id, standardizer=standardizer,
            ivf_centroids=ivf_centroids, ivf_perm=ivf_perm,
            ivf_offsets=ivf_offsets, ivf_nprobe=int(ivf_nprobe),
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
        if self.has_ivf:
            np.save(tmp / "ivf_centroids.npy", self.ivf_centroids)
            np.save(tmp / "ivf_perm.npy", self.ivf_perm)
            np.save(tmp / "ivf_offsets.npy", self.ivf_offsets)
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
            "ivf": self.has_ivf,
            "ivf_nlist": int(self.ivf_centroids.shape[0]) if self.has_ivf else 0,
            "ivf_nprobe": int(self.ivf_nprobe),
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
        ivf_centroids = ivf_perm = ivf_offsets = None
        if meta.get("ivf"):
            ivf_centroids = np.load(path / "ivf_centroids.npy")
            ivf_perm = np.load(path / "ivf_perm.npy")
            ivf_offsets = np.load(path / "ivf_offsets.npy")
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
            ivf_centroids=ivf_centroids, ivf_perm=ivf_perm,
            ivf_offsets=ivf_offsets, ivf_nprobe=int(meta.get("ivf_nprobe", 1)),
        )

    # ----- search -----

    def search(self, q: np.ndarray, K: int = 10, K_prime: Optional[int] = None,
               num_threads: int = 0, nprobe: Optional[int] = None):
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
        if self.has_ivf:
            return self._search_hamming_ivf(
                q_proj, K, K_prime, num_threads,
                nprobe if nprobe is not None else self.ivf_nprobe)
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

    def _hamming_query_bytes(self, q_proj) -> bytes:
        # Quantize the projected query to 1-bit codes (same packing as pack_hamming):
        # bit b of word w is set iff q_proj[w*64 + b] >= 0. Vectorized — the old
        # per-bit Python loop was ~49× slower and GIL-bound under the threaded
        # search path (see docs/simdq_768_investigation.md §6).
        words = self.d // 64
        signs = (q_proj >= 0.0).reshape(words, 64).astype(np.uint64)
        q_bits = (signs * _HAMMING_BIT_WEIGHTS).sum(axis=1).astype(np.uint64)
        return q_bits.tobytes()                                   # 8*words bytes

    def _search_hamming(self, q_proj, K, K_prime, num_threads):
        q_bytes = self._hamming_query_bytes(q_proj)
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

    def _search_hamming_ivf(self, q_proj, K, K_prime, num_threads, nprobe):
        """IVF Hamming search: scan only the nprobe nearest clusters.

        Route the query to clusters by L2 to the centroids (argmax of
        q·c - ½‖c‖²), **gather** the selected clusters' contiguous code ranges
        into one buffer, run a single native popcount scan over it, then
        fp32-rescore the candidates. The single gathered scan is the key to the
        speedup: scanning per-cluster instead pays the kernel's per-call
        AoS→SoA transpose + alloc nprobe times and ends up slower than flat.
        Cutting the scanned set to ~nprobe/nlist of the corpus shrinks both the
        transpose and the popcount, relieving the memory-bandwidth wall.
        """
        nlist = self.ivf_centroids.shape[0]
        nprobe = max(1, min(int(nprobe), nlist))
        chalf = 0.5 * np.einsum("ij,ij->i", self.ivf_centroids, self.ivf_centroids)
        route = self.ivf_centroids @ q_proj - chalf               # higher = closer (L2)
        probes = (np.argpartition(-route, nprobe - 1)[:nprobe]
                  if nprobe < nlist else np.arange(nlist))

        bpv = (self.d // 64) * 8                                   # bytes per vector
        off = self.ivf_offsets
        code_parts, id_parts = [], []
        for c in probes:
            s, e = int(off[c]), int(off[c + 1])
            if e > s:
                code_parts.append(self.codes[s * bpv:e * bpv])
                id_parts.append(self.ivf_perm[s:e])
        if not code_parts:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        buf = np.concatenate(code_parts)                          # one contiguous scan buffer
        cand_ids = np.concatenate(id_parts)                       # gathered pos -> original id
        M = cand_ids.shape[0]

        q_bytes = self._hamming_query_bytes(q_proj)
        db, ib = _native.scan_hamming(buf, M, self.d, q_bytes, min(K_prime, M), num_threads)
        loc = np.frombuffer(ib, dtype=np.int64)
        dists = np.frombuffer(db, dtype=np.int64)
        ids = cand_ids[loc]

        if self.has_floats and K_prime > K:
            cand = np.array(self.floats_mmap[ids], dtype=np.float32)
            rescore = cand @ q_proj
            order = np.argsort(-rescore)[:K]
            return ids[order], rescore[order].astype(np.float32)
        order = np.argsort(dists)[:K]
        return ids[order], (-dists[order]).astype(np.float32)
