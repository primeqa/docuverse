"""numpy-side wrappers around the simdq C-extension pack helpers.

C entry points return raw `bytes` objects to avoid a numpy build-time
dependency in the extension. This module wraps them into typed numpy
arrays and validates input shapes before crossing the FFI boundary.

D is fixed at 768 in Plan 2 (the kernel template is templated on it);
inputs with a different second axis are rejected here.
"""
from __future__ import annotations

import numpy as np

from docuverse.engines.retrieval.simdq import _simdq_native as _native

D_FIXED = _native.D       # 768 in Plan 2; Plan 3 widens this


def _check_Y(Y: np.ndarray) -> np.ndarray:
    if Y.ndim != 2 or Y.shape[1] != D_FIXED:
        raise ValueError(
            f"simdq quantization: Y must have shape (N, {D_FIXED}); "
            f"got shape {Y.shape}"
        )
    if Y.dtype != np.float32:
        Y = Y.astype(np.float32, copy=False)
    return np.ascontiguousarray(Y)


def fit_scales(Y: np.ndarray) -> np.ndarray:
    """Compute per-vector fp32 scales (||y||/sqrt(D)).

    Returns a contiguous (N,) fp32 array. Zero-norm rows get a scale of 1.0
    so divide-by-scale during pack does not blow up.
    """
    Y = _check_Y(Y)
    raw = _native.compute_scales(Y)
    return np.frombuffer(raw, dtype=np.float32)


def pack(Y: np.ndarray, scales: np.ndarray, b: int) -> np.ndarray:
    """Quantize Y to b-bit dim-major SoA codes.

    Args:
        Y:       (N, D) fp32, contiguous.
        scales:  (N,) fp32, contiguous.
        b:       1, 2, or 4.

    Returns:
        codes: uint8 numpy array of length D * ceil(N / (8/b)).
    """
    if b not in (1, 2, 4):
        raise ValueError(f"simdq quantization: b must be one of (1, 2, 4); got {b}")
    Y = _check_Y(Y)
    N = Y.shape[0]
    if scales.dtype != np.float32:
        scales = scales.astype(np.float32, copy=False)
    if scales.shape != (N,):
        raise ValueError(
            f"simdq quantization: scales must have shape ({N},); "
            f"got {scales.shape}"
        )
    scales = np.ascontiguousarray(scales)
    if b == 1:
        raw = _native.pack_b1(Y, scales)
    elif b == 2:
        raw = _native.pack_b2(Y, scales)
    else:
        raw = _native.pack_b4(Y, scales)
    return np.frombuffer(raw, dtype=np.uint8)


def unpack_levels(codes: np.ndarray, N: int, D: int, b: int) -> np.ndarray:
    """Decode b-bit dim-major SoA codes to (N, D) int8 levels.

    This is the canonical reference unpack used in tests; the C extension
    has no "unpack to dense" entry point because the scan kernels operate
    directly on the packed layout. Slow Python loop is fine — tests use
    small N.
    """
    if b == 1:
        row_bytes = (N + 7) // 8
        out = np.empty((N, D), dtype=np.int8)
        for w in range(D):
            row = codes[w * row_bytes : (w + 1) * row_bytes]
            for i in range(N):
                bit = (row[i >> 3] >> (i & 7)) & 1
                out[i, w] = 1 if bit else -1
        return out
    if b == 2:
        row_bytes = (N + 3) // 4
        out = np.empty((N, D), dtype=np.int8)
        levels_b2 = np.array([-3, -1, 1, 3], dtype=np.int8)
        for w in range(D):
            row = codes[w * row_bytes : (w + 1) * row_bytes]
            for i in range(N):
                code = (row[i >> 2] >> ((i & 3) * 2)) & 0x3
                out[i, w] = levels_b2[code]
        return out
    if b == 4:
        row_bytes = (N + 1) // 2
        out = np.empty((N, D), dtype=np.int8)
        for w in range(D):
            row = codes[w * row_bytes : (w + 1) * row_bytes]
            for i in range(N):
                code = (row[i >> 1] >> ((i & 1) * 4)) & 0xF
                out[i, w] = 2 * int(code) - 15
        return out
    raise ValueError(f"unpack_levels: bad b={b}")
