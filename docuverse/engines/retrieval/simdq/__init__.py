"""simdq — SIMD-quantized in-process retrieval engine.

Note: on macOS, the C extension links against Homebrew's libomp; the
top-level ``docuverse/__init__.py`` preloads that libomp with RTLD_GLOBAL
so it wins symbol resolution against the libomp copies bundled by torch /
sklearn / faiss. Do not import ``_simdq_native`` from outside a
``docuverse``-rooted import path on macOS without setting up that preload
yourself (see ``docuverse/__init__.py``).
"""
from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
from docuverse.engines.retrieval.simdq.simdq_engine import SimdqEngine

__all__ = ["SimdqIndex", "SimdqEngine"]
