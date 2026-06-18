"""simdq — SIMD-quantized in-process retrieval engine.

Plan 2 ships SimdqIndex (the storage + scan layer). Plan 3 will add
SimdqEngine (the DocUVerse SearchEngine wrapper) and the recipe sweep.
"""
from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex

__all__ = ["SimdqIndex"]
