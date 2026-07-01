"""DocUVerse top-level package.

Re-exports are lazy so that importing :mod:`docuverse.cli` (which imports the
parent package as a side effect) doesn't drag in torch / transformers /
pymilvus. ``from docuverse import SearchEngine`` keeps working — the symbol
is materialized on first access via :pep:`562` ``__getattr__``.
"""
from __future__ import annotations

# macOS OpenMP-runtime preload. torch, sklearn, and faiss each bundle their
# own libomp.dylib; the simdq C extension (_simdq_native) links against the
# Homebrew libomp. If any of those torch/sklearn/faiss copies loads first,
# the simdq extension's OpenMP regions segfault in a way that's not
# recoverable without a process restart. Preloading Homebrew's libomp with
# RTLD_GLOBAL at package init is the only reliable order-independent fix.
# Silent no-op if the file is missing (Linux, unbrewed macOS installs).
import os as _os
import sys as _sys

if _sys.platform == "darwin":
    import ctypes as _ctypes
    for _libomp in (
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/usr/local/opt/libomp/lib/libomp.dylib",
    ):
        if _os.path.exists(_libomp):
            try:
                _ctypes.CDLL(_libomp, mode=_ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break
    del _ctypes
del _os, _sys

# Cheap re-exports stay eager (no heavy deps).
from .presets import list_presets, load_preset

__all__ = [
    "SearchCorpus",
    "SearchEngine",
    "SearchQueries",
    "SearchResult",
    "list_presets",
    "load_preset",
]

# Map symbol → fully qualified import target. Resolved on demand by __getattr__.
_LAZY_EXPORTS = {
    "SearchCorpus": ("docuverse.engines", "SearchCorpus"),
    "SearchEngine": ("docuverse.engines", "SearchEngine"),
    "SearchQueries": ("docuverse.engines", "SearchQueries"),
    "SearchResult": ("docuverse.engines", "SearchResult"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module_path, attr = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_path), attr)
        globals()[name] = value  # cache so subsequent lookups are direct
        return value
    raise AttributeError(f"module 'docuverse' has no attribute {name!r}")


def __dir__():  # pragma: no cover - cosmetic
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
