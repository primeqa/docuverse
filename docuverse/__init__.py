"""DocUVerse top-level package.

Re-exports are lazy so that importing :mod:`docuverse.cli` (which imports the
parent package as a side effect) doesn't drag in torch / transformers /
pymilvus. ``from docuverse import SearchEngine`` keeps working — the symbol
is materialized on first access via :pep:`562` ``__getattr__``.
"""
from __future__ import annotations

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
