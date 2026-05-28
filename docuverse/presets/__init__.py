"""Named recipes for `SearchEngine.from_preset(...)`.

Recipes are YAML files shipped as package data under
``docuverse/presets/recipes/``. They are dual-purpose:

1. Loaded by ``SearchEngine.from_preset(name, **overrides)`` to give users a
   one-line entry point ("give me a default Milvus dense engine") without
   writing a YAML config.
2. Surfaced by ``docuverse presets show NAME`` and
   ``docuverse presets dump NAME > my.yaml`` so users can copy and edit the
   full configuration when they outgrow the defaults.

The override merge happens here so it can be reused by both the Python API
and the CLI. See ``deep_merge_overrides`` for the supported syntax.
"""
from __future__ import annotations

from copy import deepcopy
from importlib import resources
from typing import Any

import yaml

# All recipe lookups go through this. Use ``importlib.resources`` so it works
# both when the package is installed normally and when developed in-tree
# (``pip install -e .``).
_RECIPE_PACKAGE = "docuverse.presets.recipes"


def _recipe_path(name: str):
    return resources.files(_RECIPE_PACKAGE) / f"{name}.yaml"


def list_presets() -> list[str]:
    """Return the sorted list of available preset names."""
    pkg = resources.files(_RECIPE_PACKAGE)
    names: list[str] = []
    for entry in pkg.iterdir():
        # ``entry.name`` works for both Path-backed and zip-backed Traversables.
        nm = entry.name
        if nm.endswith(".yaml") and not nm.startswith("_"):
            names.append(nm[: -len(".yaml")])
    return sorted(names)


def load_preset(name: str) -> dict[str, Any]:
    """Load a preset recipe into a plain dict.

    Raises ``KeyError`` (with the available names) if ``name`` is unknown,
    so callers get a discoverable error instead of FileNotFoundError.
    """
    path = _recipe_path(name)
    if not path.is_file():
        available = ", ".join(list_presets())
        raise KeyError(
            f"unknown preset {name!r}; available: {available}. "
            f"See docuverse.presets.list_presets()."
        )
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"preset {name!r} did not parse to a dict (got {type(data).__name__})")
    return data


# ---------------------------------------------------------------------------
# Override deep-merge
# ---------------------------------------------------------------------------


def _explode_dotted(overrides: dict[str, Any]) -> dict[str, Any]:
    """Turn ``{"retriever.top_k": 10}`` into ``{"retriever": {"top_k": 10}}``.

    Each item is exploded individually and recursively merged into the
    running result, so mixed dotted-and-nested forms compose rather than
    clobber:

        {"x.y": 1, "x": {"z": 2}}  →  {"x": {"y": 1, "z": 2}}
    """
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        if "." not in key:
            piece: dict[str, Any] = {key: value}
        else:
            parts = key.split(".")
            piece = current = {}
            for part in parts[:-1]:
                nxt: dict[str, Any] = {}
                current[part] = nxt
                current = nxt
            current[parts[-1]] = value
        _merge_in_place(out, piece)
    return out


def _merge_in_place(base: dict[str, Any], top: dict[str, Any]) -> None:
    """Recursive dict merge; ``top`` wins on scalars and lists."""
    for key, value in top.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _merge_in_place(base[key], value)
        else:
            base[key] = value


def deep_merge_overrides(
    base: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deep-merge ``overrides`` into a copy of ``base`` and return the result.

    Override keys may be flat (``{"top_k": 10}``), nested
    (``{"retriever": {"top_k": 10}}``), or dotted
    (``{"retriever.top_k": 10}``). Dotted keys are exploded into nested dicts
    before merging, so all three forms are equivalent.

    Nested dicts merge recursively. Lists and scalars REPLACE the base value.
    Unknown keys pass through and will be caught downstream by
    ``HfArgumentParser.parse_dict(allow_extra_keys=True)``.
    """
    merged = deepcopy(base)
    if overrides:
        _merge_in_place(merged, _explode_dotted(overrides))
    return merged


__all__ = [
    "list_presets",
    "load_preset",
    "deep_merge_overrides",
]
