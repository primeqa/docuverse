"""Shared helpers used by the ``cmd_*`` subcommand modules.

Nothing in here imports the heavy bits — engines, torch, pymilvus, or
transformers. Anything that needs them must be imported inside a subcommand's
``run()`` function so ``docuverse --help`` and ``docuverse presets list`` stay
fast and dependency-light.
"""
from __future__ import annotations

import argparse
from typing import Any, Sequence

import yaml


def add_preset_args(parser: argparse.ArgumentParser, *, require_one: bool = True) -> None:
    """Add the standard ``--preset`` / ``--config`` / ``--override`` flags.

    These three are the universal "how do I describe an engine" inputs. The
    precedence (matching the Python API ``SearchEngine.from_preset``) is::

        preset (base)  →  --config YAML  →  --override key=value

    ``require_one`` decides whether at least one of ``--preset`` / ``--config``
    must be supplied. ``docuverse run`` historically accepts a YAML config
    only, so subcommands can flip this off when they want different defaults.
    """
    group = parser.add_argument_group("engine config")
    group.add_argument(
        "--preset",
        metavar="NAME",
        help="Named recipe (e.g. milvus-dense). See `docuverse presets list`.",
    )
    group.add_argument(
        "--config",
        metavar="YAML",
        help="Path to a YAML/JSON config file (deep-merged on top of --preset).",
    )
    group.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override individual config keys. Right-hand side is parsed as YAML "
            "(so `top_k=10` is int, `secure=true` is bool). Dotted keys walk "
            "into nested dicts "
            "(e.g. `retriever.model_name=ibm-granite/granite-embedding-small-english-r2`)."
        ),
    )
    parser._dv_require_engine_source = require_one  # type: ignore[attr-defined]


def parse_overrides(items: Sequence[str] | None) -> dict[str, Any]:
    """Parse ``["k=v", "x.y=2", ...]`` into a flat dict.

    Values are run through ``yaml.safe_load`` so that ``"10"`` becomes int,
    ``"true"`` becomes bool, ``"ibm-granite/granite-embedding-small-english-r2"``
    stays a string, and JSON-ish
    snippets (``'[1,2]'``, ``'{a: 1}'``) become lists/dicts. The dotted-key
    explosion to nested dicts happens later, inside
    ``docuverse.presets.deep_merge_overrides``.
    """
    if not items:
        return {}
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"--override expects KEY=VALUE pairs; got {item!r} (no '=')."
            )
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--override pair {item!r} has empty key.")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw  # treat unparseable RHS as a literal string
        out[key] = value
    return out


def build_engine_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve ``args.preset`` / ``args.config`` / ``args.override`` to a dict.

    The result is a plain dict suitable for ``SearchEngine.from_dict`` or
    ``DocUVerseConfig(...)``. Loaded here (not deferred to engine construction)
    so missing presets fail fast before we import torch.
    """
    from docuverse.presets import deep_merge_overrides, load_preset
    from docuverse.utils import read_config_file

    preset = getattr(args, "preset", None)
    config_path = getattr(args, "config", None)
    overrides = parse_overrides(getattr(args, "override", None))

    require_one = getattr(args, "_dv_require_engine_source", True)
    if require_one and not (preset or config_path):
        raise SystemExit("error: one of --preset or --config is required.")

    base: dict[str, Any] = {}
    if preset:
        base = load_preset(preset)
    if config_path:
        from_file = read_config_file(config_path) or {}
        if not isinstance(from_file, dict):
            raise SystemExit(f"--config {config_path}: did not parse to a dict.")
        base = deep_merge_overrides(base, from_file)
    if overrides:
        base = deep_merge_overrides(base, overrides)
    return base


def build_engine(args: argparse.Namespace):
    """Build a ``SearchEngine`` from CLI args. Imports the engine lazily."""
    config = build_engine_config_dict(args)
    from docuverse import SearchEngine

    return SearchEngine.from_dict(config)
