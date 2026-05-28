"""``docuverse presets {list,show,dump}`` — recipe discovery.

This subcommand intentionally has no heavy imports. It only touches
``docuverse.presets`` (stdlib + PyYAML), so it is safe to invoke without any
optional engine dependency installed.
"""
from __future__ import annotations

import argparse
import sys

import yaml


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "presets",
        help="List, show, or dump named recipe configurations.",
        description=(
            "Recipes are YAML files shipped with docuverse. Use them via "
            "`SearchEngine.from_preset(NAME)` in Python or "
            "`--preset NAME` on the CLI."
        ),
    )
    psub = p.add_subparsers(dest="presets_command", metavar="<action>")

    pl = psub.add_parser("list", help="List available preset names.")
    pl.add_argument(
        "--with-engine",
        action="store_true",
        help="Also print each recipe's `db_engine` value.",
    )
    pl.set_defaults(_run=_run_list)

    ps = psub.add_parser("show", help="Show a recipe's parsed config.")
    ps.add_argument("name", help="Recipe name (see `docuverse presets list`).")
    ps.set_defaults(_run=_run_show)

    pd = psub.add_parser(
        "dump",
        help="Print a recipe's raw YAML to stdout (for `> my-recipe.yaml`).",
    )
    pd.add_argument("name", help="Recipe name (see `docuverse presets list`).")
    pd.set_defaults(_run=_run_dump)

    p.set_defaults(_run=_run_default)


def _run_default(args: argparse.Namespace) -> int:
    # No subaction supplied -> show the same help as `docuverse presets`.
    print("usage: docuverse presets {list, show, dump} ...", file=sys.stderr)
    print(
        "Try `docuverse presets list` to see available recipes.",
        file=sys.stderr,
    )
    return 2


def _run_list(args: argparse.Namespace) -> int:
    from docuverse.presets import list_presets, load_preset

    names = list_presets()
    if not args.with_engine:
        for name in names:
            print(name)
        return 0
    width = max((len(n) for n in names), default=0)
    for name in names:
        engine = load_preset(name).get("db_engine", "?")
        print(f"{name.ljust(width)}  {engine}")
    return 0


def _run_show(args: argparse.Namespace) -> int:
    from docuverse.presets import load_preset

    data = load_preset(args.name)
    yaml.safe_dump(data, sys.stdout, sort_keys=False, default_flow_style=False)
    return 0


def _run_dump(args: argparse.Namespace) -> int:
    """Round-trip the raw YAML so users can pipe it into a file and edit."""
    from importlib import resources

    path = resources.files("docuverse.presets.recipes") / f"{args.name}.yaml"
    if not path.is_file():
        from docuverse.presets import list_presets

        print(
            f"unknown preset {args.name!r}; available: {', '.join(list_presets())}",
            file=sys.stderr,
        )
        return 1
    sys.stdout.write(path.read_text())
    return 0
