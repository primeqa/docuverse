"""The unified ``docuverse`` CLI.

Each subcommand lives in its own ``cmd_*.py`` module and registers itself with
the top-level argparse parser via ``register(subparsers)``. Heavy imports
(``torch``, ``pymilvus``, ``transformers``, the engine module) are pushed
inside each subcommand's ``run()`` function so plumbing operations like
``docuverse presets list`` and ``docuverse --help`` start instantly without
paying for them.

The console-script entry point is wired up in ``pyproject.toml`` as::

    [project.scripts]
    docuverse = "docuverse.cli:main"
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docuverse",
        description="DocUVerse — unified retrieval/search CLI.",
    )
    parser.set_defaults(_run=None)
    sub = parser.add_subparsers(
        title="subcommands",
        dest="command",
        metavar="<command>",
    )

    # Import subcommand modules lazily-but-eagerly here: each module's
    # ``register`` only configures argparse, it does not import the engine.
    from docuverse.cli import (
        cmd_convert,
        cmd_db,
        cmd_embed,
        cmd_evaluate,
        cmd_ingest,
        cmd_prepare,
        cmd_presets,
        cmd_run,
        cmd_search,
    )

    for mod in (
        cmd_run,
        cmd_search,
        cmd_ingest,
        cmd_evaluate,
        cmd_presets,
        cmd_embed,
        cmd_convert,
        cmd_db,
        cmd_prepare,
    ):
        mod.register(sub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``docuverse`` console script.

    Returns an exit code so it composes with ``sys.exit`` and tests that
    invoke ``main([...])`` directly.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args._run is None:
        parser.print_help()
        return 0
    rc = args._run(args)
    return 0 if rc is None else int(rc)


if __name__ == "__main__":  # pragma: no cover - convenience for ``python -m``.
    sys.exit(main())
