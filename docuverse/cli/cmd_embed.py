"""``docuverse embed`` — wrapper around scripts/compute_embeddings_from_jsonl.py."""
from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "embed",
        help="Compute embeddings for a JSONL passage file (script wrapper).",
        description=(
            "Thin wrapper around scripts/compute_embeddings_from_jsonl.py. "
            "Pass-through args land in `argv`; use `docuverse embed --help` "
            "for the underlying script's flags."
        ),
        add_help=False,  # let the underlying script print its own help
    )
    p.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying script.",
    )
    p.set_defaults(_run=_run)


def _run(args: argparse.Namespace) -> int:
    from docuverse.cli._delegate import run_main_with_argv

    return run_main_with_argv(
        "scripts.compute_embeddings_from_jsonl",
        "main",
        args.passthrough,
        argv0="docuverse-embed",
    )
