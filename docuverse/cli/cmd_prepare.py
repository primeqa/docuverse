"""``docuverse prepare`` — dataset preparation helpers (HF download, etc.)."""
from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "prepare",
        help="Prepare datasets (download from HuggingFace, etc.).",
    )
    sub = p.add_subparsers(dest="prepare_command", metavar="<action>")

    hf = sub.add_parser(
        "hf",
        help="Download a HuggingFace dataset to JSONL.",
        add_help=False,
    )
    hf.add_argument("passthrough", nargs=argparse.REMAINDER)
    hf.set_defaults(_run=_run_hf)

    p.set_defaults(_run=_run_default)


def _run_default(args: argparse.Namespace) -> int:
    print("usage: docuverse prepare {hf} ...")
    return 2


def _run_hf(args: argparse.Namespace) -> int:
    from docuverse.cli._delegate import run_main_with_argv

    return run_main_with_argv(
        "scripts.download_hf_dataset_to_jsonl",
        "main",
        args.passthrough,
        argv0="docuverse-prepare-hf",
    )
