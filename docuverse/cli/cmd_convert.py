"""``docuverse convert`` — format conversion utilities."""
from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "convert",
        help="Convert between data formats (json↔jsonl, qrels→TREC, JSONL field-ops).",
    )
    csub = p.add_subparsers(dest="convert_command", metavar="<format>")

    j2j = csub.add_parser(
        "json-to-jsonl",
        help="Convert a JSON array file to JSONL.",
        add_help=False,
    )
    j2j.add_argument("passthrough", nargs=argparse.REMAINDER)
    j2j.set_defaults(_run=_run_json_to_jsonl)

    trec = csub.add_parser(
        "to-trec",
        help="Convert qrels to TREC qrels format.",
        add_help=False,
    )
    trec.add_argument("passthrough", nargs=argparse.REMAINDER)
    trec.set_defaults(_run=_run_to_trec)

    p.set_defaults(_run=_run_default)


def _run_default(args: argparse.Namespace) -> int:
    print("usage: docuverse convert {json-to-jsonl, to-trec} ...")
    return 2


def _run_json_to_jsonl(args: argparse.Namespace) -> int:
    from docuverse.cli._delegate import run_main_with_argv

    return run_main_with_argv(
        "scripts.json_to_jsonl_converter",
        "main",
        args.passthrough,
        argv0="docuverse-convert-json-to-jsonl",
    )


def _run_to_trec(args: argparse.Namespace) -> int:
    from docuverse.cli._delegate import run_main_with_argv

    return run_main_with_argv(
        "scripts.convert_to_trec_qrels",
        "main",
        args.passthrough,
        argv0="docuverse-convert-to-trec",
    )
