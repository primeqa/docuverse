"""``docuverse db`` — Milvus collection administration helpers."""
from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "db",
        help="Database admin: stats, list, copy, query (Milvus).",
    )
    sub = p.add_subparsers(dest="db_command", metavar="<action>")

    for name, helptext, module in (
        ("stats", "Print Milvus DB stats.", "scripts.milvus_utils.milvus_db_stats"),
        ("list", "List Milvus collections.", "scripts.milvus_utils.milvus_list_collections"),
        ("copy", "Copy data between Milvus DBs (db2db).", "scripts.milvus_utils.db2db_copy"),
        ("query", "Run a query against a Milvus collection.", "scripts.milvus_utils.milvus_query"),
    ):
        sp = sub.add_parser(name, help=helptext, add_help=False)
        sp.add_argument("passthrough", nargs=argparse.REMAINDER)
        sp.set_defaults(_run=_make_runner(module), _argv0=f"docuverse-db-{name}")

    p.set_defaults(_run=_run_default)


def _run_default(args: argparse.Namespace) -> int:
    print("usage: docuverse db {stats, list, copy, query} ...")
    return 2


def _make_runner(module: str):
    def _run(args: argparse.Namespace) -> int:
        from docuverse.cli._delegate import run_main_with_argv

        return run_main_with_argv(
            module,
            "main",
            args.passthrough,
            argv0=getattr(args, "_argv0", "docuverse"),
        )

    return _run
