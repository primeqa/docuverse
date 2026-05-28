"""``docuverse search`` — one-shot retrieval against an existing index.

Builds a ``SearchEngine`` from preset/config/overrides, then runs either a
single ``--query`` string or a ``--queries FILE`` of multiple queries against
the configured index. Output is JSON Lines on stdout for trivial piping.
"""
from __future__ import annotations

import argparse
import json
import sys


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "search",
        help="Search an already-ingested index.",
        description="Run a query (or queries file) against the configured engine.",
    )
    from docuverse.cli._common import add_preset_args

    add_preset_args(p, require_one=True)

    qgroup = p.add_mutually_exclusive_group(required=True)
    qgroup.add_argument("--query", help="A single query string.")
    qgroup.add_argument(
        "--queries",
        metavar="FILE",
        help="Path to a queries file (JSONL/TSV/whatever the data_format expects).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k (default: whatever the preset/config sets).",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help="Write JSON Lines results to FILE instead of stdout.",
    )
    p.set_defaults(_run=_run)


def _run(args: argparse.Namespace) -> int:
    from docuverse.cli._common import build_engine_config_dict
    from docuverse.engines.search_queries import SearchQueries
    from docuverse.engines.search_engine_config_params import DocUVerseConfig

    config_dict = build_engine_config_dict(args)
    if args.top_k is not None:
        config_dict["top_k"] = args.top_k
    if args.queries and "input_queries" not in config_dict:
        config_dict["input_queries"] = args.queries

    config = DocUVerseConfig(config_dict)
    from docuverse import SearchEngine

    engine = SearchEngine(config, name="docuverse-search")

    if args.query:
        # Build a single Query object using the engine's query template, so
        # text_header / id_header etc. land in the right slots.
        q = SearchQueries.Query(template=engine.config.query_template, text=args.query, id="cli-0")
        queries = [q]
    else:
        queries = engine.read_questions(args.queries)

    results = engine.search(queries)

    out = sys.stdout if args.output is None else open(args.output, "w")
    try:
        for r in results:
            out.write(json.dumps(r.as_dict()) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
    return 0
