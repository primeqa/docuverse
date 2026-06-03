"""``docuverse ingest`` — populate an index from a passages file."""
from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ingest",
        help="Ingest documents into the configured engine's index.",
    )
    from docuverse.cli._common import add_preset_args

    add_preset_args(p, require_one=True)
    p.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Path to the passages file (JSONL or whatever the data_format expects).",
    )
    p.add_argument(
        "--index",
        metavar="NAME",
        help="Override `index_name` for this run.",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="Update the index instead of recreating it.",
    )
    p.set_defaults(_run=_run)


def _run(args: argparse.Namespace) -> int:
    from docuverse.cli._common import build_engine_config_dict
    from docuverse.engines.search_engine_config_params import DocUVerseConfig

    config_dict = build_engine_config_dict(args)
    config_dict["input_passages"] = args.input
    if args.index:
        config_dict["index_name"] = args.index
    config_dict["ingest"] = True
    config_dict["update"] = bool(args.update)

    config = DocUVerseConfig(config_dict)
    from docuverse import SearchEngine

    engine = SearchEngine(config, name="docuverse-ingest")
    corpus = engine.read_data(args.input)
    engine.ingest(corpus, update=args.update)
    return 0
