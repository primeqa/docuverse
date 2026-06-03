"""``docuverse evaluate`` — score a results file against gold qrels."""
from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "evaluate",
        help="Compute retrieval metrics from a results file + queries/qrels.",
    )
    from docuverse.cli._common import add_preset_args

    # Evaluation needs a config to know data_template / eval_config. Either a
    # full config or a preset is fine.
    add_preset_args(p, require_one=False)
    p.add_argument(
        "--results",
        required=True,
        metavar="FILE",
        help="Search results file (.json/.jsonl/.pkl) produced by `docuverse search` or `run`.",
    )
    p.add_argument(
        "--queries",
        required=True,
        metavar="FILE",
        help="Queries (with gold) file.",
    )
    p.set_defaults(_run=_run)


def _run(args: argparse.Namespace) -> int:
    from docuverse.cli._common import build_engine_config_dict
    from docuverse.engines.search_engine_config_params import DocUVerseConfig

    config_dict = build_engine_config_dict(args)
    config_dict["input_queries"] = args.queries
    config_dict["output_file"] = args.results

    config = DocUVerseConfig(config_dict)
    from docuverse import SearchEngine
    from docuverse.utils.evaluator import EvaluationEngine

    engine = SearchEngine(config, name="docuverse-evaluate")
    queries = engine.read_questions(args.queries)
    results = engine.read_output(args.results)
    scorer = EvaluationEngine(config)
    scored = scorer.compute_score(queries, results, model_name=engine.get_output_name())
    print(scored)
    return 0
