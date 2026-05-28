"""``docuverse run`` — full ingest+search+evaluate pipeline.

Mirrors the legacy ``ingest-and-test`` console script but routed through the
new preset/override stack. When ``--config`` alone is given (and no preset),
the call shape is identical to ``ingest-and-test --config FILE``.
"""
from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Run an ingest + search + evaluate pipeline (legacy ingest-and-test).",
        description=(
            "Equivalent to the historical `ingest-and-test` script. Accepts the "
            "same --preset / --config / --override stack as the rest of the CLI."
        ),
    )

    # Re-use the shared engine source flags. `run` does not strictly require
    # one because legacy callers might pass everything via --override; the
    # downstream DocUVerseConfig builder will complain if it's truly empty.
    from docuverse.cli._common import add_preset_args

    add_preset_args(p, require_one=False)
    p.set_defaults(_run=_run)


def _run(args: argparse.Namespace) -> int:
    """Drive the full pipeline.

    We intentionally do NOT delegate to ``ingest_and_test.main_cli`` because
    that function reads ``sys.argv`` directly. Instead we replicate its body
    against an engine we built ourselves, keeping behavior identical.
    """
    from docuverse.cli._common import build_engine_config_dict
    from docuverse.engines.search_engine_config_params import DocUVerseConfig

    config_dict = build_engine_config_dict(args)
    if not config_dict:
        # Fall back to the legacy stdargs path so `docuverse run` with no flags
        # remains a transparent alias for the old `ingest-and-test` (it parses
        # sys.argv via HfArgumentParser).
        from docuverse.utils.ingest_and_test import main_cli

        main_cli()
        return 0

    config = DocUVerseConfig(config_dict)
    from docuverse import SearchEngine

    engine = SearchEngine(config, name="docuverse-run")
    if config.ingest or config.update:
        corpus = engine.read_data(config.input_passages)
        engine.ingest(corpus, update=config.update, skip=getattr(config, "skip", False))
    if config.retrieve:
        queries = engine.read_questions(config.input_queries)
        results = engine.search(queries)
        engine.write_output(results)
        if config.evaluate and getattr(config, "eval_config", None) is not None:
            scored = engine.compute_score(queries, results)
            print(scored)
    return 0
