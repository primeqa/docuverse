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
        _show_ingested(corpus)
    if config.retrieve:
        queries = engine.read_questions(config.input_queries)
        results = engine.search(queries)
        engine.write_output(results)
        _show_results(queries, results, top_k=min(3, getattr(config, "top_k", 3) or 3))
        out = getattr(config, "output_file", None)
        if out:
            print(f"\nFull results written to {out}")
        if config.evaluate and getattr(config, "eval_config", None) is not None:
            scored = engine.compute_score(queries, results)
            print("\nEvaluation:")
            print(scored)
    return 0


def _truncate(s, n: int = 120) -> str:
    s = str(s).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _show_ingested(corpus, sample: int = 3) -> None:
    """Print a short summary of what was ingested."""
    try:
        total = len(corpus)
    except TypeError:
        corpus = list(corpus)
        total = len(corpus)

    print(f"\nIngested {total} document(s). Showing first {min(sample, total)}:")
    for i, doc in enumerate(corpus[:sample]):
        get = doc.get if hasattr(doc, "get") else (lambda k, d=None, _d=doc: getattr(_d, k, d))
        doc_id = get("id", "?")
        title = get("title", "")
        text = get("text", "")
        header = f"  [{i + 1}] id={doc_id}"
        if title:
            header += f"  title={title!r}"
        print(header)
        print(f"      {_truncate(text, 140)}")


def _show_results(queries, results, top_k: int = 3, max_queries: int = 5) -> None:
    """Print a per-query summary of the top retrieved passages."""
    n_queries = len(results)
    shown = min(n_queries, max_queries)
    print(f"\nSearch results ({n_queries} query/queries; showing first {shown}, top {top_k} hits each):")
    for qi, result in enumerate(results[:shown]):
        question = getattr(result, "question", None)
        q_text = getattr(question, "text", None) or (
            question.get("text") if hasattr(question, "get") else "?"
        )
        q_id = getattr(question, "id", None) or (
            question.get("id") if hasattr(question, "get") else "?"
        )
        print(f"\n  Q{qi + 1} (id={q_id}): {_truncate(q_text, 140)}")
        passages = list(result)[:top_k]
        if not passages:
            print("      (no hits)")
            continue
        for ri, p in enumerate(passages):
            pid = p.get("id", p.get("_id", "?"))
            score = p.get("score", p.get("rank", None))
            try:
                text = p.get_text()
            except Exception:
                text = p.get("text", "")
            score_str = f"  score={score:.4f}" if isinstance(score, float) else (
                f"  score={score}" if score is not None else ""
            )
            print(f"      {ri + 1}. id={pid}{score_str}")
            print(f"         {_truncate(text, 140)}")
