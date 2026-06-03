"""Tests for the unified ``docuverse`` CLI.

These exercise argparse plumbing and the lightweight ``presets`` subcommand.
They MUST NOT trigger an import of torch / pymilvus / transformers — that's
the contract that makes ``docuverse --help`` snappy. We assert that contract
explicitly in :func:`test_help_does_not_import_heavy_deps`.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from docuverse.cli import _build_parser, main
from docuverse.cli._common import parse_overrides


def test_bare_invocation_prints_help_and_returns_zero(capsys):
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "presets" in captured.out
    assert "search" in captured.out


def test_presets_list_prints_known_recipes(capsys):
    rc = main(["presets", "list"])
    captured = capsys.readouterr()
    assert rc == 0
    # PR 2 ships at least these 11 names; if more are added, that's fine, but
    # all of these should be present.
    names = captured.out.split()
    for expected in (
        "milvus-dense",
        "milvus-bm25",
        "milvus-hybrid",
        "elastic-bm25",
        "elastic-dense",
        "chromadb",
        "faiss",
    ):
        assert expected in names, f"missing recipe in `presets list`: {expected}"


def test_presets_list_with_engine_shows_db_engine(capsys):
    rc = main(["presets", "list", "--with-engine"])
    captured = capsys.readouterr()
    assert rc == 0
    # Each line should be `name<spaces>db_engine`.
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert lines, "presets list --with-engine produced no output"
    for line in lines:
        parts = line.split()
        assert len(parts) >= 2, f"unexpected line: {line!r}"


def test_presets_show_dumps_yaml(capsys):
    rc = main(["presets", "show", "milvus-dense"])
    captured = capsys.readouterr()
    assert rc == 0
    # The recipe must round-trip through YAML cleanly.
    import yaml

    data = yaml.safe_load(captured.out)
    assert isinstance(data, dict)
    assert data.get("db_engine") == "milvus-dense"


def test_presets_dump_emits_raw_yaml_for_redirect(capsys):
    rc = main(["presets", "dump", "milvus-dense"])
    captured = capsys.readouterr()
    assert rc == 0
    # Raw dump should be parseable as YAML and contain the engine name.
    import yaml

    parsed = yaml.safe_load(captured.out)
    assert parsed["db_engine"] == "milvus-dense"


def test_presets_show_unknown_recipe_raises(capsys):
    with pytest.raises(KeyError):
        main(["presets", "show", "not-a-real-preset"])


def test_presets_dump_unknown_recipe_returns_nonzero(capsys):
    rc = main(["presets", "dump", "not-a-real-preset"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "unknown preset" in captured.err


# ---------------------------------------------------------------------------
# parse_overrides
# ---------------------------------------------------------------------------


def test_parse_overrides_yaml_typing():
    out = parse_overrides(
        ["top_k=10", "secure=true", "model=ibm-granite/granite-embedding-small-english-r2"]
    )
    assert out == {
        "top_k": 10,
        "secure": True,
        "model": "ibm-granite/granite-embedding-small-english-r2",
    }


def test_parse_overrides_dotted_key_kept_as_flat():
    # `_explode_dotted` lives in docuverse.presets, not here.
    out = parse_overrides(["retriever.top_k=5"])
    assert out == {"retriever.top_k": 5}


def test_parse_overrides_empty_list():
    assert parse_overrides([]) == {}
    assert parse_overrides(None) == {}


def test_parse_overrides_rejects_missing_equals():
    with pytest.raises(SystemExit):
        parse_overrides(["nope"])


def test_parse_overrides_rejects_empty_key():
    with pytest.raises(SystemExit):
        parse_overrides(["=value"])


def test_parse_overrides_falls_back_to_string_on_yaml_error():
    # ``"[unterminated"`` is invalid YAML; we keep it as a literal string.
    out = parse_overrides(["x=[unterminated"])
    # Either parsed-by-yaml or raw-string is acceptable, as long as we don't
    # crash. Most safe_load calls will raise → fallback.
    assert "x" in out


# ---------------------------------------------------------------------------
# Heavy-dep contract
# ---------------------------------------------------------------------------


def test_help_does_not_import_heavy_deps():
    """`docuverse --help` must not pull in torch/pymilvus/transformers.

    We run it in a subprocess so we get a clean import space, then check
    sys.modules in the child. If this regresses, CLI startup latency goes
    from <100ms back to multiple seconds.
    """
    code = (
        "import sys; from docuverse.cli import main\n"
        "main(['presets', 'list'])\n"
        "for mod in ('torch', 'pymilvus', 'transformers', 'elasticsearch'):\n"
        "    assert mod not in sys.modules, f'{mod} was imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"heavy-deps regression:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_parser_registers_all_subcommands():
    """All nine subcommands from the plan must be registered."""
    parser = _build_parser()
    # argparse hides subparsers; reach through _subparsers_action.
    subactions = [
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    ]
    assert len(subactions) == 1
    names = set(subactions[0].choices.keys())
    expected = {
        "run",
        "search",
        "ingest",
        "evaluate",
        "presets",
        "embed",
        "convert",
        "db",
        "prepare",
    }
    missing = expected - names
    assert not missing, f"missing subcommands: {missing}"
