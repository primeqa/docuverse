"""Locks the quickstart example so the README snippet stays runnable.

We don't actually ingest into Milvus-Lite here — that needs torch on CI and
takes minutes. Instead we assert that the files are well-formed, that the
recipe's preset reference still exists, and that the override-merge produces
the same dict the README claims it does.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "quickstart"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_passages_jsonl_is_well_formed():
    rows = _read_jsonl(EXAMPLE / "passages.jsonl")
    assert len(rows) == 10, "quickstart corpus is documented as 10 passages"
    for row in rows:
        assert {"id", "title", "text"} <= row.keys()
        assert isinstance(row["text"], str) and row["text"]


def test_queries_jsonl_is_well_formed():
    rows = _read_jsonl(EXAMPLE / "queries.jsonl")
    assert len(rows) == 5, "quickstart corpus is documented as 5 queries"
    for row in rows:
        assert {"id", "text", "relevant"} <= row.keys()
        assert isinstance(row["relevant"], list) and row["relevant"]


def test_qrels_match_query_relevance():
    """qrels.tsv should encode the same gold judgments as queries.jsonl's
    ``relevant`` field. If they drift, downstream evaluation tools that read
    one or the other will silently disagree.
    """
    queries = {q["id"]: set(q["relevant"]) for q in _read_jsonl(EXAMPLE / "queries.jsonl")}
    qrels: dict[str, set[str]] = {}
    for line in (EXAMPLE / "qrels.tsv").read_text().splitlines():
        if not line.strip():
            continue
        qid, _zero, doc_id, rel = line.split("\t")
        if int(rel) > 0:
            qrels.setdefault(qid, set()).add(doc_id)
    assert queries == qrels, (
        f"qrels.tsv and queries.jsonl disagree:\n"
        f"  queries: {queries}\n"
        f"  qrels:   {qrels}"
    )


def test_recipe_yaml_is_loadable_and_uses_known_engine():
    recipe = yaml.safe_load((EXAMPLE / "recipe.yaml").read_text())
    assert recipe["db_engine"] == "milvus-dense"
    assert recipe["server"].startswith("file:"), (
        "quickstart must default to embedded Milvus-Lite for zero-setup runs"
    )
    # Paths in the recipe should point to real example files.
    assert (
        EXAMPLE.parent.parent / recipe["input_passages"]
    ).exists(), recipe["input_passages"]
    assert (
        EXAMPLE.parent.parent / recipe["input_queries"]
    ).exists(), recipe["input_queries"]


def test_recipe_yaml_matches_milvus_dense_preset_overrides():
    """The recipe should be a strict superset of `milvus-dense`'s defaults
    (apart from the keys it explicitly overrides). If the preset changes
    shape, this test forces us to update the recipe and the README in lockstep.
    """
    from docuverse.presets import deep_merge_overrides, load_preset

    recipe = yaml.safe_load((EXAMPLE / "recipe.yaml").read_text())
    preset = load_preset("milvus-dense")
    merged = deep_merge_overrides(preset, recipe)
    # Preset's db_engine should survive (the recipe sets the same value).
    assert merged["db_engine"] == "milvus-dense"
    # Recipe's index_name should win over the preset default.
    assert merged["index_name"] == "docuverse_quickstart"


def test_readme_quickstart_paths_exist():
    """The README's quickstart code block references three example files.
    If any goes missing, copy-pasting from README into a Python REPL breaks.
    """
    for relpath in (
        "examples/quickstart/passages.jsonl",
        "examples/quickstart/queries.jsonl",
        "examples/quickstart/recipe.yaml",
    ):
        path = EXAMPLE.parent.parent / relpath
        assert path.exists(), f"README references missing file: {relpath}"


@pytest.mark.parametrize(
    "doc",
    ["docs/quickstart.rst", "docs/presets.rst", "docs/cli.rst", "docs/index.rst"],
)
def test_docs_pages_exist(doc):
    path = EXAMPLE.parent.parent / doc
    assert path.exists(), f"missing docs page: {doc}"
    # Sanity: must not be empty.
    assert path.read_text().strip()


# ---------------------------------------------------------------------------
# Granite-only-models constraint
# ---------------------------------------------------------------------------

# The user-facing quickstart and examples must reference only IBM Granite
# r2 embedding models — they're permissively licensed and ship with the
# project's expected baseline numbers. Any other model name in these files
# is a regression we want to catch in CI.
_ALLOWED_MODELS = {
    "ibm-granite/granite-embedding-small-english-r2",
    "ibm-granite/granite-embedding-97m-multilingual-r2",
    # Sparse companion model used by milvus-sparse / milvus-splade recipes
    # in place of the original SPLADE checkpoint.
    "ibm-granite/granite-embedding-30m-sparse",
}

# Heuristic prefixes that signal a model identifier on a HuggingFace-style
# `org/repo` line. We don't want to regex every word — false positives there
# are noisy. These are the families we historically had references to.
_BANNED_MODEL_TOKENS = (
    "BAAI/",
    "sentence-transformers/",
    "cross-encoder/",
    "naver/splade",
    "hf-internal-testing/",
    "intfloat/",
    "ibm-granite/granite-embedding-30m",  # older r1-era model
)


def _scan_for_models(text: str) -> list[str]:
    """Return every banned model token that appears in ``text``."""
    return [tok for tok in _BANNED_MODEL_TOKENS if tok in text]


@pytest.mark.parametrize(
    "relpath",
    [
        "examples/quickstart/recipe.yaml",
        "examples/quickstart/README.md",
        "docs/quickstart.rst",
        "docs/presets.rst",
        "README.md",
    ],
)
def test_quickstart_files_only_reference_granite_r2(relpath):
    path = EXAMPLE.parent.parent / relpath
    text = path.read_text()
    found = _scan_for_models(text)
    assert not found, (
        f"{relpath} references disallowed model(s) {found}; "
        f"quickstart/examples must use only {_ALLOWED_MODELS}"
    )


def test_quickstart_recipe_uses_allowed_model():
    """recipe.yaml's ``model_name`` must be one of the two Granite r2 models."""
    recipe = yaml.safe_load((EXAMPLE / "recipe.yaml").read_text())
    assert recipe.get("model_name") in _ALLOWED_MODELS, (
        f"recipe.yaml model_name={recipe.get('model_name')!r} is not in {_ALLOWED_MODELS}"
    )


def test_shipped_recipes_only_reference_granite():
    """Every recipe under ``docuverse/presets/recipes/`` that sets a
    ``model_name`` must use one of the allowed Granite identifiers.

    Recipes that don't set ``model_name`` (e.g. ``elastic-bm25``,
    ``milvus-bm25``, ``elastic-elser``) are skipped — those engines either
    don't need an embedding model client-side or use a server-deployed model.
    """
    from docuverse.presets import list_presets, load_preset

    offenders: list[tuple[str, str]] = []
    for name in list_presets():
        recipe = load_preset(name)
        model = recipe.get("model_name")
        if model is None:
            continue
        if model not in _ALLOWED_MODELS:
            offenders.append((name, model))
    assert not offenders, (
        f"shipped recipes must reference only {_ALLOWED_MODELS}; "
        f"offending recipes: {offenders}"
    )
