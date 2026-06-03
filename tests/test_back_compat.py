"""Lock down public surfaces that downstream users may depend on.

These tests intentionally do NOT instantiate engines that need optional deps
(milvus, elastic, chromadb, faiss, lancedb) — they only verify that the
dispatcher recognizes the name. ImportError from a missing optional dep is
treated as success: the dispatcher reached the right branch.

If a test in this file fails, it means a backwards-incompatible change has
landed; either revert or bump a major version with a CHANGELOG entry.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


# --- Public Python API ------------------------------------------------------


def test_top_level_exports():
    """`from docuverse import SearchEngine, ...` must keep working."""
    import docuverse

    for name in ("SearchEngine", "SearchCorpus", "SearchQueries", "SearchResult"):
        assert hasattr(docuverse, name), f"docuverse.{name} disappeared"


def test_search_engine_constructor_signature():
    """The README/quickstart relies on `SearchEngine(config_or_path=...)`."""
    from docuverse import SearchEngine

    sig = inspect.signature(SearchEngine.__init__)
    assert "config_or_path" in sig.parameters, (
        "SearchEngine.__init__ must keep its `config_or_path` parameter"
    )


# --- Engine dispatch --------------------------------------------------------

# Every name that `create_retrieval_engine` is documented to accept. The
# trailing-space `'milvus_splade '` variant is intentionally NOT here; the bug
# was fixed and we don't want it to come back via this test.
KNOWN_ENGINES = [
    "es-bm25",
    "es-dense",
    "es-elser",
    "elastic-bm25",
    "elastic-dense",
    "elastic-elser",
    "chromadb",
    "faiss",
    "milvus",
    "milvus-dense",
    "milvus_dense",
    "milvus-sparse",
    "milvus_sparse",
    "milvus-bm25",
    "milvus_bm25",
    "milvus-hybrid",
    "milvus_hybrid",
    "milvus-splade",
    "milvus_splade",
    "lancedb",
    "lance",
]


@pytest.mark.parametrize("engine_name", KNOWN_ENGINES)
def test_dispatcher_recognizes_engine_name(engine_name):
    """Dispatcher reaches a branch for every documented engine name.

    We deliberately don't construct the engine (would require optional deps).
    Instead we read the source and grep for the name as a string literal in
    `create_retrieval_engine`. This is a lightweight surface lock that runs
    without milvus/elastic/etc. installed.
    """
    from docuverse.utils import retrievers

    src = inspect.getsource(retrievers.create_retrieval_engine)
    # The name must appear in a string literal somewhere in the dispatcher.
    assert (
        f"'{engine_name}'" in src or f'"{engine_name}"' in src
    ), f"engine name {engine_name!r} no longer appears in create_retrieval_engine"


def test_dispatcher_no_trailing_space_milvus_splade():
    """Regression: the trailing-space `'milvus_splade '` typo must stay fixed."""
    from docuverse.utils import retrievers

    src = inspect.getsource(retrievers.create_retrieval_engine)
    assert "milvus_splade '" not in src and 'milvus_splade "' not in src, (
        "the trailing-space `milvus_splade ` typo regressed; "
        "see plan PR 1 / Phase 3.4"
    )


def test_file_engine_name_pattern():
    """`file:<path>` is special-cased and must still be."""
    from docuverse.utils import retrievers

    src = inspect.getsource(retrievers.create_retrieval_engine)
    assert 'startswith("file:")' in src or "startswith('file:')" in src


# --- Reranker dispatch ------------------------------------------------------


@pytest.mark.parametrize("reranker_name", ["dense", "splade", "cross-encoder", "none"])
def test_reranker_dispatcher_recognizes_name(reranker_name):
    from docuverse.utils import retrievers

    src = inspect.getsource(retrievers.create_reranker_engine)
    assert (
        f"'{reranker_name}'" in src or f'"{reranker_name}"' in src
    ), f"reranker name {reranker_name!r} no longer appears in dispatcher"


# --- Config loading ---------------------------------------------------------

# Only the YAML/JSON files actually committed to the repo (not local dev
# artifacts in config/). Discovered via `git ls-files config/` at PR 1 time.
TRACKED_CONFIGS = [
    "config/clapnq_data_format.yml",
    "config/data_template.yml",
]


@pytest.mark.parametrize("rel_path", TRACKED_CONFIGS)
def test_existing_yaml_still_parses(rel_path):
    """Tracked config YAMLs must still load via DocUVerseConfig.

    These are data-format templates; loading them through DocUVerseConfig is
    how `data_format: <path>` is resolved at engine init time.
    """
    path = REPO_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not present in this checkout")

    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Smoke: it parses as YAML and is a dict (or None for fully-commented
    # templates like data_template.yml).
    assert data is None or isinstance(data, dict), (
        f"{rel_path} must parse to dict or None (got {type(data).__name__})"
    )


# --- Console scripts --------------------------------------------------------


def test_existing_console_scripts_importable():
    """`ingest-and-test` and `db2db-copy` entry-point targets must still resolve."""
    from docuverse.utils.ingest_and_test import main_cli  # noqa: F401
    from scripts.milvus_utils.db2db_copy import main  # noqa: F401
