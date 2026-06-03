"""Verify every shipped preset YAML is well-formed and round-trips through DocUVerseConfig.

These tests intentionally do NOT instantiate engines (would require optional
deps) — they only verify that:
  1. Every recipe parses as YAML.
  2. Every recipe declares a known ``db_engine`` value.
  3. ``DocUVerseConfig`` accepts the recipe dict without raising.
"""
from __future__ import annotations

import pytest

from docuverse.presets import list_presets, load_preset

# Mirrors the choices in `RetrievalArguments.db_engine`. Kept inline rather
# than imported to make the test independent of refactors to that schema.
KNOWN_DB_ENGINES = {
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
    "milvus-sparse",
    "milvus-bm25",
    "milvus-hybrid",
    "milvus-splade",
    "lancedb",
    "lance",
}


def test_eleven_presets_are_shipped():
    """Sanity: the recipe set didn't accidentally shrink."""
    names = list_presets()
    assert len(names) == 11, f"expected 11 recipes, got {len(names)}: {names}"


@pytest.mark.parametrize("name", list_presets())
def test_preset_loads_to_dict_with_db_engine(name):
    recipe = load_preset(name)
    assert isinstance(recipe, dict)
    assert "db_engine" in recipe, f"recipe {name!r} missing required 'db_engine' field"
    assert recipe["db_engine"] in KNOWN_DB_ENGINES, (
        f"recipe {name!r} declares unknown db_engine={recipe['db_engine']!r}"
    )


@pytest.mark.parametrize("name", list_presets())
def test_preset_builds_docuverse_config(name):
    """Recipes must be acceptable input to DocUVerseConfig (the path used by from_preset)."""
    from docuverse.engines.search_engine_config_params import DocUVerseConfig

    recipe = load_preset(name)
    cfg = DocUVerseConfig(recipe)
    assert cfg.retriever_config is not None
    assert cfg.retriever_config.db_engine == recipe["db_engine"]


def test_load_preset_unknown_name_raises_keyerror_with_listing():
    with pytest.raises(KeyError) as excinfo:
        load_preset("definitely-not-a-real-preset")
    msg = str(excinfo.value)
    assert "milvus-dense" in msg, "KeyError should list available presets"
