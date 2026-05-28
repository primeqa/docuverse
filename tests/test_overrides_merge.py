"""Unit tests for the preset override deep-merge.

The merge is shared between ``SearchEngine.from_preset(**overrides)`` and the
CLI's ``--override key=value`` flag, so the rules need to be precise.
"""
from __future__ import annotations

from docuverse.presets import deep_merge_overrides


def test_flat_override_replaces_scalar():
    base = {"top_k": 10, "model_name": "a"}
    out = deep_merge_overrides(base, {"top_k": 5})
    assert out == {"top_k": 5, "model_name": "a"}


def test_dotted_key_explodes_into_nested_dict():
    base = {"retriever": {"top_k": 10, "other": "x"}}
    out = deep_merge_overrides(base, {"retriever.top_k": 5})
    assert out == {"retriever": {"top_k": 5, "other": "x"}}


def test_nested_dict_overrides_merge_recursively():
    base = {"a": {"b": {"c": 1, "d": 2}, "e": 3}}
    out = deep_merge_overrides(base, {"a": {"b": {"c": 99}}})
    assert out == {"a": {"b": {"c": 99, "d": 2}, "e": 3}}


def test_lists_replace_not_append():
    base = {"ranks": [1, 5, 10]}
    out = deep_merge_overrides(base, {"ranks": [3]})
    assert out == {"ranks": [3]}


def test_unknown_keys_pass_through():
    base = {"db_engine": "milvus-dense"}
    out = deep_merge_overrides(base, {"completely_new_key": "value"})
    assert out == {"db_engine": "milvus-dense", "completely_new_key": "value"}


def test_base_is_not_mutated():
    base = {"a": {"b": 1}}
    deep_merge_overrides(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}, "deep_merge_overrides must not mutate the base dict"


def test_none_overrides_returns_copy_of_base():
    base = {"a": 1}
    out = deep_merge_overrides(base, None)
    assert out == base
    assert out is not base


def test_empty_overrides_returns_copy_of_base():
    base = {"a": 1}
    out = deep_merge_overrides(base, {})
    assert out == base
    assert out is not base


def test_dotted_key_creates_missing_intermediate_dicts():
    base = {"existing": 1}
    out = deep_merge_overrides(base, {"new.deep.path": "value"})
    assert out == {"existing": 1, "new": {"deep": {"path": "value"}}}


def test_dotted_key_overwrites_non_dict_intermediate():
    """If intermediate is a scalar, the dotted-path override replaces it with a dict."""
    base = {"a": "scalar"}
    out = deep_merge_overrides(base, {"a.b": 1})
    assert out == {"a": {"b": 1}}


def test_mixed_dotted_and_nested_in_same_overrides():
    base = {"x": {"y": 0, "z": 0}}
    out = deep_merge_overrides(base, {"x.y": 1, "x": {"z": 2}})
    # Both forms exploded then merged; later wins on conflict in dict update order.
    assert out["x"]["y"] == 1
    assert out["x"]["z"] == 2
