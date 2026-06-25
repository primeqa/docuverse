"""Tests for the redesigned read_config_file: Jinja2 rendering + path-aware overrides.

Spec: docs/superpowers/specs/2026-06-25-config-reader-jinja2-design.md
"""
from __future__ import annotations

import os
import textwrap

import pytest

from docuverse.utils import (
    _apply_overrides,
    _build_render_context,
)


# ---------- _apply_overrides ----------


def test_apply_overrides_no_overrides_returns_copy():
    base = {"a": 1, "b": {"c": 2}}
    out = _apply_overrides(base, None)
    assert out == base
    assert out is not base


def test_apply_overrides_empty_dict_returns_copy():
    base = {"a": 1}
    out = _apply_overrides(base, {})
    assert out == base
    assert out is not base


def test_apply_overrides_does_not_mutate_base():
    base = {"a": {"b": 1}}
    _apply_overrides(base, {"a.b": 2, "x": 3})
    assert base == {"a": {"b": 1}}, "base must not be mutated"


def test_apply_overrides_dotted_path_targets_one_nested_key():
    base = {"retriever": {"model_name": "old", "top_k": 10},
            "reranker":  {"model_name": "old"}}
    out = _apply_overrides(base, {"retriever.model_name": "new"})
    assert out == {"retriever": {"model_name": "new", "top_k": 10},
                   "reranker":  {"model_name": "old"}}


def test_apply_overrides_leaf_key_match_replaces_every_matching_leaf():
    base = {"retriever": {"model_name": "a"},
            "reranker":  {"model_name": "b"},
            "model_name": "c"}
    out = _apply_overrides(base, {"model_name": "REPLACED"})
    assert out["retriever"]["model_name"] == "REPLACED"
    assert out["reranker"]["model_name"] == "REPLACED"
    assert out["model_name"] == "REPLACED"


def test_apply_overrides_dotted_creates_intermediate_dicts():
    base = {"existing": 1}
    out = _apply_overrides(base, {"new.deep.path": "value"})
    assert out == {"existing": 1, "new": {"deep": {"path": "value"}}}


def test_apply_overrides_dotted_and_leaf_key_compose():
    base = {"retriever": {"model_name": "old", "top_k": 10},
            "reranker":  {"model_name": "old", "top_k": 20}}
    out = _apply_overrides(base, {"retriever.model_name": "new",
                                  "top_k": 5})
    # Dotted hits the one location; leaf-key replaces every matching leaf.
    assert out["retriever"]["model_name"] == "new"
    assert out["retriever"]["top_k"] == 5
    assert out["reranker"]["top_k"] == 5
    assert out["reranker"]["model_name"] == "old"


def test_apply_overrides_non_scalar_leaf_via_dotted_path():
    """Dotted-path overrides may target dict/list values, not just scalars."""
    base = {"retriever": {"params": {"a": 1}}}
    out = _apply_overrides(base, {"retriever.params": {"b": 2}})
    assert out == {"retriever": {"params": {"b": 2}}}


def test_apply_overrides_dotted_overwrites_non_dict_intermediate():
    """When an intermediate segment is not a dict, the dotted path replaces it
    with a fresh dict and assigns the leaf. Matches deep_merge_overrides."""
    base = {"retriever": "scalar_value"}
    out = _apply_overrides(base, {"retriever.model_name": "x"})
    assert out == {"retriever": {"model_name": "x"}}


# ---------- _build_render_context ----------


def test_build_render_context_exposes_top_level_keys():
    ctx = _build_render_context({"a": 1, "b": "hello"})
    assert ctx["a"] == 1
    assert ctx["b"] == "hello"


def test_build_render_context_exposes_nested_dicts_for_dot_access():
    """Jinja2 supports `{{ retriever.model_name }}` if context has a dict value."""
    cfg = {"retriever": {"model_name": "granite", "top_k": 10}}
    ctx = _build_render_context(cfg)
    # Access pattern Jinja2 will use:
    assert ctx["retriever"]["model_name"] == "granite"


def test_build_render_context_does_not_mutate_input():
    cfg = {"a": {"b": 1}}
    ctx = _build_render_context(cfg)
    ctx["new_key"] = "x"
    assert "new_key" not in cfg
    # Pin the shallow-copy contract: nested dicts are shared by design.
    assert ctx["a"] is cfg["a"]