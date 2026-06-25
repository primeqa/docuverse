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
    _render_with_jinja2,
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


# ---------- _render_with_jinja2 ----------


def test_render_passes_through_when_no_templates():
    cfg = {"a": 1, "b": "hello", "c": [1, 2], "d": {"e": "x"}}
    out = _render_with_jinja2(cfg, ctx={})
    assert out == cfg


def test_render_resolves_simple_var():
    cfg = {"name": "granite", "model": "{{ name }}-base"}
    ctx = _build_render_context(cfg)
    out = _render_with_jinja2(cfg, ctx=ctx)
    assert out["model"] == "granite-base"


def test_render_resolves_nested_var_via_dot():
    cfg = {"retriever": {"model_name": "granite"},
           "index": "{{ retriever.model_name }}_idx"}
    ctx = _build_render_context(cfg)
    out = _render_with_jinja2(cfg, ctx=ctx)
    assert out["index"] == "granite_idx"


def test_render_multipass_chains_resolution():
    cfg = {"a": "{{ b }}", "b": "{{ c }}", "c": "hello"}
    ctx = _build_render_context(cfg)
    out = _render_with_jinja2(cfg, ctx=ctx)
    assert out["a"] == "hello"
    assert out["b"] == "hello"
    assert out["c"] == "hello"


def test_render_jinja2_filter():
    cfg = {"name": "granite", "upper": "{{ name | upper }}"}
    ctx = _build_render_context(cfg)
    out = _render_with_jinja2(cfg, ctx=ctx)
    assert out["upper"] == "GRANITE"


def test_render_jinja2_conditional():
    cfg = {"env": "prod", "host": "{% if env == 'prod' %}prod.example.com{% else %}dev.example.com{% endif %}"}
    ctx = _build_render_context(cfg)
    out = _render_with_jinja2(cfg, ctx=ctx)
    assert out["host"] == "prod.example.com"


def test_render_strings_inside_lists():
    cfg = {"base": "/data", "paths": ["{{ base }}/x", "{{ base }}/y", 42]}
    ctx = _build_render_context(cfg)
    out = _render_with_jinja2(cfg, ctx=ctx)
    assert out["paths"] == ["/data/x", "/data/y", 42]


def test_render_strict_undefined_raises_with_field_path():
    cfg = {"retriever": {"input": "{{ benchmar_dir }}/file"}}  # typo
    ctx = _build_render_context(cfg)
    with pytest.raises(RuntimeError) as excinfo:
        _render_with_jinja2(cfg, ctx=ctx)
    msg = str(excinfo.value)
    assert "retriever.input" in msg, "error must name the offending field path"
    assert "benchmar_dir" in msg, "error must name the missing variable"


def test_render_circular_reference_raises_after_max_iterations():
    cfg = {"a": "{{ b }}", "b": "{{ a }}"}
    ctx = _build_render_context(cfg)
    with pytest.raises(RuntimeError) as excinfo:
        _render_with_jinja2(cfg, ctx=ctx)
    msg = str(excinfo.value)
    assert "iteration" in msg.lower() or "resolve" in msg.lower()
    # The error should name at least one of the offending paths.
    assert "a" in msg and "b" in msg, f"error must name the offending key(s): {msg}"


def test_render_template_syntax_error_names_field_path():
    cfg = {"a": {"b": "{{ unclosed"}}
    ctx = _build_render_context(cfg)
    with pytest.raises(RuntimeError) as excinfo:
        _render_with_jinja2(cfg, ctx=ctx)
    assert "a.b" in str(excinfo.value)


def test_render_does_not_mutate_input():
    cfg = {"a": "{{ b }}", "b": "x"}
    ctx = _build_render_context(cfg)
    _render_with_jinja2(cfg, ctx=ctx)
    assert cfg == {"a": "{{ b }}", "b": "x"}