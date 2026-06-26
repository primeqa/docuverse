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
    _scope_unscoped_variables,
    read_config_file,
    short_model,
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


# ---------- read_config_file end-to-end ----------


def _write_yaml(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return str(p)


def test_read_config_file_plain_yaml_unchanged(tmp_path):
    path = _write_yaml(tmp_path, "c.yaml", """
        a: 1
        b: hello
    """)
    assert read_config_file(path) == {"a": 1, "b": "hello"}


def test_read_config_file_resolves_template_via_self(tmp_path):
    path = _write_yaml(tmp_path, "c.yaml", """
        name: granite
        model: "{{ name }}-base"
    """)
    assert read_config_file(path) == {"name": "granite", "model": "granite-base"}


def test_read_config_file_nested_template_dot_access(tmp_path):
    path = _write_yaml(tmp_path, "c.yaml", """
        retriever:
            model_name: granite
        index: "{{ retriever.model_name }}_idx"
    """)
    out = read_config_file(path)
    assert out["index"] == "granite_idx"


def test_read_config_file_leaf_key_override_then_render(tmp_path):
    path = _write_yaml(tmp_path, "c.yaml", """
        name: from-file
        model: "{{ name }}-base"
    """)
    out = read_config_file(path, override_vals={"name": "from-cli"})
    assert out["model"] == "from-cli-base"


def test_read_config_file_dotted_override_targets_nested_only(tmp_path):
    path = _write_yaml(tmp_path, "c.yaml", """
        retriever:
            model_name: orig-r
        reranker:
            model_name: orig-rr
    """)
    out = read_config_file(path, override_vals={"retriever.model_name": "new-r"})
    assert out["retriever"]["model_name"] == "new-r"
    assert out["reranker"]["model_name"] == "orig-rr"


def test_read_config_file_undefined_variable_raises(tmp_path):
    path = _write_yaml(tmp_path, "c.yaml", """
        retriever:
            input: "{{ benchmar_dir }}/x"
    """)
    with pytest.raises(RuntimeError) as excinfo:
        read_config_file(path)
    assert "retriever.input" in str(excinfo.value)
    assert "benchmar_dir" in str(excinfo.value)


def test_read_config_file_non_dict_root_raises(tmp_path):
    """Root must be a mapping; lists/scalars at the top are rejected explicitly."""
    path = _write_yaml(tmp_path, "list_root.yaml", """
        - one
        - two
    """)
    with pytest.raises(RuntimeError) as excinfo:
        read_config_file(path)
    assert "mapping" in str(excinfo.value).lower() or "dict" in str(excinfo.value).lower()


def test_read_config_file_json_with_template(tmp_path):
    """JSON configs render through the same Jinja2 pipeline as YAML."""
    import json
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"name": "granite", "model": "{{ name }}-base"}))
    out = read_config_file(str(path))
    assert out == {"name": "granite", "model": "granite-base"}


def test_read_config_file_unscoped_var_auto_resolved(tmp_path):
    """An unscoped {{name}} that lives in exactly one nested scope is auto-rewritten."""
    path = _write_yaml(tmp_path, "c.yaml", """
        retriever:
            name: granite
            label: "{{ name }}"
    """)
    out = read_config_file(path)
    assert out["retriever"]["label"] == "granite"

    # Explicit dotted form continues to work:
    path2 = _write_yaml(tmp_path, "c2.yaml", """
        retriever:
            name: granite
            label: "{{ retriever.name }}"
    """)
    out2 = read_config_file(path2)
    assert out2["retriever"]["label"] == "granite"


def test_read_config_file_unscoped_var_cross_section(tmp_path):
    """An unscoped var defined in one section can be referenced from another section."""
    path = _write_yaml(tmp_path, "c.yaml", """
        retriever:
            model_name: granite
        index: "{{ model_name }}_idx"
    """)
    out = read_config_file(path)
    assert out["index"] == "granite_idx"


def test_read_config_file_ambiguous_var_raises(tmp_path):
    """An unscoped var that appears in multiple nested scopes raises RuntimeError."""
    path = _write_yaml(tmp_path, "c.yaml", """
        retriever:
            model_name: granite
        reranker:
            model_name: cross-enc
        index: "{{ model_name }}_idx"
    """)
    with pytest.raises(RuntimeError) as excinfo:
        read_config_file(path)
    msg = str(excinfo.value)
    assert "model_name" in msg
    assert "Ambiguous" in msg or "ambiguous" in msg


def test_scope_unscoped_variables_rewrites_nested_var():
    cfg = {"retriever": {"model_name": "granite"},
           "index": "{{ model_name }}_idx"}
    out = _scope_unscoped_variables(cfg)
    assert out["index"] == "{{ retriever.model_name }}_idx"
    assert out["retriever"]["model_name"] == "granite"  # untouched


def test_scope_unscoped_variables_leaves_top_level_var_alone():
    cfg = {"model_name": "granite", "index": "{{ model_name }}_idx"}
    out = _scope_unscoped_variables(cfg)
    assert out["index"] == "{{ model_name }}_idx"  # top-level: no rewrite


def test_scope_unscoped_variables_leaves_already_scoped_var_alone():
    cfg = {"retriever": {"model_name": "granite"},
           "index": "{{ retriever.model_name }}_idx"}
    out = _scope_unscoped_variables(cfg)
    assert out["index"] == "{{ retriever.model_name }}_idx"


def test_scope_unscoped_variables_ambiguous_raises():
    cfg = {"retriever": {"model_name": "a"},
           "reranker": {"model_name": "b"},
           "index": "{{ model_name }}"}
    with pytest.raises(RuntimeError, match="Ambiguous"):
        _scope_unscoped_variables(cfg)


def test_scope_unscoped_variables_var_with_filter():
    cfg = {"retriever": {"model_name": "ibm/granite-30m"},
           "index": "{{ model_name | short_model }}_idx"}
    out = _scope_unscoped_variables(cfg)
    assert out["index"] == "{{ retriever.model_name | short_model }}_idx"


def test_scope_unscoped_variables_deep_nesting():
    cfg = {"section": {"sub": {"val": "x"}},
           "label": "{{ val }}-label"}
    out = _scope_unscoped_variables(cfg)
    assert out["label"] == "{{ section.sub.val }}-label"


def test_read_config_file_empty_yaml_returns_empty_dict(tmp_path):
    """Empty / comment-only YAML files yield {} for back-compat with `or {}` callers."""
    path = tmp_path / "empty.yaml"
    path.write_text("# just a comment\n")
    assert read_config_file(str(path)) == {}


# ---------- short_model filter ----------


def test_short_model_basename_strips_org_prefix():
    assert short_model("ibm-granite/granite-embedding-30m-english") == \
        "granite-embedding-30m-english"
    assert short_model("sentence-transformers/all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"


def test_short_model_basename_handles_no_slash():
    assert short_model("granite-embedding-30m-english") == "granite-embedding-30m-english"


def test_short_model_none_or_empty_returns_unknown():
    assert short_model(None) == "unknown"
    assert short_model("") == "unknown"


def test_short_model_compact_granite():
    assert short_model("ibm-granite/granite-embedding-30m-english", "compact") == "granite30m"
    assert short_model("ibm-granite/granite-embedding-149m-english", "compact") == "granite149m"
    assert short_model("ibm-granite/granite-embedding-311m-multilingual-r2", "compact") == "granite311m"


def test_short_model_compact_other_families():
    assert short_model("sentence-transformers/all-MiniLM-L6-v2", "compact") == "MiniLM"
    assert short_model("intfloat/e5-base-v2", "compact") == "e5"
    assert short_model("BAAI/bge-large-en-v1.5", "compact") == "bge"


def test_short_model_compact_falls_back_to_basename_when_no_family_or_size():
    assert short_model("custom-org/weirdname", "compact") == "weirdname"


def test_short_model_slug_replaces_non_alphanum():
    assert short_model("ibm-granite/granite-embedding-30m-english", "slug") == \
        "granite_embedding_30m_english"
    assert short_model("BAAI/bge-large-en-v1.5", "slug") == "bge_large_en_v1_5"


def test_short_model_unknown_style_raises():
    with pytest.raises(ValueError) as excinfo:
        short_model("a/b", "weird")
    assert "weird" in str(excinfo.value)


def test_short_model_works_in_jinja2_template(tmp_path):
    """End-to-end: the filter is registered on the renderer and usable from YAML."""
    path = tmp_path / "c.yaml"
    path.write_text(
        "model_name: ibm-granite/granite-embedding-30m-english\n"
        "index_name: \"corpus-{{ model_name | short_model('compact') }}-idx\"\n"
    )
    out = read_config_file(str(path))
    assert out["index_name"] == "corpus-granite30m-idx"


def test_short_model_default_style_works_in_jinja2_template(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "model_name: ibm-granite/granite-embedding-30m-english\n"
        "label: \"{{ model_name | short_model }}\"\n"
    )
    out = read_config_file(str(path))
    assert out["label"] == "granite-embedding-30m-english"