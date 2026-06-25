# Config Reader: Jinja2 + Path-Aware Overrides — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom `{{var}}` resolver in `read_config_file` with Jinja2 (StrictUndefined) and route command-line overrides through dotted-path-aware merging while preserving every legacy helper in `docuverse/utils/__init__.py`.

**Architecture:** Internally, `read_config_file` becomes a four-step pipeline — load → apply overrides → build render context → render strings with Jinja2. New helpers (`_apply_overrides`, `_build_render_context`, `_render_with_jinja2`) live in `docuverse/utils/__init__.py`. Dotted-path overrides reuse `docuverse.presets.deep_merge_overrides`; leaf-key overrides reuse `_replace_leaf_keys`. Existing helpers (`_process_dictionary`, `_resolve_variable`, `_replace_leaf_keys`, `load_config_from_file`, `get_config_dir`) stay defined and importable.

**Tech Stack:** Python 3.10–3.14, jinja2≥3.1.6 (already in `pyproject.toml`), pytest.

**Spec:** [docs/superpowers/specs/2026-06-25-config-reader-jinja2-design.md](../specs/2026-06-25-config-reader-jinja2-design.md)

---

## Files

- **Create:** `tests/test_read_config_file.py` — all new tests for the redesigned `read_config_file` and its helpers.
- **Modify:** `docuverse/utils/__init__.py` — add `_apply_overrides`, `_build_render_context`, `_render_with_jinja2`, rewrite `read_config_file` body. Keep all current helpers and the `from jinja2 import Template, Undefined` import (legacy code paths still touch `NullUndefined`).
- **Read-only reference:** `docuverse/presets/__init__.py:deep_merge_overrides` — reused for dotted-path merging.
- **Read-only sweep:** `experiments/**/*.yaml`, `config/**/*.yaml` — to verify no config relies on parent-aware variable resolution.

---

## Task 1: Add `_apply_overrides` helper (tests first)

**Files:**
- Modify: `docuverse/utils/__init__.py` (add helper after `_replace_leaf_keys`, around line 260)
- Create: `tests/test_read_config_file.py`

- [ ] **Step 1: Write the failing test file with override tests**

Create `tests/test_read_config_file.py`:

```python
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
    read_config_file,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_read_config_file.py -x -v`
Expected: ImportError or "cannot import name '_apply_overrides'" — symbols don't exist yet.

- [ ] **Step 3: Add `_apply_overrides` to `docuverse/utils/__init__.py`**

Insert immediately AFTER the existing `_replace_leaf_keys` function (around line 260, before `read_config_file`):

```python
def _apply_overrides(config: dict[str, Any],
                     override_vals: dict[str, Any] | None) -> dict[str, Any]:
    """Return a new dict with override_vals applied to a deep copy of config.

    Override keys are interpreted in two modes:

    - **Dotted path** (key contains "."): target exactly one nested location,
      creating intermediate dicts as needed. Reuses
      ``docuverse.presets.deep_merge_overrides``.
    - **Leaf-key match** (no "."): every leaf in the tree whose key matches
      is replaced. Preserves the historical ``_replace_leaf_keys`` behavior.

    Both forms may be mixed in the same call. The base dict is never mutated.
    """
    if not override_vals:
        return copy.deepcopy(config)

    from docuverse.presets import deep_merge_overrides

    dotted = {k: v for k, v in override_vals.items() if "." in k}
    flat   = {k: v for k, v in override_vals.items() if "." not in k}

    # Dotted-path overrides via the existing presets helper (returns a new dict).
    result = deep_merge_overrides(config, dotted) if dotted else copy.deepcopy(config)

    # Leaf-key match for non-dotted keys, recursively replacing matching leaves.
    if flat:
        result = _replace_leaf_keys(result, flat)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_read_config_file.py -x -v -k apply_overrides`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add docuverse/utils/__init__.py tests/test_read_config_file.py
git commit -m "Add _apply_overrides helper (dotted-path + leaf-key match)"
```

---

## Task 2: Add `_build_render_context` helper (tests first)

The render context exposes top-level keys as variables AND lets templates dot into nested dicts (`{{ retriever.model_name }}`). Since Python dicts already support dotted access in Jinja2, the simplest implementation is to return the config dict itself as the context.

**Files:**
- Modify: `docuverse/utils/__init__.py`
- Modify: `tests/test_read_config_file.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_read_config_file.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_read_config_file.py -x -v -k build_render_context`
Expected: ImportError on `_build_render_context`.

- [ ] **Step 3: Add `_build_render_context`**

Insert in `docuverse/utils/__init__.py` immediately AFTER `_apply_overrides`:

```python
def _build_render_context(config: dict[str, Any]) -> dict[str, Any]:
    """Build a Jinja2 render context from a config dict.

    Returns a shallow copy so the caller can mutate the context (e.g., to
    add CLI-only variables) without affecting the source config. Nested
    dicts are not deep-copied: Jinja2 only reads them, and the renderer is
    in the same process.
    """
    return dict(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_read_config_file.py -x -v -k build_render_context`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add docuverse/utils/__init__.py tests/test_read_config_file.py
git commit -m "Add _build_render_context helper for Jinja2 rendering"
```

---

## Task 3: Add `_render_with_jinja2` helper (tests first)

The renderer walks the config recursively. Strings are passed through Jinja2; dicts and lists are recursed; everything else passes through. The walk repeats until the dict stops changing or `MAX_RESOLUTION_ITERATIONS` is reached.

**Files:**
- Modify: `docuverse/utils/__init__.py`
- Modify: `tests/test_read_config_file.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_read_config_file.py`:

```python
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
    assert "iteration" in str(excinfo.value).lower() or "resolve" in str(excinfo.value).lower()


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_read_config_file.py -x -v -k render`
Expected: ImportError on `_render_with_jinja2`.

- [ ] **Step 3: Add `_render_with_jinja2`**

Insert in `docuverse/utils/__init__.py` immediately AFTER `_build_render_context`. The function uses `jinja2.Environment` (not the existing `Template`/`NullUndefined`) so we get `StrictUndefined` semantics. Reuse the existing module-level `MAX_RESOLUTION_ITERATIONS` constant — define it next to the function if it isn't a module constant yet.

```python
def _render_with_jinja2(config: dict[str, Any],
                        ctx: dict[str, Any]) -> dict[str, Any]:
    """Render every string leaf with Jinja2 against ``ctx``.

    Walks ``config`` recursively. Each string passes through a Jinja2
    Environment with ``StrictUndefined``. Lists and dicts are recursed.
    Non-string scalars pass through unchanged.

    The walk repeats until the rendered tree stops changing or
    ``MAX_RESOLUTION_ITERATIONS`` is reached. Undefined variables and
    template syntax errors are wrapped in ``RuntimeError`` with the
    full dotted key path of the offending leaf.
    """
    from jinja2 import Environment, StrictUndefined
    from jinja2 import UndefinedError, TemplateSyntaxError

    env = Environment(undefined=StrictUndefined, autoescape=False)

    def render_node(node, path):
        if isinstance(node, dict):
            return {k: render_node(v, f"{path}.{k}" if path else k)
                    for k, v in node.items()}
        if isinstance(node, list):
            return [render_node(v, f"{path}[{i}]") for i, v in enumerate(node)]
        if isinstance(node, str) and "{{" in node or (isinstance(node, str) and "{%" in node):
            try:
                return env.from_string(node).render(**ctx)
            except UndefinedError as e:
                raise RuntimeError(
                    f"Undefined variable while rendering {path!r}: {e}"
                ) from e
            except TemplateSyntaxError as e:
                raise RuntimeError(
                    f"Template syntax error in {path!r}: {e}"
                ) from e
        return node

    MAX_ITER = 10
    current = config
    for _ in range(MAX_ITER):
        rendered = render_node(current, path="")
        if rendered == current:
            return rendered
        # Re-build the context against the partially-rendered config so
        # later passes see the resolved values.
        ctx = {**ctx, **{k: v for k, v in rendered.items() if k in ctx}}
        current = rendered
    raise RuntimeError(
        f"Could not resolve template variables after {MAX_ITER} iterations "
        f"(possible circular reference)"
    )
```

> **Note for the implementer:** the condition `"{{" in node or "{%" in node` short-circuits the Jinja2 call for plain strings. Watch out for operator precedence — wrap correctly: `if isinstance(node, str) and ("{{" in node or "{%" in node):`.

Replace the line in the snippet above accordingly:

```python
        if isinstance(node, str) and ("{{" in node or "{%" in node):
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_read_config_file.py -x -v -k render`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add docuverse/utils/__init__.py tests/test_read_config_file.py
git commit -m "Add _render_with_jinja2 helper (StrictUndefined, multi-pass)"
```

---

## Task 4: Rewire `read_config_file` to use the new pipeline

**Files:**
- Modify: `docuverse/utils/__init__.py:262-312` (the `read_config_file` body)
- Modify: `tests/test_read_config_file.py`

- [ ] **Step 1: Write the failing end-to-end tests**

Append to `tests/test_read_config_file.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_read_config_file.py -x -v -k read_config_file`
Expected: failures showing the current resolver swallowing typos and lacking dotted-path support.

- [ ] **Step 3: Replace `read_config_file` body**

Replace the existing `read_config_file` function body in `docuverse/utils/__init__.py` (around lines 262-312). The new body:

```python
def read_config_file(config_file, override_vals: dict[str, Any] = None) -> dict[str, Any]:
    """Read a YAML/JSON config file, apply overrides, render Jinja2 templates.

    Pipeline:
      1. Resolve config path (search ``get_config_dir`` if missing).
      2. Load YAML or JSON via ``load_config_from_file``.
      3. Apply ``override_vals`` via ``_apply_overrides`` (dotted-path
         + leaf-key match).
      4. Build a render context with ``_build_render_context``.
      5. Render every string leaf through Jinja2 (StrictUndefined,
         multi-pass) via ``_render_with_jinja2``.

    Args:
        config_file: Path to the YAML/JSON file. Relative paths are resolved
            via ``get_config_dir``.
        override_vals: Optional dict whose keys may be dotted paths
            (``"retriever.top_k"``) or bare leaf names (``"top_k"``).
            Dotted keys target one location; bare keys replace every
            matching leaf in the tree.

    Returns:
        The fully-rendered config as a dict (or whatever YAML produced).

    Raises:
        FileNotFoundError: if the file cannot be located.
        RuntimeError: on Jinja2 ``UndefinedError`` / ``TemplateSyntaxError``
            (with the field path in the message), or if templates fail to
            converge within 10 passes.
    """
    if not os.path.exists(config_file):
        config_file = os.path.join(get_config_dir(os.path.dirname(config_file)),
                                   os.path.basename(config_file))

    config = load_config_from_file(config_file)
    config = _apply_overrides(config, override_vals)
    if not isinstance(config, dict):
        return config  # Non-dict YAML (e.g., list at root) — nothing to render.
    ctx = _build_render_context(config)
    return _render_with_jinja2(config, ctx)
```

- [ ] **Step 4: Run the new end-to-end tests**

Run: `python -m pytest tests/test_read_config_file.py -x -v -k read_config_file`
Expected: 6 passed.

- [ ] **Step 5: Run the FULL test suite to catch regressions**

Run: `python -m pytest tests/ -x --timeout=60`
Expected: all green. If any test that was previously passing now fails, it likely depends on parent-aware variable resolution (see Task 5).

- [ ] **Step 6: Commit**

```bash
git add docuverse/utils/__init__.py tests/test_read_config_file.py
git commit -m "Rewire read_config_file to use Jinja2 + path-aware overrides"
```

---

## Task 5: YAML config sweep (verify parent-aware resolution wasn't load-bearing)

The legacy `_resolve_variable` walked up the parent path before falling back to global. Pure flat-context Jinja2 won't reproduce this. Sweep all YAML configs to make sure no real config relied on it.

**Files (read-only sweep):**
- `experiments/**/*.yaml`
- `config/**/*.yaml`

- [ ] **Step 1: Find every `{{...}}` reference in YAML configs**

Run:
```bash
grep -rEn "\\{\\{[^}]+\\}\\}" experiments/ config/ 2>/dev/null
```

For every match, note the variable name and the path of the file.

- [ ] **Step 2: For each unique variable name, check it resolves at top level**

For each variable name `NAME` you saw, scan the config file containing the reference and confirm:

- `NAME` exists as a top-level key, OR
- `NAME` is reachable via a dotted path that the user could write (`{{ retriever.NAME }}`), OR
- `NAME` would come from a CLI override (`--NAME ...`).

If a variable is defined ONLY inside a sub-dict and referenced from a sibling sub-dict, the legacy parent-walk would have found it but the new flat lookup will not. Flag those cases.

- [ ] **Step 3: Fix any flagged configs**

For each flagged YAML, hoist the variable to a top level OR rewrite the reference with the dotted path. Commit each fix as its own change with the YAML file as the only modification.

```bash
git add <fixed_yaml>
git commit -m "Hoist <var> to top level for flat Jinja2 context"
```

- [ ] **Step 4: If no flagged cases, record that in the plan log**

If the sweep finds nothing, no commit — just confirm in the implementation summary that the sweep was run and was clean.

---

## Task 6: Smoke test against the original failing command

**Files:** none modified — just verify behavior end-to-end.

- [ ] **Step 1: Run the kaz_hybrid command that started this work**

```bash
env CUDA_VISIBLE_DEVICES=1 python -m docuverse.utils.ingest_and_test \
    --config experiments/kaz_rag/kaz_hybrid_granite311m.yaml \
    --input_queries benchmark/Kaz-RAG-search-benchmark/voc_gap_queries.jsonl \
    --actions r
```

Expected: command starts up, loads config, parses templates, reaches the search step. (If the user has not re-ingested, the pipeline may bail at `.load()` — that's a state issue, not a config issue, and is unrelated to this work.)

- [ ] **Step 2: Verify a CLI override flows through**

Run the same command with an extra override, e.g.:

```bash
env CUDA_VISIBLE_DEVICES=1 python -m docuverse.utils.ingest_and_test \
    --config experiments/kaz_rag/kaz_hybrid_granite311m.yaml \
    --top_k 3 \
    --actions r
```

Verify in the startup log (search_engine.py prints config) that `top_k=3` made it into the retriever config.

- [ ] **Step 3: Verify a dotted override flows through**

Run:

```bash
env CUDA_VISIBLE_DEVICES=1 python -m docuverse.utils.ingest_and_test \
    --config experiments/kaz_rag/kaz_hybrid_granite311m.yaml \
    --retriever.top_k 7 \
    --actions r
```

Verify the retriever's top_k is 7 and that other top_k values (e.g., reranker.top_k if present) are NOT changed.

> **Note:** If `parse_args_to_dict` doesn't currently understand `--retriever.top_k` syntax (it lstrips dashes and uses the remainder verbatim, which IS fine — `'retriever.top_k'` as a key flows straight through), the override will arrive at `read_config_file` exactly as the dotted-path key it needs to be. If you see "unknown argument" from HfArgumentParser instead, that's a separate parsing issue — flag it; it's out of scope for this plan.

- [ ] **Step 4: Commit smoke-test results in the implementation summary**

No code commit; just record the smoke-test outcomes in the PR description / handoff message.

---

## Self-Review

**Spec coverage:**
- §Design / API surface → Task 4
- §Override semantics → Task 1 (`_apply_overrides`)
- §Jinja2 rendering details → Task 3 (`_render_with_jinja2`) — StrictUndefined, multi-pass, list-string rendering, error wrapping all covered
- §Render context → Task 2 (`_build_render_context`)
- §Compatibility → Task 5 (YAML sweep), Task 6 (smoke test)
- §Tests (cases 1–11) → Tasks 1–4 cover all 11 listed in the spec
- §File touches → Task list updates `__init__.py` and adds `tests/test_read_config_file.py`. `pyproject.toml` does NOT need editing — `jinja2>=3.1.6` is already declared.

**Placeholder scan:** No "TBD"/"TODO"/"add error handling" — every step has the actual code or command.

**Type/name consistency:** All three new helpers are named consistently across tests, implementation, and the `read_config_file` rewrite. `MAX_ITER = 10` matches the spec's `MAX_RESOLUTION_ITERATIONS`.

**Out-of-scope reminders:** literal-dot keys in YAML are not handled (spec). `read_yaml_config` in `yaml_config_reader.py` is not touched (spec: keep existing functions). The custom `_resolve_variable` / `_process_dictionary` remain importable but unused.