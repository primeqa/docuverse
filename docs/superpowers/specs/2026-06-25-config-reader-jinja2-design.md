# Config Reader: Jinja2 Templating + Path-Aware CLI Overrides

**Date:** 2026-06-25
**Author:** Hans Florian (with Claude)
**Status:** Approved (design)

## Background

DocUVerse loads experiment configurations from YAML/JSON files via
`read_config_file` in `docuverse/utils/__init__.py`. Today it has two
behaviors:

1. **Leaf-key overrides** (`override_vals` dict): replace any nested leaf
   whose key matches an entry in `override_vals`, regardless of where it
   sits in the tree.
2. **Custom `{{var}}` resolver**: walk every string leaf, find `{{name}}`
   patterns, and resolve them by recursively searching the loaded config
   dict (preferring keys near the current parent path, falling back to
   the global root).

The custom resolver is limited:
- No filters (`{{ name | upper }}`), no conditionals, no loops.
- Parent-aware lookup is undocumented and surprising.
- Strings inside lists are not rendered.

The override mechanism is also blunt: with hybrid-engine configs that
have multiple sub-models, a CLI `--model_name foo` mutates every
`model_name` in the tree, even ones meant to differ.

## Goal

Replace the custom resolver with **Jinja2** for full templating power,
add **dotted-path** override targeting (with leaf-key match retained for
back-compat), keep the public API of `read_config_file` and all existing
helper functions in `docuverse/utils/__init__.py`.

## Non-goals

- Whole-file Jinja2 pre-pass over raw YAML text (no `{% include %}` over
  files). Strings-only rendering is enough for current use cases.
- Adding a new CLI parser. Existing `parse_args_to_dict` and
  `HfArgumentParser` flow stay as-is. Only the dict that flows into
  `read_config_file` is reinterpreted.

## Design

### API surface

`read_config_file(config_file, override_vals=None)` keeps its signature
and call sites. New internal flow:

```python
def read_config_file(config_file, override_vals=None):
    raw = load_config_from_file(config_file)            # unchanged
    raw = _apply_overrides(raw, override_vals or {})    # NEW
    context = _build_render_context(raw)                # NEW
    return _render_with_jinja2(raw, context)            # NEW
```

The legacy helpers `_process_dictionary`, `_resolve_variable`,
`_replace_leaf_keys`, `load_config_from_file`, and `get_config_dir`
remain defined and importable. They are no longer called from
`read_config_file`.

### Override semantics (`_apply_overrides`)

For each `(key, value)` in `override_vals`:

- **Key contains `.`** → treat as a dotted path. Walk the config dict
  step-by-step, creating intermediate dicts when missing, and assign
  `value` at the leaf. Targets exactly one location.
- **Key has no `.`** → leaf-key match. Recursively replace every leaf
  whose key matches. Identical to today's `_replace_leaf_keys`.

A literal-dot key in YAML (e.g., `"foo.bar": 1` at the top level) is
unlikely in this codebase; if it ever appears, the dotted-path
interpretation wins. We will note this in the docstring; if it causes
real friction later, add an escape syntax (`\.`) — out of scope here.

Type coercion of CLI values stays in `parse_args_to_dict`. The override
dict is consumed as-is.

### Jinja2 rendering

`_render_with_jinja2(config, context)` traverses `config` recursively:

- **String leaf** → pass through `Environment(undefined=StrictUndefined,
  autoescape=False).from_string(value).render(**context)`. The result
  may itself contain `{{...}}` (templates that reference other
  templated values), so the traversal repeats up to
  `MAX_RESOLUTION_ITERATIONS = 10` until the dict stops changing.
  `UndefinedError` and `TemplateSyntaxError` are re-raised with the
  full key path (`retriever.input_passages`) so the user can find the
  offending field.
- **Dict** → recurse into values.
- **List** → render strings inside lists too (fixes a bug in the
  current code where list-string templates silently pass through).
- **Other scalar** → unchanged.

`StrictUndefined` is chosen so that `{{benchmar_dir}}` (typo) raises
loudly instead of silently rendering as empty string.

### Render context (`_build_render_context`)

The context exposes both top-level keys and dotted paths:

```python
{
    'benchmark_dir': '/data/...',
    'db_engine': 'milvus-hybrid',
    'retriever': {'model_name': '...', 'top_k': 10, ...},
    'hybrid':    {'shared_tokenizer': True, ...},
    ...
}
```

So both `{{ db_engine }}` and `{{ retriever.model_name }}` work in
Jinja2. CLI overrides are already merged in step 2, so there is no
separate "CLI extras" channel — the rendered dict IS the context.

### Compatibility

- Every existing `{{var}}` reference in `experiments/**/*.yaml` and
  `config/**/*.yaml` is valid Jinja2 syntax and continues to work,
  provided the variable is reachable as a top-level key or dotted path
  in the config (or as a CLI override).
- The current resolver's *parent-aware* lookup (a variable in
  `retriever.foo` resolves from `retriever.bar` first, then global) is
  dropped in favor of flat global lookup. The implementation phase will
  grep the YAML corpus and surface any config that depends on
  parent-scope resolution; if found, decide between flat-fixing the
  config or implementing per-subtree contexts.
- `jinja2` is added to base install requirements (it is already a
  transitive dependency of HuggingFace stack, but make it explicit).

### Errors

- `UndefinedError` → re-raise as `RuntimeError` with the full dotted key
  path of the offending leaf and the missing variable name.
- `TemplateSyntaxError` → re-raise as `RuntimeError` with the key path
  and the original Jinja2 error message.
- Iterations cap unchanged: 10 passes; raises `RuntimeError` if values
  are still changing after 10.

## Tests

New file `tests/test_config_reader.py`. Cases:

1. Plain YAML, no templates, no overrides → returned dict equals input.
2. `{{var}}` resolved from same file (top-level key).
3. Nested ref: `index_name: "{{retriever.model_name}}_idx"`.
4. Multi-pass: `a: "{{b}}"`, `b: "{{c}}"`, `c: hello` → `a == "hello"`.
5. CLI override (leaf-key) replaces all matching leaves, then template
   renders against the new value.
6. CLI override (dotted-path) targets one nested key; siblings with the
   same name elsewhere are untouched.
7. Jinja2 filter: `{{ name | upper }}`.
8. Jinja2 conditional: `{% if env == 'prod' %}A{% else %}B{% endif %}`.
9. `StrictUndefined`: typo `{{benchmar_dir}}` raises `RuntimeError`
   naming the field path and the missing variable.
10. Circular reference (`a: {{b}}`, `b: {{a}}`) raises after
    `MAX_RESOLUTION_ITERATIONS`.
11. String inside list is rendered: `paths: ["{{base}}/x", "{{base}}/y"]`.

## Out of scope / open questions

- **Literal-dot keys** in YAML: not handled; flagged in docstring.
- **Parent-aware resolution** dropped: implementation phase will sweep
  the YAML corpus and confirm no config depends on it.
- **Whole-file pre-pass**: explicitly rejected for this iteration.

## File touches

- `docuverse/utils/__init__.py` — add `_apply_overrides`,
  `_build_render_context`, `_render_with_jinja2`; rewrite
  `read_config_file` body to use them. Keep all existing helpers.
- `pyproject.toml` — declare `jinja2>=3` in base dependencies if not
  already present transitively-only.
- `tests/test_config_reader.py` — new file with the cases above.