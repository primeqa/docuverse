# DocUVerse

Document retrieval and search experimentation library with unified interface to Elasticsearch, Milvus, ChromaDB, FAISS, lancedb.

## Environment

- IMPORTANT: Always use conda environment `ndocu`: `conda activate ndocu`
- Python 3.10-3.14
- 4 GPUs available; prefer GPU 1 (RTX 5090): set `CUDA_VISIBLE_DEVICES=1`
- Set `TOKENIZERS_PARALLELISM=false` when using multiprocessing with tokenizers

## Build & Install

```bash
pip install -e .
# Backend extras: pip install -e ".[milvus]", ".[elastic]", ".[chromadb]", ".[faiss]", ".[all]", ".[lancedb]"
```

Build wheel: `bash scripts/build_wheel.sh` (flags: `--clean`, `--verify`, `--test`, `--upload`)

Version is in the `VERSION` file (read dynamically by pyproject.toml) — update there, not in pyproject.toml.

## Tests

```bash
python -m pytest tests/
python -m pytest tests/test_search_data.py::TestDefaultProcessor::test_cleanup
```

Tests use both `unittest.TestCase` and `pytest` styles. Backend-dependent tests require mocks (no live services assumed).

## CLI Entry Point

```bash
python -m docuverse.utils.ingest_and_test --config path/to/config.yaml --actions "ire"
```

Action flags: `i` (ingest), `u` (update), `r` (retrieve), `e` (evaluate), `R` (rerank).

## Code Style

- Classes: `PascalCase`. Functions/methods: `snake_case`. Constants: `UPPER_SNAKE_CASE`
- Private methods: single leading underscore (`_create_retriever`)
- Type hints on all public method signatures
- Mixed type hint styles exist (`Union[X, Y]` and `X | Y`) — match the style of the file you're editing
- No automated formatter — use 4-space indentation, match surrounding code style
- Config dataclasses use `@dataclass` with `field(default=..., metadata={"help": "..."})` pattern (HuggingFace-style)
- YAML configs use `{{variable}}` templating, resolved by real Jinja2 (`StrictUndefined`)

## Gotchas

- IMPORTANT: Optional backend packages (pymilvus, elasticsearch, chromadb) use lazy imports. Never add top-level imports for these — wrap in `try/except` or import inside functions to avoid import-time failures.
- The `VERSION` file is the single source of truth for package version.
- Config `{{var}}` templating is real Jinja2 (`StrictUndefined`, multi-pass) via `_render_with_jinja2` in `docuverse/utils/__init__.py`. Undefined variables raise `RuntimeError`; unscoped names are auto-qualified to their nested path. The old `_process_params` still exists but is a dead no-op — no longer called by `read_config_file`.
- Elasticsearch credentials come from env vars: `ES_HOST`, `ES_USER`, `ES_PASSWORD`, `ES_API_KEY`, `ES_SSL_FINGERPRINT`.
- `DOCUVERSE_CONFIG_PATH` env var overrides the config directory location.
- Dataset format configs (field mappings for BEIR, ClapNQ, SAP, etc.) live in `config/`.

## Git Workflow

- Release branches: `v0.1.X` (current: `v0.1.0`), merged to `main`
- Commit messages: descriptive prose (no Conventional Commits format)
