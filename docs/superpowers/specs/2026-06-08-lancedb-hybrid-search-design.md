# LanceDB Hybrid Search — Design

Status: approved (pending user review)
Date: 2026-06-08
Owner: docuverse / lancedb engine

## Goal

Add hybrid retrieval to the LanceDB backend, configurable to combine any of:
dense embeddings (one or more models), BM25 via LanceDB's native FTS, and learned
sparse vectors (e.g. SPLADE). Configuration shape mirrors the existing
`MilvusHybridEngine` so that experiment YAML translates with a one-line edit
to `db_engine`.

## Non-goals

- Cross-backend hybrid (mixing `milvus_*` and `lancedb_*` sub-engines).
- A first-class sparse-vector index inside LanceDB (LanceDB has no native sparse
  type or scoring; we accept the cost of Python-side scoring).
- Reranker integration changes — the existing reranker pipeline is unchanged.

## Architecture

The current monolithic `LanceDBEngine` is refactored into a base class plus
sub-engines, mirroring the Milvus layout. A composer engine instantiates
sub-engines that all share one LanceDB table.

```
docuverse/engines/retrieval/lancedb/
  __init__.py
  lancedb.py             # LanceDBEngine base — connection, schema, ingestion plumbing
  lancedb_dense.py       # LanceDBDenseEngine — what current LanceDBEngine does
  lancedb_bm25.py        # LanceDBBM25Engine — LanceDB FTS index on `text`
  lancedb_sparse.py      # LanceDBSparseEngine — sparse struct column + Python scoring
  lancedb_hybrid.py      # LanceDBHybridEngine — composes sub-engines, merges results
  lancedb_engine.py      # back-compat alias re-exporting LanceDBDenseEngine as LanceDBEngine
```

Dispatch in `docuverse/utils/retrievers.py` is extended:

```python
elif name in ['lancedb', 'lance', 'lancedb-dense', 'lancedb_dense']:
    engine = LanceDBDenseEngine(retriever_config)
elif name in ['lancedb-bm25', 'lancedb_bm25']:
    engine = LanceDBBM25Engine(retriever_config)
elif name in ['lancedb-sparse', 'lancedb_sparse']:
    engine = LanceDBSparseEngine(retriever_config)
elif name in ['lancedb-hybrid', 'lancedb_hybrid']:
    engine = LanceDBHybridEngine(retriever_config)
```

`db_engine: lancedb` and `db_engine: lance` continue to resolve to the dense
engine. Existing experiments and tests are unaffected.

## Components

### `LanceDBEngine` (base)

Extracted from the current `lancedb_engine.py`. Owns:

- The `lancedb.connect(...)` client and `self.db`.
- The shared `self.table` handle and `_open_table()` auto-detect logic.
- Always-present columns: `id`, `text`, `title`, plus `extra_fields` from
  `data_template.extra_fields`.
- `has_index`, `delete_index`, `info`, ingestion batching loop, and the
  `_create_data` / `_insert_data` skeleton (the parts that do not depend on
  any specific encoder).

Subclasses contribute via three hooks:

- `extra_schema_fields() -> list[pa.field]` — columns this engine adds to the
  shared table.
- `encode_data(texts: list[str]) -> list` — values to write into this engine's
  column(s) for each document.
- `encode_query(text: str) -> Any` — what to pass to LanceDB at search time
  (vector, sparse struct, or query string).
- `build_indexes()` — indexes to create after ingestion (ANN, FTS, or no-op).
- `search(query) -> list[(id, score, payload)]` — engine-specific retrieval.

### `LanceDBDenseEngine(LanceDBEngine)`

Behaves exactly as the current `LanceDBEngine`. One dense vector column per
instance (default name `vector`, configurable via `embeddings_name`). Uses
`DenseEmbeddingFunction`. `build_indexes()` calls the existing
`_maybe_create_ann_index`. Search uses `table.search(vec, vector_column_name=...)`
and converts `_distance` to a score using the existing dot/cosine/L2 mapping.

### `LanceDBBM25Engine(LanceDBEngine)`

Adds no new column — relies on the shared `text` column.
`build_indexes()` calls `table.create_fts_index("text", replace=True, ...)`.
`encode_data` is a no-op. `encode_query` returns the raw query string.
Search runs `table.search(query_text, query_type="fts").limit(top_k).to_list()`
and exposes LanceDB's `_score` as `score`.

### `LanceDBSparseEngine(LanceDBEngine)`

LanceDB has no native sparse vector type, so each sparse model gets a struct
column `{indices: list<int32>, values: list<float32>}` (named via
`embeddings_name`). `encode_data` uses `SparseEmbeddingFunction` to produce
sparse vectors. `build_indexes()` is a no-op.

Search:

- When invoked inside a hybrid: takes the union of candidate ids produced by
  the other sub-engines and rescoring only those rows. The sparse column for
  those rows is fetched via filter (`id in (...)`) and a dot product is
  computed against the query sparse vector.
- When invoked standalone: full table scan with a logged warning. Standalone
  use is supported but not recommended for large corpora.

### `LanceDBHybridEngine(LanceDBEngine)`

Reads `config.hybrid` (same shape as `MilvusHybridEngine`). Instantiates one
sub-engine per entry in `hybrid.models`, all sharing `self.db` and `self.table`.

- Schema: union of every sub-engine's `extra_schema_fields()` plus the base
  fields. Built once at `create_index` time.
- Ingestion: single pass over the corpus, batched by `ingestion_batch_size`.
  For each batch, every sub-engine encodes its representation; one merged
  record per document is written via a single `table.add(records)` call.
- Index build: each sub-engine's `build_indexes()` is called on the shared
  table after ingestion completes.
- Search: per-sub-engine retrieval followed by combination (see Search & Combination).

The shared-table assumption is the load-bearing simplification: sub-engines
know how to encode, score, and index their column(s), but they do not own a
table.

## Config schema

Reuses Milvus's hybrid config shape so existing YAML translates by changing
`db_engine` and the sub-engine names.

```yaml
retriever:
    db_engine: lancedb-hybrid
    server: /path/to/lancedb_data        # local LanceDB directory
    index_name: "{{corpus}}-lancedb-hybrid-{{date}}"
    top_k: 100
    hybrid:
        shared_tokenizer: true
        model_name: /path/to/granite.30m
        short_model_name: granite30m
        title_handling: all
        combination: weighted             # or: rrf
        rrf_k: 60                         # optional, default 60
        models:
            granite-dense:
                weight: 0.7
                top_k: 100
                db_engine: lancedb_dense
                embeddings_name: granite_dense_vector
                index_params:             # optional ANN config
                    index_type: IVF_PQ
                    num_partitions: 256
                    num_sub_vectors: 16
            bm25:
                weight: 0.2
                top_k: 100
                db_engine: lancedb_bm25
                # no model_name needed — FTS works on `text` column
            splade-sparse:
                weight: 0.1
                top_k: 100
                db_engine: lancedb_sparse
                model_name: naver/splade-v3
                embeddings_name: splade_sparse
```

Validation rules (raised at `LanceDBHybridEngine.__init__`):

- `combination` must be `rrf` or `weighted`.
- For `combination: weighted`, every sub-model must have a numeric `weight`;
  weights are normalized to sum to 1.
- Each sub-model's `db_engine` must be one of `lancedb_dense`, `lancedb-dense`,
  `lancedb_bm25`, `lancedb-bm25`, `lancedb_sparse`, `lancedb-sparse`. Any
  `milvus_*` value raises (no cross-backend hybrid).
- Every sub-engine that owns a column must declare a unique `embeddings_name`;
  collisions raise at init. BM25 has no such requirement.

Standalone `lancedb-bm25` and `lancedb-sparse` are first-class engines —
the same classes, used directly without a hybrid wrapper.

## Search & combination

`LanceDBHybridEngine.search(query)`:

1. **Per-sub-engine retrieval.** Each sub-engine's `search()` runs
   independently with `top_k = max(self.config.top_k, sub.config.top_k)` and
   returns `[(id, score, payload), ...]`.
   - Dense: `table.search(vec, vector_column_name=...).limit(k).to_list()`,
     score derived from `_distance` per the existing metric mapping.
   - BM25: `table.search(text, query_type="fts").limit(k).to_list()`, score
     is LanceDB's `_score`.
   - Sparse: receives the candidate id pool from the other sub-engines (union),
     filter-fetches just those rows' sparse columns, computes dot product
     against the query sparse vector. Standalone fallback is a full scan with
     a warning.

2. **Native fast path.** If the configuration is exactly one dense + one
   BM25 sub-engine with `combination: rrf`, bypass the per-engine path and
   use LanceDB's built-in `query_type="hybrid"` with `RRFReranker()`. This
   is faster and avoids a Python merge round trip.

3. **Python merge.** Two strategies:
   - `combination: rrf`: standard RRF,
     `score(d) = Σ_i 1 / (rrf_k + rank_i(d))`, default `rrf_k = 60`,
     configurable via `hybrid.rrf_k`.
   - `combination: weighted`: min-max normalize each sub-engine's scores into
     `[0, 1]`, then `score(d) = Σ_i w_i · norm_score_i(d)`. Documents absent
     from a sub-engine's top-k contribute 0 from that sub-engine.

4. **Output.** Top `config.top_k` documents by combined score, each carrying
   `id`, `text`, `title`, every `extra_fields` value, and the combined `score`.
   `duplicate_removal` and `rouge_duplicate_threshold` are applied as in
   `MilvusHybridEngine.search`.

5. **Edge cases.**
   - Single sub-engine: skip merge, return that sub-engine's results.
   - Zero-norm or empty sparse query: skip that sub-engine for that query
     (matches Milvus tolerance).
   - Sub-engine returning 0 hits: contributes nothing; not an error.

## Ingestion

`LanceDBHybridEngine.ingest(corpus, update=False)`:

- Build merged schema once (base fields + every sub-engine's extras).
- Iterate `corpus` in `ingestion_batch_size` batches:
  - Construct base record (`id`, `text`, `title`, extras).
  - For each sub-engine, call `encode_data(texts_batch)` and copy each value
    into the corresponding row's column(s).
  - Skip rows where any sub-engine returns an unusable encoding (zero-norm
    sparse, empty text).
  - Single `table.add(records)` per batch.
- After all batches, call each sub-engine's `build_indexes()` on the shared
  table (ANN for dense, FTS for BM25, no-op for sparse).

## Error handling

- Top-of-file `try/except` for `lancedb` / `pyarrow` only (matches existing
  pattern). `SparseEmbeddingFunction` deps are imported lazily inside
  `LanceDBSparseEngine.init_model`.
- Config validation errors raise at `__init__` with messages naming the
  offending field.
- Schema collisions (duplicate `embeddings_name`) raise at init with the
  conflicting names.
- Unknown sub-engine `db_engine` values raise with the list of allowed names.
- Search-time exceptions in a single sub-engine are caught and logged; merge
  proceeds with the remaining sub-engines.
- Mixing `milvus_*` sub-engines into a `lancedb-hybrid` raises at init.

## Testing

`tests/test_lancedb_hybrid.py`, no live services. LanceDB runs locally on a
`tmp_path` directory, so tests use a real LanceDB instance against a small
synthetic corpus. `DenseEmbeddingFunction` and `SparseEmbeddingFunction` are
mocked to deterministic outputs so tests do not download models.

Coverage:

1. Engine dispatch — each new `db_engine` name routes to the correct class
   via `create_retrieval_engine`.
2. Schema construction — the hybrid table contains every sub-engine's columns.
3. Ingest-then-search round trip for:
   - dense-only,
   - BM25-only,
   - dense + BM25 (exercises the native fast path),
   - dense + BM25 + sparse with `combination: rrf`,
   - dense + BM25 + sparse with `combination: weighted`.
4. Combination math — RRF and weighted produce the expected ordering on a
   hand-constructed candidate set.
5. Edge cases — single sub-engine no-merge, zero-norm sparse query is skipped,
   missing weight in `weighted` config raises with a clear message,
   `milvus_*` sub-engine in `lancedb-hybrid` raises.
6. Back-compat — `db_engine: lancedb` still resolves to the dense engine;
   the existing `tests/test_engine_dispatch.py`, `tests/test_back_compat.py`,
   and `tests/test_presets.py` continue to pass unchanged.
