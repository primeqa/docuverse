# DocUVerse Architecture Reference

Comprehensive guide to the codebase structure, class hierarchies, and design patterns.

## Package Layout

```
docuverse/
    __init__.py                           # Public API exports
    engines/
        __init__.py                       # Re-exports core classes
        search_engine.py                  # Top-level orchestrator
        search_engine_config_params.py    # All config dataclasses
        search_corpus.py                  # Document collection
        search_queries.py                 # Query collection + relevance
        search_result.py                  # Search result container
        search_data.py                    # Data loading/preprocessing
        data_template.py                  # DataTemplate + defaults
        sparse_config.py                  # SparseConfig dataclass

        preprocessors/
            base_preprocessor.py
            list_preprocessor.py
            model_runner_preprocessor.py

        retrieval/
            __init__.py
            retrieval_engine.py           # Abstract base for all backends
            retrieval_servers.py          # Server config management
            search_filter.py              # Field-based filtering
            elastic/                      # Elasticsearch backends
            milvus/                       # Milvus backends
            chromadb/                     # ChromaDB backend
            faiss/                        # FAISS backend
            lancedb/                      # LanceDB backend
            file/                         # File-based backend
            primeqa/                      # PrimeQA (ColBERT, DPR)

        reranking/
            reranker.py                   # Abstract base Reranker
            bi_encoder_reranker.py        # BiEncoder reranker base
            dense_reranker.py             # Dense bi-encoder reranker
            splade_reranker.py            # SPLADE bi-encoder reranker
            cross_encoder_reranker.py     # Cross-encoder reranker

    utils/
        __init__.py                       # get_param, read_config_file, etc.
        embeddings/
            embedding_function.py         # EmbeddingFunction base class
            dense_embedding_function.py   # SentenceTransformer wrapper
            sparse_embedding_function.py  # SPLADE wrapper
            splade3_embedding_function.py # pymilvus SPLADE wrapper
            bm25_embedding_function.py    # BM25 wrapper
            ollama_embedding_function.py  # Ollama wrapper
        text_tiler.py                     # Document chunking
        evaluator.py                      # EvaluationEngine (MRR, NDCG, etc.)
        evaluation_output.py              # EvaluationOutput container
        ingest_and_test.py                # CLI entry point (main_cli)
        retrievers.py                     # Factory functions
        timer.py                          # Timing utilities
        yaml_config_reader.py             # YAML parsing
        jsonl_utils.py                    # JSONL helpers

scripts/
    matryoshka/                           # Matryoshka training (see matryoshka_training.md)
    biohash/                              # BioHash text hashing
    milvus_utils/                         # Milvus admin scripts
    speed_test/                           # Benchmarking
    compute_embeddings_from_jsonl.py      # Standalone embedding computation
    export_to_openvino_onnx.py            # ONNX/OpenVINO export
    granite_embedding_model_converter.py  # Granite ONNX conversion
    granite_to_openvino_int8.py           # INT8 quantization
    noreen_statistical_testing.py         # Significance tests
    ...
```

## Class Hierarchies

### Retrieval Engines

All retrieval backends extend `RetrievalEngine`, which provides the contract for
indexing documents and searching queries.

```
RetrievalEngine (engines/retrieval/retrieval_engine.py)
    Abstract base. Key methods:
      search(query, **kwargs) -> SearchResult
      ingest(corpus, **kwargs)
      create_index(index_name, **kwargs)
      delete_index(index_name, **kwargs)
      has_index(index_name) -> bool
      encode_data(texts, batch_size, **kwargs)
      load_model_config(config_params)
      init_client()

    |-- ElasticEngine (retrieval/elastic/elastic.py)
    |     |-- ElasticBM25Engine (elastic_bm25.py)
    |     |-- ElasticDenseEngine (elastic_dense.py) + DenseEmbeddingFunction
    |     |-- ElasticElserEngine (elastic_elser.py)
    |
    |-- MilvusEngine (retrieval/milvus/milvus.py)
    |     |-- MilvusDenseEngine (milvus_dense.py) + DenseEmbeddingFunction
    |     |-- MilvusSparseEngine (milvus_sparse.py) + SparseEmbeddingFunction
    |     |-- MilvusBM25Engine (milvus_bm25.py)
    |     |-- MilvusSpladeEngine (milvus_splade.py) + SpladeEmbeddingFunction
    |     |-- MilvusHybridEngine (milvus_hybrid.py) (multiple sub-engines)
    |
    |-- ChromaDBEngine (retrieval/chromadb/chromadb_engine.py)
    |-- FAISSEngine (retrieval/faiss/faiss_engine.py) + DenseEmbeddingFunction
    |-- LanceDBEngine (retrieval/lancedb/lancedb_engine.py) + DenseEmbeddingFunction
    |-- FileEngine (retrieval/file/file_engine.py)
```

**Backend selection** is driven by the `db_engine` config parameter. The factory
function `create_retrieval_engine(config)` in `utils/retrievers.py` maps strings
to classes:

| `db_engine` value | Class |
|---|---|
| `es-bm25` | `ElasticBM25Engine` |
| `es-dense` | `ElasticDenseEngine` |
| `es-elser` | `ElasticElserEngine` |
| `milvus-dense` | `MilvusDenseEngine` |
| `milvus-sparse` | `MilvusSparseEngine` |
| `milvus-bm25` | `MilvusBM25Engine` |
| `milvus-hybrid` | `MilvusHybridEngine` |
| `milvus-splade` | `MilvusSpladeEngine` |
| `chromadb` | `ChromaDBEngine` |
| `faiss` | `FAISSEngine` |
| `lancedb` | `LanceDBEngine` |

### Rerankers

```
Reranker (engines/reranking/reranker.py)
    |-- BiEncoderReranker (bi_encoder_reranker.py)
    |     |-- DenseReranker (dense_reranker.py)     uses DenseEmbeddingFunction
    |     |-- SpladeReranker (splade_reranker.py)   uses SparseEmbeddingFunction
    |-- CrossEncoderReranker (cross_encoder_reranker.py)  uses CrossEncoder model
```

### Embedding Functions

```
EmbeddingFunction (utils/embeddings/embedding_function.py)
    Base API:
      __call__(texts, **kwargs) -> encodes texts
      encode(texts, _batch_size, show_progress_bar, prompt_name, **kwargs)
      encode_query(texts, prompt_name, **kwargs)
      create_model(model_or_directory_name, device, **kwargs)
      tokenizer -> HuggingFace tokenizer
      vocab_size -> int
      device -> torch.device

    |-- DenseEmbeddingFunction (dense_embedding_function.py)
    |     Wraps sentence_transformers.SentenceTransformer
    |     Multi-GPU via encode_multi_process pool
    |     OOM fallback: halves batch size, then falls back to CPU
    |     Normalizes embeddings by default
    |
    |-- SparseEmbeddingFunction (sparse_embedding_function.py)
    |     Wraps SparseSentenceTransformer (custom SPLADE impl)
    |     Uses AutoModelForMaskedLM + AutoTokenizer
    |     Returns scipy csr_matrix sparse vectors
    |
    |-- SpladeEmbeddingFunction (splade3_embedding_function.py)
    |     Wraps pymilvus.model.sparse.SpladeEmbeddingFunction
    |
    |-- BM25EmbeddingFunction (bm25_embedding_function.py)
    |     Wraps pymilvus BM25 with default analyzer
    |
    |-- OllamaEmbeddingFunction (ollama_embedding_function.py)
```

**Usage pattern** in engines:
```python
self.model = DenseEmbeddingFunction(
    model_or_directory_name,
    **self.config.__dict__
)
embeddings = self.model.encode(texts, _batch_size=128)
```

### SearchEngine (Orchestrator)

`SearchEngine` (`engines/search_engine.py`) ties everything together:

```python
SearchEngine
    owns: retriever (RetrievalEngine), reranker (BiEncoderReranker), config (DocUVerseConfig)

    create()           -> instantiates retriever + reranker via factory
    read_data()        -> SearchData.read_data() -> SearchCorpus
    read_questions()   -> SearchQueries
    ingest(corpus)     -> delegates to retriever.ingest()
    search(queries)    -> retriever.search() [+ reranker] -> List[SearchResult]
    compute_score()    -> EvaluationEngine
    write_output()     -> JSON/JSONL
    read_output()      -> from cache
```

## Configuration System

### Config Dataclasses

All configuration is in `engines/search_engine_config_params.py`:

```
GenericArguments             # Base with get(), __getitem__(), is_default()
    |-- RetrievalArguments   # ~40 fields: model_name, db_engine, index_name,
    |                        #   max_doc_length, stride, top_k, etc.
    |-- RerankerArguments    # reranker_model, reranker_engine, batch_size, etc.
    |-- EvaluationArguments  # ranks, eval_measure, compute_rouge, etc.
    |-- EngineArguments      # output_file, actions, cache_dir, skip, etc.

DocUVerseConfig              # Top-level config combining all sub-configs
    retriever_config: RetrievalArguments
    reranker_config: RerankerArguments
    eval_config: EvaluationArguments
    run_config: EngineArguments
```

`DocUVerseConfig` also flattens all sub-config fields onto itself for convenience
(e.g., `config.model_name` works alongside `config.retriever_config.model_name`).

**CLI parsing** uses HuggingFace `HfArgumentParser` in `utils/ingest_and_test.py`.

### YAML Variable Templating

Config files support `{{variable}}` references, resolved recursively up to 10
passes by `read_config_file()` in `utils/__init__.py`:

```yaml
retriever:
    project_dir: data/clapnq_small
    input_passages: "{{project_dir}}/passages.tsv"
    model_name: "ibm-granite/granite-embedding-30m-english"
    index_name: "{{dataset}}-{{model_name}}-{{max_doc_length}}"
```

## Key Design Patterns

### 1. Factory Pattern

Engine and reranker creation uses factories in `utils/retrievers.py`:
```python
create_retrieval_engine(retriever_config)   # db_engine string -> class
create_reranker_engine(reranker_config)     # reranker_engine string -> class
```

### 2. Embedding Function Composition

Every dense engine holds `self.model` (an `EmbeddingFunction` subclass). The engine
delegates encoding via `self.model.encode()`. This decouples the embedding model
from the storage/search backend.

### 3. Caching

`SearchEngine` caches retrieval and reranking results as `.pkl.bz2` files.
`open_stream()` in `utils/__init__.py` transparently handles `.bz2`, `.gz`, `.xz`
compression.

### 4. Parallel Processing

`parallel_process()` in `utils/__init__.py` uses `multiprocessing` with a fork
context. Module-level globals enable pickling of worker functions. Used for
ingestion preprocessing and parallel tiling.

### 5. TextTiler for Chunking

`TextTiler` (`utils/text_tiler.py`) splits long documents with configurable:
- Max size (token or character count)
- Stride (overlap between tiles)
- Sentence-boundary alignment (via `pyizumo`)

Passed from `SearchEngine.create_tiler()` down to `SearchData.read_data()`.

### 6. `get_param()` Utility

Used pervasively for safe attribute/key access:
```python
get_param(obj, "key")                  # dict or object
get_param(obj, "index|index_name")     # fallback key with |
get_param(obj, "a.b.c")               # dotted nested keys
```

## Data Flow

```
YAML Config
    |
    v
DocUVerseConfig  ->  SearchEngine.create()
                        |
              +---------+---------+
              |                   |
    RetrievalEngine          Reranker (optional)
    (e.g. MilvusDense)       (e.g. CrossEncoder)
              |
    DenseEmbeddingFunction
    (SentenceTransformer)
              |
    +---------+---------+
    |                   |
  ingest(corpus)    search(queries)
    |                   |
  encode -> index    encode -> ANN search -> SearchResult
                        |
                    rerank (optional)
                        |
                    EvaluationEngine
                    (NDCG, MRR, Match@k)
```

## File Formats

| Format | Extension | Usage |
|---|---|---|
| Passages | `.tsv`, `.jsonl` | Corpus documents, field mapping via `data_template` |
| Queries | `.tsv`, `.jsonl` | Queries with relevance judgments, field mapping via `query_template` |
| Config | `.yaml`, `.json` | Engine configuration with `{{variable}}` templating |
| Output | `.json`, `.jsonl` | Search results |
| Cache | `.pkl.bz2` | Intermediate retrieval/rerank results |
| Embeddings | `.pkl` | Pre-computed embedding arrays (numpy) |

## Scripts Directory

| Script | Purpose |
|---|---|
| `compute_embeddings_from_jsonl.py` | Compute embeddings from JSONL with TextTiler support, output to pickle or Milvus |
| `export_to_openvino_onnx.py` | Export models to ONNX/OpenVINO format |
| `granite_embedding_model_converter.py` | Convert Granite models to ONNX via `optimum` |
| `granite_to_openvino_int8.py` | INT8 quantization for OpenVINO |
| `noreen_statistical_testing.py` | Noreen statistical significance tests |
| `onnx_vs_sentence_transformer_comparison.py` | Accuracy/speed comparison across backends |
| `probability_calibration_and_ece_evaluation.py` | ECE and calibration evaluation |
| `matryoshka/` | Matryoshka adapter and permutation training (see `matryoshka_training.md`) |
| `biohash/` | BioHash text hashing implementation |
| `milvus_utils/` | Milvus admin: copy, stats, list, query, HNSW index creation |
| `speed_test/` | Embedding timing benchmarks |
