Configuration
=============

Every DocUVerse run — from the CLI, a preset, or the Python API — is driven
by a single configuration dictionary. That dictionary is assembled from a
YAML/JSON file (or a preset recipe), deep-merged with any overrides, and then
parsed into a set of typed dataclasses. This page covers where configs live,
how ``{{variable}}`` templating resolves, and the fields you are most likely
to set.

.. contents:: On this page
   :local:
   :depth: 2

File structure
--------------

The top-level keys in a config file (``retriever:``, ``reranker:``,
``evaluate:``) correspond to the dataclass groups below. You can also pass a
flat dict without those top-level keys — DocUVerse will route each key to the
right dataclass automatically.

.. code-block:: yaml

    retriever:
      # RetrievalArguments fields (model, index, preprocessing …)
      db_engine: milvus-dense
      model_name: ibm-granite/granite-embedding-small-english-r2
      index_name: my-index
      input_passages: benchmark/corpus.jsonl
      input_queries:  benchmark/queries.jsonl
      top_k: 40
      actions: ire

    reranker:          # null to disable
      reranker_model: null

    evaluate:
      eval_measure: match,mrr,ndcg
      ranks: 1,5,10

Configuration dataclasses
--------------------------

The config surface is a handful of HuggingFace-style ``@dataclass`` argument
groups, all defined in
``docuverse/engines/search_engine_config_params.py``. Each field carries a
default and a ``metadata={"help": ...}`` string that also feeds the CLI's
``--help`` output:

- **RetrievalArguments** — retrieval backend, model, and preprocessing.
- **RerankerArguments** — optional reranking stage.
- **EvaluationArguments** — metrics and rank cuts.
- **EngineArguments** — pipeline actions and I/O paths.

``RetrievalArguments``
~~~~~~~~~~~~~~~~~~~~~~~

The largest group — it configures the embedder, the chunker, and the
retrieval backend. These fields live under the ``retriever:`` key (or at the
top level for flat configs).

Data paths
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 24 8 12 56

   * - Field
     - Type
     - Default
     - Description
   * - ``input_passages``
     - str
     - ``None``
     - Path(s) to the document corpus (TSV, JSONL, or compressed variants).
       Multiple paths, glob patterns, and ``ds:`` HuggingFace specs are
       accepted — see :doc:`data-formats`.
   * - ``input_queries``
     - str
     - ``None``
     - Path to the query file. Leave ``None`` for ingestion-only runs.
   * - ``project_dir``
     - str
     - ``None``
     - Root directory for configuration and output files.
       Used when building index paths with ``{{project_dir}}``.
   * - ``data_format``
     - str
     - ``None``
     - Data/query field-mapping config (see :doc:`data-formats`).

Backend selection
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 24 8 12 56

   * - Field
     - Type
     - Default
     - Description
   * - ``db_engine``
     - str
     - ``"es-bm25"``
     - The retrieval backend to use. Underscores and dashes are
       interchangeable (``milvus_dense`` == ``milvus-dense``). Valid values:

       * ``milvus-dense`` (or just ``milvus``) — Milvus with dense embeddings.
       * ``milvus-sparse`` — Milvus with sparse (SPLADE) embeddings.
       * ``milvus-bm25`` — Milvus built-in BM25.
       * ``milvus-hybrid`` — Milvus hybrid dense + sparse.
       * ``milvus-splade`` — Milvus with a SPLADE model.
       * ``es-bm25`` / ``elastic-bm25`` — Elasticsearch BM25 (default).
       * ``es-dense`` / ``elastic-dense`` — Elasticsearch with dense vectors.
       * ``es-elser`` / ``elastic-elser`` — Elasticsearch ELSER sparse.
       * ``chromadb`` — ChromaDB vector store.
       * ``faiss`` — File-based FAISS index.
       * ``lancedb`` / ``lance`` / ``lancedb-dense`` — LanceDB dense; also
         ``lancedb-bm25``, ``lancedb-sparse``, ``lancedb-hybrid``.
       * ``simdq`` — in-process SIMD scan engine.
       * ``fagin`` — in-process exact top-k engine (Fagin's Threshold
         Algorithm). Also accepted as ``fagin-threshold``.

       See :doc:`backends/index` for a comparison table.
   * - ``server``
     - str
     - ``None``
     - Backend server endpoint. Format varies by engine:

       * Milvus embedded: ``file:/path/to/db``
       * Milvus remote:   ``http://host:19530``
       * Elasticsearch:   URL or named alias (or env vars — see below).
   * - ``index_name``
     - str
     - ``None``
     - Name of the index / collection to create or search.
       Supports ``{{…}}`` templating.

Model settings
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 8 12 52

   * - Field
     - Type
     - Default
     - Description
   * - ``model_name``
     - str
     - ``""``
     - HuggingFace model name or local path of the embedding model.
   * - ``model_torch_dtype``
     - str
     - ``"torch.float32"``
     - PyTorch dtype string for loading the model weights.
   * - ``attn_implementation``
     - str
     - ``"auto"``
     - Attention backend. ``"auto"`` selects ``flash_attention_2``
       when available on Ampere+ GPUs; otherwise falls back to ``sdpa``.
   * - ``torch_compile``
     - bool
     - ``False``
     - Apply ``torch.compile()`` for faster inference (PyTorch 2.0+).
   * - ``model_on_server``
     - bool
     - ``False``
     - When ``True``, DocUVerse assumes the model is hosted on the
       backend server (Elasticsearch ELSER pattern).
   * - ``matryoshka_dim``
     - int
     - ``0``
     - Truncate embeddings to the first N dimensions. ``0`` keeps the
       full model dimension.
   * - ``query_prompt_name``
     - str
     - ``None``
     - Prompt name passed to the encoder for queries (required by
       some instruction-following embedding models).
   * - ``document_prompt_name``
     - str
     - ``None``
     - Prompt name passed to the encoder for documents.

Document preprocessing
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 8 12 52

   * - Field
     - Type
     - Default
     - Description
   * - ``max_doc_length``
     - int
     - ``512``
     - Maximum chunk length **in word-pieces** (tokens by default).
       Documents longer than this are split into overlapping tiles.
   * - ``stride``
     - int
     - ``None``
     - The overlap between consecutive tiles when splitting documents —
       HuggingFace-tokenizer "stride" semantics, where the stride *is*
       the overlap. Prefer the clearer alias ``tile_overlap``; if both
       are given, ``stride`` wins.
   * - ``tile_overlap``
     - int
     - ``None``
     - Readable alias for ``stride``: the number of tokens/chars shared
       between consecutive tiles. Ignored when ``stride`` is set.
   * - ``count_type``
     - str
     - ``"token"``
     - Unit for ``max_doc_length`` and ``stride``. ``"token"``
       counts word-pieces; ``"char"`` counts characters.
   * - ``aligned_on_sentences``
     - bool
     - ``True``
     - When ``True``, tile boundaries snap to sentence ends to avoid
       cutting mid-sentence.
   * - ``sentence_segmenter``
     - str
     - ``"pyizumo"``
     - Sentence segmentation backend. Options: ``"pyizumo"``, ``"spacy"``.
   * - ``doc_based``
     - bool
     - ``False``
     - When ``True``, ingest whole documents instead of passage-level
       chunks.
   * - ``title_handling``
     - str
     - ``"all"``
     - How to prepend document titles to tiles:
       ``"all"`` (every tile), ``"first"`` (first tile only),
       ``"none"`` (no titles).
   * - ``max_text_size``
     - int
     - ``-1``
     - Maximum stored text length. ``-1`` = no limit; ``0`` disables
       text storage entirely.
   * - ``store_text_in_index``
     - bool
     - ``True``
     - If ``False``, only embeddings are stored; raw text is not kept.
   * - ``duplicate_removal``
     - str
     - ``None``
     - Duplicate suppression on retrieved results: ``"rouge"``,
       ``"exact"``, or ``"key:<field>"`` (keep one entry per value of
       that field, e.g. one per URL).
   * - ``rouge_duplicate_threshold``
     - float
     - ``0.9``
     - ROUGE threshold above which two passages are considered
       duplicates (only active when ``duplicate_removal="rouge"``).
   * - ``lang``
     - str
     - ``"en"``
     - Language code passed to the segmenter.

Ingestion performance
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 8 12 52

   * - Field
     - Type
     - Default
     - Description
   * - ``bulk_batch``
     - int
     - ``512``
     - Number of documents per bulk-write call to the backend.
   * - ``encode_batch_size``
     - int
     - ``512``
     - Documents per embedding-model encode call. Decoupled from
       ``bulk_batch`` (the DB insert size) so the GPU batch can be tuned
       independently.
   * - ``ingestion_batch_size``
     - int
     - ``40``
     - Elasticsearch only: the ingestion batch size.
   * - ``num_preprocessor_threads``
     - int
     - ``-1``
     - Thread count for text preprocessing. ``-1`` = automatic.

Search
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 8 12 52

   * - Field
     - Type
     - Default
     - Description
   * - ``top_k``
     - int
     - ``10``
     - Number of results to return per query.
   * - ``num_search_threads``
     - int
     - ``-1``
     - Thread count for parallel query execution.
   * - ``hybrid``
     - str or dict
     - ``{}``
     - Fusion strategy for hybrid search: ``"rrf"`` or a dict.
   * - ``hybrid_submodules``
     - str
     - ``None``
     - Comma-separated sub-module names to activate during hybrid search
       (useful for Milvus hybrid when only some modules should run).

Volume limits
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 8 12 52

   * - Field
     - Type
     - Default
     - Description
   * - ``max_num_documents``
     - int
     - ``None``
     - Cap on the number of documents ingested (useful for smoke tests).
   * - ``max_num_questions``
     - int
     - ``None``
     - Cap on the number of queries run.
   * - ``ignore_empty_questions``
     - bool
     - ``False``
     - Skip queries that have no relevant documents in the ground truth.
   * - ``docid_filter``
     - str
     - ``None``
     - Path to a file listing document IDs to ingest; all others skipped.
   * - ``exclude_docid_filter``
     - str
     - ``None``
     - Inverse of ``docid_filter``: document IDs listed here are skipped.

simdq fields
^^^^^^^^^^^^

These fields are only active when ``db_engine: simdq``. For a full
guide with tuning advice see :doc:`simdq/parameters`.

.. list-table::
   :header-rows: 1
   :widths: 24 8 12 56

   * - Field
     - Type
     - Default
     - Description
   * - ``simdq_family``
     - str
     - ``"asymmetric"``
     - Scan kernel: ``"asymmetric"`` (float query × b-bit codes) or
       ``"hamming"`` (1-bit symmetric).
   * - ``simdq_b``
     - int
     - ``2``
     - Bits per dimension (1, 2, or 4). Ignored when
       ``simdq_family="hamming"``.
   * - ``simdq_d``
     - int
     - ``None``
     - Reduced embedding dimension. ``None`` keeps the full model dim;
       otherwise must equal D or D/2.
   * - ``simdq_projection``
     - str
     - ``"identity"``
     - Projection before quantisation: ``"identity"``,
       ``"random_orthogonal"`` (required when ``simdq_d = D/2``), or
       ``"learned_orthogonal"`` (ITQ rotation fit on the corpus).
   * - ``simdq_projection_seed``
     - int
     - ``42``
     - RNG seed for the random / learned orthogonal projection.
   * - ``simdq_itq_iters``
     - int
     - ``50``
     - ITQ alternating-minimisation iterations for
       ``projection="learned_orthogonal"``.
   * - ``simdq_standardize``
     - bool
     - ``False``
     - Fit an affine ``(x−mu)/sigma`` standardizer on the corpus and apply
       it before the projection, balancing the sign bits.
   * - ``simdq_store_floats``
     - bool
     - ``True``
     - Write ``floats.bin`` for the two-stage rescore tier. Disable to
       halve index size at the cost of recall.
   * - ``simdq_rescore_alpha``
     - int
     - ``10``
     - K' = ``simdq_rescore_alpha × top_k`` candidates rescored.
       Set to ``1`` to skip rescoring (codes-only mode).
   * - ``simdq_num_threads``
     - int
     - ``0``
     - OpenMP thread count for the scan kernel. ``0`` = OMP default.
   * - ``simdq_ivf_nlist``
     - int
     - ``0``
     - Hamming only: number of IVF clusters for the outer index.
       ``0`` = no IVF (flat scan over all codes).
   * - ``simdq_ivf_nprobe``
     - int
     - ``32``
     - Hamming IVF: nearest clusters scanned per query. Higher = better
       recall, slower. Ignored if no IVF.

fagin fields
^^^^^^^^^^^^

These fields are only active when ``db_engine: fagin``. The defaults
give exact top-k inner-product search; see :doc:`backends/fagin` for a
full guide with tuning advice.

.. list-table::
   :header-rows: 1
   :widths: 24 8 12 56

   * - Field
     - Type
     - Default
     - Description
   * - ``fagin_batch_rows``
     - int
     - ``64``
     - Rows of sorted access taken from each active dimension's list per
       TA round. Larger = fewer threshold checks but more overshoot;
       does not affect correctness.
   * - ``fagin_epsilon``
     - float
     - ``0.0``
     - Additive halting slack: stop when the k-th best score is
       ``≥ T − epsilon``. ``0.0`` = exact Threshold Algorithm.
   * - ``fagin_max_depth``
     - int
     - ``0``
     - Cap on sorted-access depth (rows per dimension). ``0`` =
       unlimited (exact); nonzero makes results approximate.
   * - ``fagin_num_threads``
     - int
     - ``0``
     - OpenMP thread count for the TA scan kernel. ``0`` = OMP default.

``EngineArguments``
~~~~~~~~~~~~~~~~~~~~

Top-level pipeline control:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Purpose
   * - ``actions``
     - ``ir``
     - Pipeline stages to run. Each character activates one stage:
       ``i`` ngest, ``u`` pdate, ``r`` etrieve, ``R`` erank,
       ``e`` valuate. Example: ``"ire"`` ingests, retrieves, and
       evaluates.
   * - ``output_file``
     - ``None``
     - Where retrieval results are written (JSON or JSONL).
   * - ``config``
     - ``None``
     - Path to a config file (CLI args override it).
   * - ``cache_dir``
     - ``None``
     - Cache location for intermediate results.
   * - ``no_cache``
     - ``False``
     - Ignore on-disk caches and recompute everything.
   * - ``skip``
     - ``0``
     - Skip the first N documents during ingestion (resume support).

``RerankerArguments``
~~~~~~~~~~~~~~~~~~~~~~

Fields live under the ``reranker:`` key; set ``reranker: null`` to disable
reranking entirely. See :doc:`reranking` for the full field list and
behavior.

``EvaluationArguments``
~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`evaluation`. Key fields: ``eval_measure`` (default ``match``),
``ranks`` (default ``"1,3,5"``), and ``match_by`` (default ``id``).

``{{variable}}`` templating
---------------------------

Config files may reference other config values with ``{{...}}`` (and use
``{% ... %}`` control blocks). Despite the Jinja-looking syntax, note the
resolver is the real Jinja2 engine, run in ``StrictUndefined`` mode by
``_render_with_jinja2`` in ``docuverse/utils/__init__.py``:

.. code-block:: yaml

    model_name: ibm-granite/granite-embedding-small-english-r2
    index_name: "docuverse_{{ model_name | short_model }}"

Behavior worth knowing:

* **Strict undefined.** Referencing a variable that does not exist raises a
  ``RuntimeError`` naming the dotted key path — there are no silent empty
  substitutions.
* **Multi-pass.** The tree is rendered repeatedly (up to 10 passes) until it
  stabilizes, so a value may reference another templated value. Circular
  references raise an error.
* **Unscoped names are auto-qualified.** A bare ``{{model_name}}`` is
  rewritten to its fully-qualified path (e.g. ``{{retriever.model_name}}``).
  If the same name exists in more than one nested scope the resolver raises
  an ambiguity error — qualify it yourself in that case.
* **Custom filter.** ``short_model`` derives a compact identifier from a
  model name (handy for building index names).
* **Plain strings at runtime.** Templating is resolved before the
  dataclasses are populated, so every field resolved this way is still a
  plain string at runtime.

.. note::

   ``_process_params`` still exists in the codebase but is a legacy no-op —
   it is no longer called by ``read_config_file``. All templating now goes
   through the Jinja2 pipeline described above.

How a config is loaded
----------------------

``read_config_file`` (in ``docuverse/utils/__init__.py``) runs this pipeline:

#. Resolve the path (relative paths are looked up via ``get_config_dir()``).
#. Load the file (YAML or JSON, detected by extension).
#. Apply ``override`` values — dotted keys like ``retriever.top_k=10`` walk
   into nested dicts; bare leaf keys replace all matching scalars.
#. Rewrite unscoped ``{{var}}`` names to fully-qualified paths.
#. Render every string leaf through Jinja2.

The Python API exposes three entry points on
:py:class:`docuverse.SearchEngine`, all of which produce a merged dict and
hand it to ``DocUVerseConfig``:

.. code-block:: python

    from docuverse import SearchEngine

    # From a named preset, with keyword overrides:
    engine = SearchEngine.from_preset("milvus-dense", top_k=20)

    # From a YAML file, with optional overrides:
    engine = SearchEngine.from_yaml("recipe.yaml", top_k=20)

    # From an in-memory dict:
    engine = SearchEngine.from_dict({"db_engine": "faiss", ...})

Overrides are deep-merged last, so they always win over the file or preset.
See :doc:`presets` for the full merge order.

You can also load a config file directly into the typed object:

.. code-block:: python

    from docuverse.engines.search_engine_config_params import DocUVerseConfig

    cfg = DocUVerseConfig("experiments/my_run.yaml")
    # cfg.retriever_config, cfg.reranker_config, cfg.eval_config,
    # cfg.run_config are the four typed sub-configs.
    # All fields are also promoted to the top-level object:
    print(cfg.db_engine, cfg.top_k)

Config file locations
----------------------

Config files are resolved by ``get_config_dir()``, which searches the current
directory, the project root, and the packaged defaults — unless overridden:

* ``DOCUVERSE_CONFIG_PATH`` — if set, this directory becomes the root for
  resolving relative config paths.

The packaged ``config/`` directory holds reusable **data-format** configs —
field-mapping files for common datasets (BEIR, ClapNQ, SAP, …). These are
referenced by the ``data_format`` field and are documented in
:doc:`data-formats`.

Elasticsearch credentials
--------------------------

Elasticsearch connection details are read from the environment when no
explicit ``server`` is configured (via ``load_dotenv()``, so a local
``.env`` file works too):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Meaning
   * - ``ES_HOST``
     - Host URL, e.g. ``https://localhost:9200``.
   * - ``ES_USER``
     - Basic-auth username.
   * - ``ES_PASSWORD``
     - Basic-auth password.
   * - ``ES_API_KEY``
     - API key (alternative to user/password).
   * - ``ES_SSL_FINGERPRINT``
     - SSL certificate fingerprint to pin.

Complete example
----------------

.. code-block:: yaml

    retriever:
      corpus:       scifact
      project_dir:  experiments/scifact
      input_passages: ds:BeIR/scifact:corpus
      input_queries:  ds:BeIR/scifact:queries
      ignore_empty_questions: true

      db_engine:    milvus-dense
      server:       "file:{{project_dir}}/{{corpus}}.db"
      model_name:   ibm-granite/granite-embedding-278m-multilingual
      index_name:   "{{corpus}}-dense-granite-512"

      top_k:        40
      actions:      ire
      max_doc_length: 512
      stride:       100
      title_handling: all
      aligned_on_sentences: true
      bulk_batch:   128
      output_file:  "output/{{index_name}}.json"

    reranker: null

    evaluate:
      eval_measure: ndcg,match,mrr
      ranks: 1,5,10,40

Caching
-------

DocUVerse caches expensive intermediate results next to ``output_file``
(or in ``cache_dir`` if set):

- ``*.retrieve.pkl.bz2`` — raw retrieval results.
- ``*.rerank.pkl.bz2`` — reranked results.

Set ``no_cache: true`` or delete these files to force recomputation.

See also
--------

* :doc:`quickstart` — end-to-end example.
* :doc:`presets` — named recipes and the override merge order.
* :doc:`data-formats` — the ``data_format`` field-mapping configs.
* :doc:`backends/index` — per-backend configuration.
* :doc:`simdq/parameters` — simdq-specific parameter guide.
* :doc:`cli` — passing config via the command line.
