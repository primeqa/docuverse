Data formats
============

DocUVerse reads three kinds of input — a **corpus** of passages, a set of
**queries**, and (for evaluation) **qrels** of gold relevance judgments — and
writes retrieval **results** as JSON. Field names are not hard-coded: a
*data-format* config maps the JSON/TSV columns in your files onto the internal
field names, so the same code ingests BEIR, ClapNQ, SAP, and your own data
without changes.

The two config fields that point at your data are ``input_passages`` (the
corpus to ingest) and ``input_queries`` (the queries to run and evaluate).
Both accept the same range of source types described below.

Input sources
-------------

Local files
~~~~~~~~~~~

Three file layouts are recognized by extension:

* **JSONL** — one JSON object per line (the canonical format, used by the
  quickstart example and BEIR datasets).
* **JSON** — a single file holding a JSON array of the same objects.
* **TSV** — tab-separated with a header row. Fields whose values look like
  Python lists (e.g. ``"['a', 'b']"``) are automatically coerced to lists.

Glob patterns
~~~~~~~~~~~~~

Shell-style glob patterns and brace-expansion are expanded before
reading, letting you point at a directory of shards in one line:

.. code-block:: yaml

   retriever:
     input_passages: data/shards/*.jsonl

   # or brace-expansion:
     input_passages: data/{train,dev,test}.jsonl.bz2

Compressed files (``.bz2``, ``.gz``, ``.xz``) are decompressed on the
fly. A deterministic cache key is derived from the glob pattern so the
processed-passage cache (see `Chunking and the processing cache`_) still
works across runs.

Multiple files
~~~~~~~~~~~~~~

Pass a YAML list when your corpus lives in several separate files:

.. code-block:: yaml

   retriever:
     input_passages:
       - data/corpus_part1.jsonl
       - data/corpus_part2.jsonl.bz2

Files are read in order and concatenated before ingestion.

HuggingFace datasets
~~~~~~~~~~~~~~~~~~~~

Prefix the dataset path with ``ds:`` to load directly from the
HuggingFace Hub without downloading manually:

.. code-block:: yaml

   retriever:
     input_passages: ds:BeIR/scifact:corpus
     input_queries:  ds:BeIR/scifact:queries

See `HuggingFace dataset format`_ for the full syntax.

The corpus
----------

Passages are read by :py:class:`docuverse.SearchCorpus`
(``docuverse/engines/search_corpus.py``). A minimal JSONL record:

.. code-block:: json

    {"id": "p001", "title": "Photosynthesis", "text": "Photosynthesis is the biological process..."}

Internal passage fields (and the config keys that map to them):

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Field
     - Default header(s)
     - Config key
   * - id
     - ``id`` / ``_id`` / ``docid``
     - ``id_header``
   * - text
     - ``text`` / ``document``
     - ``text_header``
   * - title
     - ``title``
     - ``title_header``

Nested / document-based inputs (a document containing a list of passages) are
handled via ``passage_header``, ``passage_text_header``, and
``passage_id_header``.

Queries
-------

Queries are read by :py:class:`docuverse.SearchQueries`
(``docuverse/engines/search_queries.py``):

.. code-block:: json

    {"id": "q1", "text": "How do plants make energy from sunlight?", "relevant": ["p001"]}

Internal query fields:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Field
     - Default header(s)
     - Config key
   * - id
     - ``id`` / ``qid`` / ``_id``
     - ``id_header``
   * - text
     - ``text`` / ``query`` / ``question``
     - ``text_header``
   * - relevant
     - ``relevant``
     - ``relevant_header``
   * - answers
     - ``answers``
     - ``answers_header``

``relevant`` holds the gold document IDs inline (in TSV files, as a
comma-separated list in the ``relevant`` column). Alternatively, a query
config JSON can point at separate files with ``question_file`` and
``goldstandard_file``, in which case relevance comes from a qrels file.

Queries with an empty ``relevant`` list are kept by default. Set
``ignore_empty_questions: true`` to skip them (useful when ``relevant``
is populated at load time from HuggingFace qrels — see below).

Qrels (gold relevance)
----------------------

When gold judgments live in a separate file, DocUVerse expects the standard
TREC-style qrels layout:

.. code-block:: text

    query_id   iteration   doc_id   relevance
    q1         0           p001     1
    q2         0           p002     1

The columns that identify the query and document are configurable via
``truth_id`` (default ``query-id``) and ``truth_label`` (default
``corpus-id``). Each row contributes ``doc_id`` to the gold set for its
``query_id``.

HuggingFace dataset format
---------------------------

The ``ds:`` prefix instructs DocUVerse to call
``datasets.load_dataset`` instead of opening a local file.

**Syntax**::

    ds:<dataset>[:<subset>[:<split>]]

All three components after ``ds:`` are separated by colons:

============  ===============================================================
Component     Meaning
============  ===============================================================
``dataset``   HuggingFace dataset ID, e.g. ``BeIR/scifact``
``subset``    Dataset configuration name, e.g. ``corpus``, ``queries``,
              ``qrels``
``split``     HuggingFace split, e.g. ``train``, ``test``.
              When omitted, defaults to ``train`` for corpus and ``test``
              for qrels auto-loading.
============  ===============================================================

**Examples**

.. code-block:: yaml

   # BEIR SciFact corpus (train split by default)
   input_passages: ds:BeIR/scifact:corpus

   # All queries with qrels merged from the test split
   input_queries: ds:BeIR/scifact:queries

   # Queries from a specific split
   input_queries: ds:BeIR/scifact:queries:test

**Field normalisation**

BEIR datasets use ``_id`` rather than ``id``. DocUVerse automatically
adds ``id`` as an alias so the default data template resolves both
without any extra configuration.

**qrels auto-merge**

When the subset is ``queries``, DocUVerse automatically fetches the
matching qrels (same split, defaulting to ``test``) and merges the
relevance judgements into each query record under the ``relevant`` key.
It first tries a ``qrels`` subset of the same dataset; if that fails it
tries ``<dataset>-qrels`` (the BEIR convention, e.g.
``BeIR/scifact-qrels``). The raw qrel format expected from HuggingFace is:

.. code-block:: text

   query-id    corpus-id    score
   q1          p001         1
   q2          p002         1

After merging, each query dict contains ``"relevant": ["p001"]`` etc.,
ready for evaluation. Queries that appear in no qrel row receive an empty
list.

.. note::

   Set ``ignore_empty_questions: true`` alongside any ``ds:…:queries``
   spec to skip queries that have no qrel entries in the chosen split, so
   they do not skew evaluation metrics.

Data-format config files
-------------------------

A data-format config declares two sections — ``data_format`` (corpus) and
``query_format`` (queries) — each a set of header mappings. These live in the
packaged ``config/`` directory and are selected with the ``data_format``
field (see :doc:`config`). The BEIR mapping, for example:

.. code-block:: yaml

    data_format:
      id_header: _id
      text_header: text
      title_header: title
      extra_fields: null

    query_format:
      id_header: _id
      text_header: text
      relevant_header: relevant
      extra_fields: null
      truth_id: "query-id"
      truth_label: "corpus-id"

Datasets with idiosyncratic column names just override the headers. The SAP
mapping, for instance, reads ``document_id`` / ``document`` for the corpus and
``Count`` / ``Question`` for queries, and preserves extra columns (``url``,
``filestem``) via ``extra_fields``.

The full set of supported header keys is defined by ``DataTemplate`` in
``docuverse/engines/data_template.py``: ``id_header``, ``text_header``,
``title_header``, ``relevant_header``, ``answers_header``, ``passage_header``,
``passage_text_header``, ``passage_id_header``, ``truth_id``, ``truth_label``,
``keep_fields``, and ``extra_fields``.

Chunking and the processing cache
---------------------------------

After loading, corpus documents are optionally chunked before ingestion,
controlled by ``max_doc_length``, ``tile_overlap``/``stride``,
``aligned_on_sentences``, and ``title_handling`` (see :doc:`config` for the
full field reference). ``title_handling`` accepts three values:

=========  ==============================================================
Value      Effect
=========  ==============================================================
``all``    The document title is prepended to **every** chunk (default).
``first``  The title is prepended only to the first chunk.
``none``   Titles are not added to any chunk.
=========  ==============================================================

Chunked passages are cached on disk so that re-running with the same
corpus and chunking parameters skips re-processing entirely. The cache
lives under::

    ~/.local/share/elastic_ingestion/

Delete a cache file (or the whole directory) to force re-processing.

Results output
--------------

Retrieval results are written as one JSON object per query by
:py:class:`docuverse.SearchResult` (``docuverse/engines/search_result.py``).
Each record pairs the original question with its ranked passages:

.. code-block:: json

    {
      "question": {
        "text": "How do plants make energy from sunlight?",
        "id": "q1",
        "relevant": ["p001"]
      },
      "retrieved_passages": [
        {"id": "p001-0-174", "title": "Photosynthesis", "text": "Photosynthesis is...", "score": 0.8777},
        {"id": "p002-0-191", "title": "Mitochondria",   "text": "Mitochondria are...",  "score": 0.7737}
      ]
    }

Each retrieved passage carries at least ``id``, ``text``, and ``score``;
``title`` and any preserved extra fields appear when present. Note the
passage ``id`` (e.g. ``p001-0-174``) encodes the chunk offsets appended to the
original document id.

Alongside the results file, a run also leaves stage caches:

.. code-block:: text

   results.json              — ranked results for each query
   results.retrieve.pkl.bz2  — cached retrieval results
   results.rerank.pkl.bz2    — cached reranking results

The ``.pkl.bz2`` cache files let you iterate quickly: re-running with
only the ``e`` (evaluate) action reads directly from the cache without
touching the index.

Format conversions
------------------

The ``convert`` CLI subcommand (``docuverse/cli/cmd_convert.py``) offers two
helpers:

.. code-block:: bash

    # Explode a JSON array into one object per line:
    docuverse convert json-to-jsonl input.json -o output.jsonl

    # Emit TREC qrels from a results file (gold + system runs):
    docuverse convert to-trec -i results.jsonl -g gold.qrels -s system.qrels

``to-trec`` strips the trailing chunk offsets from passage ids
(``p001-0-174`` → ``p001``) so a run scores against document-level qrels.

See also
--------

* :doc:`config` — selecting a data-format config with ``data_format``.
* :doc:`evaluation` — how qrels are used to score a run.
* :doc:`quickstart` — an end-to-end example with real files.
* :doc:`simdq/quickstart` — BEIR SciFact via the ``ds:`` prefix.
* :doc:`simdq/adapting` — custom field names and data templates.
