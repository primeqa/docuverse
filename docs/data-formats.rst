Data formats
============

DocUVerse reads three kinds of input — a **corpus** of passages, a set of
**queries**, and (for evaluation) **qrels** of gold relevance judgments — and
writes retrieval **results** as JSON. Field names are not hard-coded: a
*data-format* config maps the JSON/TSV columns in your files onto the internal
field names, so the same code ingests BEIR, ClapNQ, SAP, and your own data
without changes.

The corpus
----------

Passages are read by :py:class:`docuverse.SearchCorpus`
(``docuverse/engines/search_corpus.py``). Both JSONL and TSV are supported.
A minimal JSONL record:

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

``relevant`` holds the gold document IDs inline. Alternatively, a query
config JSON can point at separate files with ``question_file`` and
``goldstandard_file``, in which case relevance comes from a qrels file.

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
