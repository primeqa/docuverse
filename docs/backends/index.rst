Backends
========

DocUVerse presents one interface — :py:class:`docuverse.SearchEngine` — over
several retrieval backends. You pick a backend with the ``db_engine`` config
field (or a preset); the dispatcher in ``docuverse/utils/retrievers.py``
(``create_retrieval_engine``) maps that string to a concrete engine class.
Engine names are case-insensitive and treat ``-`` and ``_`` as
interchangeable, so ``milvus-dense`` and ``milvus_dense`` are the same.

Available backends
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - ``db_engine``
     - Backend
     - Modes
   * - ``es-bm25`` / ``elastic-bm25``
     - Elasticsearch
     - BM25 keyword search.
   * - ``es-dense`` / ``elastic-dense``
     - Elasticsearch
     - Dense kNN (optional RRF hybrid).
   * - ``es-elser`` / ``elastic-elser``
     - Elasticsearch
     - Learned-sparse ELSER (server-side).
   * - ``milvus`` / ``milvus-dense``
     - Milvus
     - Dense (HNSW / IVF / AUTOINDEX).
   * - ``milvus-sparse``
     - Milvus
     - Learned-sparse (SPLADE-style).
   * - ``milvus-bm25``
     - Milvus
     - BM25 (sparse inverted index).
   * - ``milvus-splade``
     - Milvus
     - SPLADE sparse expansion.
   * - ``milvus-hybrid``
     - Milvus
     - Dense + sparse fusion.
   * - ``chromadb``
     - ChromaDB
     - Dense (cosine).
   * - ``faiss``
     - FAISS
     - Dense, in-memory (Flat/IVF/HNSW).
   * - ``lancedb`` / ``lance``
     - LanceDB
     - Dense; also ``-bm25`` / ``-sparse`` / ``-hybrid``.
   * - ``simdq``
     - simdq (in-process)
     - SIMD binary-quantized dense.
   * - ``fagin`` / ``fagin-threshold``
     - Fagin (in-process)
     - Exact top-k via Threshold Algorithm.

The in-process engines need no external service; simdq has its own guide (see
:doc:`../simdq/index`).

Installing backend dependencies
-------------------------------

The optional backend clients are extras, so a base install stays light. Only
the backend you use is imported (lazily — see below), so you install just what
you need:

.. code-block:: bash

    pip install -e ".[milvus]"      # pymilvus (+ milvus-lite)
    pip install -e ".[elastic]"     # elasticsearch 8.x
    pip install -e ".[chromadb]"    # chromadb
    pip install -e ".[faiss]"       # faiss-cpu
    pip install -e ".[lancedb]"     # lancedb
    pip install -e ".[all]"         # everything

The ``simdq`` and ``fagin`` engines are in-process and build a native
extension locally; they do not require a client package.

.. note::

   Optional backend packages (``pymilvus``, ``elasticsearch``, ``chromadb``,
   …) are imported **lazily** — inside ``create_retrieval_engine`` and behind
   ``try/except`` — never at module import time. This is why ``docuverse
   --help`` and ``import docuverse`` stay fast and work even when a backend
   package is not installed; picking a backend you have not installed raises a
   clear "you need to install …" message rather than an import error at
   startup.

Common configuration
--------------------

Regardless of backend, these fields apply (see :doc:`../config`):

* ``db_engine`` — the backend selector (required).
* ``index_name`` — collection / index name.
* ``model_name`` — embedding model for the dense backends.
* ``top_k`` — number of results.
* ``server`` — connection spec: a dict, a named registry key, or a ``file:``
  URL for embedded/local stores (e.g. ``file:./.docuverse/milvus.db``).

Elasticsearch
~~~~~~~~~~~~~~

Connection details are read from the environment when ``server`` is not set
(see :doc:`../config`): ``ES_HOST``, ``ES_USER``, ``ES_PASSWORD``,
``ES_API_KEY``, ``ES_SSL_FINGERPRINT``. The ELSER variant runs the model
server-side (``model_on_server``); the dense variant can enable an RRF hybrid
combining BM25 and kNN.

Milvus
~~~~~~

``server`` may be a ``{host, port}`` dict, a named server, or a ``file:`` path
for Milvus-Lite (the quickstart default). The dense engine defaults to an
HNSW index with inner-product metric; ``index_params`` / ``search_params``
accept custom dicts or named presets.

FAISS / ChromaDB
~~~~~~~~~~~~~~~~~

Both are local dense stores keyed on ``project_dir``. ChromaDB is dense-only
in this project (its local mode does not enable sparse indexing); use the
Milvus sparse/hybrid variants for BM25, SPLADE, or hybrid retrieval.

LanceDB
~~~~~~~

Dense by default (``metric`` = ``dot`` / ``cosine`` / ``euclidean``), with
BM25, sparse, and hybrid variants. ``index_params`` selects the ANN index
type (e.g. ``ivf_pq``, ``lsh``).

Milvus admin CLI
----------------

The ``db`` subcommand wraps common Milvus admin tasks
(``docuverse/cli/cmd_db.py``):

.. code-block:: bash

    docuverse db stats      # print collection stats
    docuverse db list       # list collections
    docuverse db copy ...   # copy data between Milvus DBs
    docuverse db query ...  # run a query against a collection

See also
--------

* :doc:`../presets` — one-line recipes per backend.
* :doc:`../config` — the shared configuration fields.
* :doc:`../simdq/index` — the in-process quantized engine.
