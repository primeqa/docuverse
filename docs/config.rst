Configuration
=============

Every DocUVerse run — from the CLI, a preset, or the Python API — is driven
by a single configuration dictionary. That dictionary is assembled from a
YAML/JSON file (or a preset recipe), deep-merged with any overrides, and then
parsed into a set of typed dataclasses. This page covers where configs live,
how ``{{variable}}`` templating resolves, and the fields you are most likely
to set.

Configuration dataclasses
--------------------------

The config surface is a handful of HuggingFace-style ``@dataclass`` argument
groups, all defined in
``docuverse/engines/search_engine_config_params.py``. Each field carries a
default and a ``metadata={"help": ...}`` string that also feeds the CLI's
``--help`` output.

``RetrievalArguments``
~~~~~~~~~~~~~~~~~~~~~~~

The largest group — it configures the embedder, the chunker, and the
retrieval backend. Frequently-set fields:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Purpose
   * - ``db_engine``
     - ``es-bm25``
     - Backend to use (see :doc:`backends/index`).
   * - ``model_name``
     - ``""``
     - Embedding model name or path.
   * - ``index_name``
     - ``None``
     - Target collection / index name.
   * - ``top_k``
     - ``10``
     - Maximum number of results to return.
   * - ``max_doc_length``
     - ``512``
     - Max word-pieces per chunk.
   * - ``tile_overlap``
     - ``None``
     - Overlap (in tokens) between consecutive chunks.
   * - ``aligned_on_sentences``
     - ``True``
     - Snap chunk boundaries to sentence edges.
   * - ``bulk_batch``
     - ``512``
     - Database insert batch size.
   * - ``encode_batch_size``
     - ``512``
     - Documents per embedding-model encode call.
   * - ``matryoshka_dim``
     - ``0``
     - Truncate embeddings to N dims (``0`` = full).
   * - ``data_format``
     - ``None``
     - Data/query format config (see :doc:`data-formats`).

The same group also holds the backend-specific tuning blocks — the
``simdq_*`` fields (documented in :doc:`simdq/parameters`) and the
``fagin_*`` fields for the in-process engines.

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
     - Pipeline stages: ``i`` ngest, ``u`` pdate, ``r`` etrieve,
       ``R`` erank, ``e`` valuate.
   * - ``output_file``
     - ``None``
     - Where retrieval results are written.
   * - ``config``
     - ``None``
     - Path to a config file (CLI args override it).
   * - ``cache_dir``
     - ``None``
     - Cache location for intermediate results.

``RerankerArguments``
~~~~~~~~~~~~~~~~~~~~~~

See :doc:`reranking` for the full field list and behavior.

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

See also
--------

* :doc:`presets` — named recipes and the override merge order.
* :doc:`data-formats` — the ``data_format`` field-mapping configs.
* :doc:`backends/index` — per-backend configuration.
* :doc:`cli` — passing config via the command line.
