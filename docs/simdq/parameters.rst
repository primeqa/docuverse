simdq parameters
================

This page is the authoritative reference for every ``simdq_*`` field on
``RetrievalArguments``. For tuning guidance see :doc:`tuning`; for error
messages see :doc:`troubleshooting`.

.. list-table:: simdq\_\* fields on ``RetrievalArguments``
   :header-rows: 1
   :widths: 22 8 12 18 22 18

   * - Field
     - Type
     - Default
     - Valid values
     - Effect
     - When to change
   * - ``simdq_family``
     - str
     - ``"asymmetric"``
     - ``asymmetric`` | ``hamming``
     - Selects scan kernel family
     - Switch to ``hamming`` for 1-bit Hamming + rescore (R0)
   * - ``simdq_b``
     - int
     - ``2``
     - 1, 2, 4
     - Bits/dim in asymmetric codes
     - b=1 → 1-byte/8 dims; b=2 → 1-byte/4 dims; b=4 → 1-byte/2 dims
   * - ``simdq_d``
     - int?
     - ``None``
     - None, D, D/2
     - Reduced dim; None ⇒ D
     - Halve to shrink index 2× at the cost of recall
   * - ``simdq_projection``
     - str
     - ``"identity"``
     - ``identity`` | ``random_orthogonal``
     - Projection W (d×D)
     - ``random_orthogonal`` is required for d=D/2
   * - ``simdq_projection_seed``
     - int
     - ``42``
     - any int
     - RNG seed for the projection
     - Pin for reproducibility across hosts
   * - ``simdq_store_floats``
     - bool
     - ``True``
     - True / False
     - Whether ``floats.bin`` is written for rescore
     - False saves disk; disables two-stage rescore
   * - ``simdq_rescore_alpha``
     - int
     - ``10``
     - ≥ 1
     - K' = α·K candidates rescored
     - 1 ⇒ codes-only; 10 closes most of the recall gap
   * - ``simdq_num_threads``
     - int
     - ``0``
     - 0…N
     - OMP threads for scan; 0 = OMP default
     - Tune to ``#physical cores`` for memory-bound asym scans

Cross-cutting fields
--------------------

The following top-level ``RetrievalArguments`` fields also affect simdq runs
but are shared with all backends; see the top-level config documentation for
full descriptions.

- ``top_k`` — number of results returned per query.
- ``index_name`` — name of the simdq index directory written under
  ``project_dir``.
- ``project_dir`` — root directory where index files (``codes.bin``,
  ``floats.bin``, ``meta.json``) are stored.
- ``model_name`` — HuggingFace encoder used to embed documents and queries;
  its output dimension sets D.
- ``bulk_batch`` — number of documents encoded per batch during ingestion.

.. note::

   This table mirrors the ``simdq_*`` fields in
   ``docuverse/engines/search_engine_config_params.py``.  Update both this
   page and the source dataclass together whenever a field is added, removed,
   or renamed.
