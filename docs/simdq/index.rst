simdq — in-process binary-quantized retrieval engine
=====================================================

``simdq`` is a DocUVerse retrieval engine for ≤10M-vector evaluation that
runs entirely in-process: no Milvus, no Elasticsearch, no network. It uses
SIMD-accelerated binary and asymmetric scalar quantization to fit a typical
BEIR corpus into a few hundred megabytes of RAM and answer queries in
**sub-millisecond** time on a single CPU — e.g. on Natural Questions
(276k × 768-dim) it serves ~0.4 ms/query, lossless vs. fp32 and ~29× faster
than an exact Milvus FLAT baseline. See :doc:`tuning` (*Performance results*).

If you want to ...

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Goal
     - Page
   * - Try simdq in 5 minutes on SciFact
     - :doc:`quickstart`
   * - Look up what ``simdq_b`` (or any other field) does
     - :doc:`parameters`
   * - Pick a recipe for your corpus
     - :doc:`tuning`
   * - Understand *why* b=2 vs 1-bit + rescore
     - :doc:`tuning` (Math intuition)
   * - See latest speed/quality numbers vs Milvus
     - :doc:`tuning` (Performance results)
   * - Ingest your own dataset
     - :doc:`adapting`
   * - Run the R0-R9 recipe sweep
     - :doc:`adapting` (Recipe sweep)
   * - Debug a crash, slow query, or low recall
     - :doc:`troubleshooting`

.. toctree::
   :hidden:
   :maxdepth: 1

   quickstart
   parameters
   tuning
   adapting
   troubleshooting
