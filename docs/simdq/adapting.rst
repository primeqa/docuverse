Adapting SimdQ to New Datasets and Encoders
============================================

This page covers extending the SimdQ retrieval engine to new datasets,
new encoders (including unsupported embedding dimensions), custom corpus
formats, and additional entries in the recipe sweep.

New BEIR-Format Dataset
-----------------------

SimdQ expects your dataset in standard BEIR layout: three files,
optionally living anywhere on disk.

**File format**

.. code-block:: text

   passages.jsonl  — one JSON object per line: {"id": "...", "text": "...", "title": "..."}
   queries.jsonl   — one JSON object per line: {"id": "...", "text": "..."}
   qrels.tsv       — tab-separated: query-id <TAB> corpus-id <TAB> relevance-score

.. note::

   **The qrels file is optional.**  You only need it when the queries file does
   **not** carry its relevance judgements inline.  If each query object already
   has a ``relevant`` key listing its relevant corpus-ids, evaluation reads the
   judgements straight from the queries file and no separate ``qrels`` file is
   required:

   .. code-block:: text

      queries.jsonl   — {"id": "...", "text": "...", "relevant": ["corpus-id-1", "corpus-id-2"]}

   (The key name follows the template's ``relevant_header``, which defaults to
   ``relevant``.)  Supply ``qrels`` only as the external alternative, for
   BEIR-style datasets that ship judgements in a separate file.

**Pointing the config at your files**

In ``config/beir_simdq_base.yaml`` (or a copy you derive from it), set
the three path keys under ``retrieval`` and ``evaluation``:

.. code-block:: yaml

   retrieval:
     input_passages: data/mydata/passages.jsonl
     input_queries:  data/mydata/queries.jsonl

   evaluation:
     qrels: data/mydata/qrels.tsv   # omit if queries.jsonl has a "relevant" key

The benchmark script fills these in from CLI flags, so for a sweep you
rarely need to edit the base YAML directly.  Leave ``qrels`` unset when your
queries file already carries inline relevance keys (see the note above).

**Running the sweep**

.. code-block:: bash

   python scripts/bench_simdq_beir.py \
       --dataset fiqa \
       --base-yaml config/beir_simdq_base.yaml \
       --passages data/fiqa/passages.jsonl \
       --queries  data/fiqa/queries.jsonl \
       --qrels    data/fiqa/qrels.tsv \
       --project-dir /scratch/simdq-fiqa \
       --encoder-dim 768 \
       --out simdq_recipe_sweep_fiqa.csv

``--qrels`` is optional — omit it when ``queries.jsonl`` carries its judgements
in a ``relevant`` key (see the note above).  ``--encoder-dim`` is the embedding
dimension D of your encoder.  It is
required for recipes that use ``simdq_d=HALF`` (R4 and R5), which halve
D before quantisation.  See :doc:`parameters` for all ``simdq_*``
fields.


New Encoder
-----------

To swap the encoder, change ``model_name`` in ``beir_simdq_base.yaml``
(or pass it as an override) to any HuggingFace model ID:

.. code-block:: yaml

   search_engine:
     model_name: sentence-transformers/all-mpnet-base-v2

**Supported embedding dimensions**

The current SimdQ kernel is optimised for D ∈ {384, 768, 1024, 1536}.
If your encoder produces a different dimension you have two options:

1. **Project to the nearest supported D in caller code.**
   Wrap the encoder with a small linear projection layer that maps its
   output to one of the four supported sizes before handing vectors to
   SimdQ.  A minimal example:

   .. code-block:: python

      import torch, torch.nn as nn

      class ProjectingEncoder(nn.Module):
          def __init__(self, base_encoder, in_dim: int, out_dim: int):
              super().__init__()
              self.encoder = base_encoder
              self.proj = nn.Linear(in_dim, out_dim, bias=False)

          def forward(self, *args, **kwargs):
              emb = self.encoder(*args, **kwargs)
              return self.proj(emb)

   Then pass ``--encoder-dim <out_dim>`` to the bench script.

2. **Extend the kernel template (advanced).**
   Adding a new D requires touching the low-level SIMD code.  This is
   out of scope for v1; see the simdq design spec §6 for guidance (the
   spec lives at
   ``docs/superpowers/specs/2026-06-18-simdq-test-and-docs-design.md``).


Custom Corpus Format
--------------------

If your corpus does not already use BEIR field names (``id``, ``text``,
``title``), point DocUVerse at a *data-template* config that maps your
actual field names to the expected ones.  Ready-made templates in
``config/`` include:

* ``beir_data_format.yml`` — standard BEIR layout (the default)
* ``clapnq_beir_data_format.yml`` — ClapNQ BEIR-compatible layout
* ``ibm_search_beir_data.yml`` — IBM internal search benchmark layout

Copy one of these files, adjust the field-mapping section, and reference
it in your config (``retrieval.data_template: config/my_format.yml``).
You can also override individual header names directly on
``RetrievalArguments`` without writing a template file:

.. code-block:: yaml

   retrieval:
     text_header:  body          # your field for the main passage text
     title_header: headline      # your field for the title (optional)
     id_header:    doc_id        # your field for the unique passage ID

See :doc:`parameters` for the full list of ``RetrievalArguments`` fields.


Plugging Into the Recipe Sweep
------------------------------

``scripts/bench_simdq_beir.py`` iterates over a module-level ``RECIPES``
list of ``(recipe_id, label, simdq_overrides)`` tuples.  For each entry
the script deep-copies the base YAML, injects the data-path CLI flags,
and merges ``simdq_overrides`` into ``search_engine``.  The sentinel
``"HALF"`` in ``simdq_d`` resolves to ``encoder_dim // 2`` at run
time.

**Adding a new recipe**

Append a new tuple to the ``RECIPES`` list in
``scripts/bench_simdq_beir.py``:

.. code-block:: python

   RECIPES = [
       # … existing R0–R5 entries …
       ("R7", "asym b=8, d=D/4, random_orthogonal",
        {"simdq_family": "asymmetric", "simdq_b": 8,
         "simdq_projection": "random_orthogonal",
         "simdq_d": "QUARTER",          # extend _materialize_yaml if needed
         "simdq_store_floats": False,
         "simdq_rescore_alpha": 1}),
   ]

Then rerun the bench script with your updated source.  See :doc:`tuning`
for guidance on choosing ``b``, ``d``, and ``rescore_alpha`` values.

**Running only a subset of recipes**

The ``--recipes`` flag accepts one or more recipe IDs:

.. code-block:: bash

   python scripts/bench_simdq_beir.py \
       --dataset fiqa \
       --base-yaml config/beir_simdq_base.yaml \
       --passages data/fiqa/passages.jsonl \
       --queries  data/fiqa/queries.jsonl \
       --qrels    data/fiqa/qrels.tsv \
       --project-dir /scratch/simdq-fiqa \
       --encoder-dim 768 \
       --recipes R0 R3 \
       --out simdq_recipe_sweep_fiqa_r0r3.csv

(As above, ``--qrels`` may be dropped when the queries file has inline
``relevant`` keys.)

Recipes not listed are silently skipped; if no recipe IDs match the
filter the script prints a warning and writes no CSV.
