<!---
Copyright 2022 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h3 align="center">
    <img width="350" alt="DocUVerse D⌕" src="docs/_static/img/DocUVerse.png">
    <p>Repository for (almost) *all* your document search needs.</p>
    <p>Part of the Prime Repository for State-of-the-Art Multilingual QuestionAnswering Research and Development.</p>
</h3>

[//]: # (![Build Status]&#40;https://github.com/primeqa/docuverse/actions/workflows/docuverse-ci.yml/badge.svg&#41;)

[//]: # ([![LICENSE|Apache2.0]&#40;https://img.shields.io/github/license/saltstack/salt?color=blue&#41;]&#40;https://www.apache.org/licenses/LICENSE-2.0.txt&#41;)

[//]: # ([![sphinx-doc-build]&#40;https://github.com/primeqa/docuverse/actions/workflows/sphinx-doc-build.yml/badge.svg&#41;]&#40;https://github.com/primeqa/docuverse/actions/workflows/sphinx-doc-build.yml&#41;   )

DocUVerse is a public open source repository that enables researchers and developers to quickly
experiment with various search engines (such as ElasticSearch, ChromaDB, Milvus, FAISS, LanceDB)
both in direct search and reranking scenarios. By using DocUVerse, a researcher
can replicate the experiments outlined in a paper published in the latest NLP
conference while also enjoying the capability to download pre-trained models
(from an online repository) and run them on their own custom data. DocUVerse is built
on top of the [Transformers](https://github.com/huggingface/transformers) and
[sentence-transformers](https://www.sbert.net/) toolkits and uses
[datasets](https://huggingface.co/datasets) and
[models](https://huggingface.co/models) that are directly
downloadable from the Hugging Face Hub.

## ✔️ Getting Started

### Install

The curated `quickstart` extra pulls in everything the README example needs
(Milvus-Lite + sentence-transformers); no external service required.

```shell
pip install -e .[quickstart]
```

Other extras: `elastic`, `chromadb`, `faiss`, `lancedb`, `extra` (pyizumo,
huggingface CLI), `dev` (pytest, ruff, mypy). Combine them as needed:
`pip install -e .[quickstart,elastic,dev]`.

### Run the quickstart — three equivalent surfaces

The repo ships a 10-passage / 5-query toy corpus under
[`examples/quickstart/`](examples/quickstart/). All three of the following
produce the same ranked results.

**Python — preset with overrides** (the headline API):

```python
from docuverse import SearchEngine

engine = SearchEngine.from_preset(
    "milvus-dense",
    model_name="ibm-granite/granite-embedding-small-english-r2",
    index_name="docuverse_quickstart",
    input_passages="examples/quickstart/passages.jsonl",
    input_queries="examples/quickstart/queries.jsonl",
    output_file="examples/quickstart/output.json",
)
engine.ingest(engine.read_data())
queries = engine.read_questions()
results = engine.search(queries)
print(engine.compute_score(queries, results))
```

**Python — explicit YAML** (when you want the config in version control):

```python
from docuverse import SearchEngine

engine = SearchEngine(config_or_path="examples/quickstart/recipe.yaml")
engine.ingest(engine.read_data())
queries = engine.read_questions()
print(engine.compute_score(queries, engine.search(queries)))
```

**CLI**:

```bash
docuverse run --config examples/quickstart/recipe.yaml
```

### In-memory documents and queries (ChromaDB)

When you don't want to write JSONL files at all — e.g. you already have a
list of strings in Python — wrap each string in a small dict and hand it
straight to `read_data` / `read_questions`. The example below uses the
`chromadb` preset, which runs a local persistent ChromaDB instance on disk
(no external service). Install with `pip install -e .[chromadb]`.

```python
from docuverse import SearchEngine

documents = [
    "Photosynthesis is the biological process by which green plants convert light into chemical energy stored in glucose.",
    "Mitochondria are membrane-bound organelles that generate most of the cell's ATP, the main currency of cellular energy.",
    "DNA is a double-helix polymer that carries the genetic instructions for the development and function of all known organisms.",
    "Newton's three laws of motion describe the relationship between a body and the forces acting on it; he formulated them in 1687.",
    "World War II was a global conflict from 1939 to 1945 between the Allies and the Axis powers.",
    "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
]

queries = [
    "How do plants make energy from sunlight?",
    "What organelle produces ATP in cells?",
    "Who discovered the laws of motion?",
    "What is the largest ocean on Earth?",
]

engine = SearchEngine.from_preset(
    "chromadb",
    index_name="docuverse_readme_demo",
    top_k=3,
)

passages = [{"id": f"d{i}", "text": text} for i, text in enumerate(documents)]
engine.ingest(engine.read_data(file=passages))

question_records = [{"id": f"q{i}", "text": q} for i, q in enumerate(queries)]
search_queries = engine.read_questions(file=question_records)
results = engine.search(search_queries)

for query, result in zip(search_queries, results):
    top = result[0]
    print(f"Q: {query.text}\n  → {top.text[:80]}...\n")
```

The pattern generalises: any `read_data` / `read_questions` call accepts a
list of dicts in place of a file path, so you can pull documents from a
database, an API, or a generator without staging them on disk first. Required
keys per record are `id` and `text`; `title` and any other fields are
preserved into the index.

### Discover presets

```bash
docuverse presets list --with-engine    # name + db_engine
docuverse presets show milvus-dense     # parsed config
docuverse presets dump milvus-dense > my-recipe.yaml   # copy-and-edit
```

In Python: `SearchEngine.list_presets()`.

### Configuration

DocUVerse looks for config files under `./config/<rel_path>` (with a
`./config/<basename>` legacy fallback that emits one `DeprecationWarning`),
plus operator-level `$DOCUVERSE_HOME/...` and per-user `~/.docuverse/...`
overrides. See [`config/README.md`](config/README.md) for the full
six-tier resolver and the categorized layout (`servers/`, `engines/`,
`recipes/`, `data_formats/`).

## 🔭 Learn more

| Section                                                                                     | Description                                                |
|---------------------------------------------------------------------------------------------|------------------------------------------------------------|
| 📒 [Documentation](https://primeqa.github.io/docuverse)                                     | DocUVerse API documentation and tutorials                  |
| 📓 [Tutorials: Jupyter Notebooks](https://github.com/primeqa/docuverse/tree/main/notebooks) | Notebooks to get started with retrieval and reranking      |
| 🤗 [Model sharing and uploading](https://huggingface.co/docs/transformers/model_sharing)    | Upload and share your fine-tuned models with the community |
| ✅ [Pull Request](https://primeqa.github.io/docuverse/pull_request_template.html)            | DocUVerse Pull Request template                            |
| 📄 [Generate Documentation](https://primeqa.github.io/docuverse/README.html)                | How the documentation is built                             |

## ❤️ DocUVerse collaborators include: Sara Rosenthal, Parul Awasthy, Scott McCarley, Jatin Ganhotra, and Radu Florian.       
