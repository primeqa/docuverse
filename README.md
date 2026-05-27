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

[//]: # (![Build Status]&#40;https://github.com/primeqa/primeqa/actions/workflows/primeqa-ci.yml/badge.svg&#41;)

[//]: # ([![LICENSE|Apache2.0]&#40;https://img.shields.io/github/license/saltstack/salt?color=blue&#41;]&#40;https://www.apache.org/licenses/LICENSE-2.0.txt&#41;)

[//]: # ([![sphinx-doc-build]&#40;https://github.com/primeqa/primeqa/actions/workflows/sphinx-doc-build.yml/badge.svg&#41;]&#40;https://github.com/primeqa/primeqa/actions/workflows/sphinx-doc-build.yml&#41;   )

DocUServe is a public open source repository that enables researchers and developers to quickly
experiment with various search engines (such as ElasticSearch, ChromaDB, Milvus, PrimeQA, FAISS)
both in direct search and reranking scenarios. By using DocUVerse, a researcher
can replicate the experiments outlined in a paper published in the latest NLP 
conference while also enjoying the capability to download pre-trained models 
(from an online repository) and run them on their own custom data. DocUVerse is built 
on top of the [Transformers](https://github.com/huggingface/transformers), PrimeQA, and Elasticsearch toolkits and uses [datasets](https://huggingface.co/datasets/viewer/) and 
[models](https://huggingface.co/PrimeQA) that are directly 
downloadable.

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

## 🔭 Learn more (not yet working)

| Section                                                                                     | Description                                                |
|---------------------------------------------------------------------------------------------|------------------------------------------------------------|
| 📒 [Documentation](https://primeqa.github.io/primeqa)                                       | Start API documentation and tutorials                      |
| 📓 [Tutorials: Jupyter Notebooks](https://github.com/primeqa/docuverse/tree/main/notebooks) | Notebooks to get started on QA tasks                       |
| 🤗 [Model sharing and uploading](https://huggingface.co/docs/transformers/model_sharing)    | Upload and share your fine-tuned models with the community |
| ✅ [Pull Request](https://primeqa.github.io/docuverse/pull_request_template.html)            | PrimeQA Pull Request                                       |
| 📄 [Generate Documentation](https://primeqa.github.io/primeqa/README.html)                  | How Documentation works                                    |        

## ❤️ DocUVerse collaborators include: Sara Rosenthal, Parul Awasthy, Scott McCarley, Jatin Ganhotra, and Radu Florian.       
