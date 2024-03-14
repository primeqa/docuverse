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
    <img width="350" alt="primeqa" src="docs/_static/img/PrimeQA.png">
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

## Design

The following is a code snippet showing how to run a query search, and also how to ingest a corpus,
followed by an evaluation search.
```python
from docuverse import SearchEngine, SearchCorpus, SearchQueries

# Test an existing engine
engine = SearchEngine(config="experiments/sap/elastic_v2/setup.yaml")
queries = SearchQueries(data="benchmark_v2.csv")

results = engine.search(queries)
scores = engine.compute_score(queries, results)
print (f"Results:\n{scores.to_string()}")
```

Ingesting a new corpus (create an index for a specific engine) should be just as easy:
```python
from docuverse import SearchEngine, SearchCorpus, SearchQueries

corpus = SearchCorpus(filepaths="experiments/claspnq/passages.jsonl")
engine.ingest(corpus, max_doc_length=512, stride=100, title_handling="all", 
              index="my_new_index")

queries = SearchQueries(data="ClaspNQ.jsonl")
scores = engine.compute_score(queries, results)
print (f"Results:\n{scores.to_string()}")
```

## ✔️ Getting Started

### Installation
[Installation doc](https://primeqa.github.io/primeqa/installation.html)       

```shell
# cd to project root

# If you want to run on GPU make sure to install torch appropriately

# E.g. for torch 1.11 + CUDA 11.3:
pip install 'torch~=1.11.0' --extra-index-url https://download.pytorch.org/whl/cu113

# Install as editable (-e) or non-editable using pip, with extras (e.g. tests) as desired
# Example installation commands:

# Minimal install (non-editable)
pip install .

# GPU support
pip install .[gpu]

# Full install (editable)
pip install -e .[all]
```

Please note that dependencies (specified in [setup.py](./setup.py)) are pinned to provide a stable experience.
When installing from source these can be modified, however this is not officially supported.

**Note:** in many environments, conda-forge based faiss libraries perform substantially better than the default ones installed with pip. To install faiss libraries from conda-forge, use the following steps:

- Create and activate a conda environment
- Install faiss libraries, using a command

```conda install -c conda-forge faiss=1.7.0 faiss-gpu=1.7.0```

- In `setup.py`, remove the faiss-related lines:

```commandline
"faiss-cpu~=1.7.2": ["install", "gpu"],
"faiss-gpu~=1.7.2": ["gpu"],
```

- Continue with the `pip install` commands as desctibed above.

## :speech_balloon: Blog Posts
There're several blog posts by members of the open source community on how they've been using PrimeQA for their needs. Read some of them:
1. [PrimeQA and GPT 3](https://www.marktechpost.com/2023/03/03/with-just-20-lines-of-python-code-you-can-do-retrieval-augmented-gpt-based-qa-using-this-open-source-repository-called-primeqa/)
2. [Enterprise search with PrimeQA](https://heidloff.net/article/introduction-neural-information-retrieval/)
3. [A search engine for Trivia geeks](https://www.deleeuw.me.uk/posts/Using-PrimeQA-For-NLP-Question-Answering/)


## 🧪 Unit Tests
[Testing doc](https://primeqa.github.io/primeqa/testing.html)       

To run the unit tests you first need to [install PrimeQA](#Installation).
Make sure to install with the `[tests]` or `[all]` extras from pip.

From there you can run the tests via pytest, for example:
```shell
pytest --cov PrimeQA --cov-config .coveragerc tests/
```

For more information, see:
- Our [tox.ini](./tox.ini)
- The [pytest](https://docs.pytest.org) and [tox](https://tox.wiki/en/latest/) documentation    

## 🔭 Learn more

| Section | Description |
|-|-|
| 📒 [Documentation](https://primeqa.github.io/primeqa) | Full API documentation and tutorials |
| 📓 [Tutorials: Jupyter Notebooks](https://github.com/primeqa/primeqa/tree/main/notebooks) | Notebooks to get started on QA tasks |
| 🤗 [Model sharing and uploading](https://huggingface.co/docs/transformers/model_sharing) | Upload and share your fine-tuned models with the community |
| ✅ [Pull Request](https://primeqa.github.io/primeqa/pull_request_template.html) | PrimeQA Pull Request |
| 📄 [Generate Documentation](https://primeqa.github.io/primeqa/README.html) | How Documentation works |        

## ❤️ PrimeQA collaborators include       
