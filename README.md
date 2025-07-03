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

The following is a code snippet showing how to ingesting a new corpus (create an index for a specific engine), 
read the query file, run the search, compute the results and print them:
```python
from docuverse import SearchEngine
engine = SearchEngine(config_or_path="data/clapnq_small/milvus-test.yaml")

# Read the ClapNQ dataset
data = engine.read_data() # or engine.read_data(engine.config.input_passages)
#Ingest the data
engine.ingest(data)

# Read the queries
queries = engine.read_questions() # or engine.read_questions(engine.config.input_queries)
# Run the retrieval
results = engine.search(queries)
# Evaluation and print the results
scores = engine.compute_score(queries, results)

# Print the evaluation results in a human-readable format.
print(f"Results:\n{scores}")
```

## ✔️ Getting Started

### Installation
[Installation doc](https://primeqa.github.io/primeqa/installation.html)       

```shell
# cd to project root

# If you want to run on GPU make sure to install torch appropriately

# Install as editable (-e) or non-editable using pip, with extras (e.g. tests) as desired
# Example installation commands:

# Minimal install (non-editable)
pip install .

# Full install (editable)
pip install -e .

# Install milvus and/or elastic dependencies, and the pyizumo library (if you have acecess to it)
pip install -r requirements-milvus.txt
pip install -r requirements-elastic.txt
pip install -r requirements_extra.txt
```

Please note that dependencies (specified in [setup.py](./setup.py)) are pinned to provide a stable experience.
When installing from source these can be modified, however this is not officially supported.

## 🔭 Learn more (not yet working)

| Section                                                                                     | Description                                                |
|---------------------------------------------------------------------------------------------|------------------------------------------------------------|
| 📒 [Documentation](https://primeqa.github.io/primeqa)                                       | Start API documentation and tutorials                      |
| 📓 [Tutorials: Jupyter Notebooks](https://github.com/primeqa/docuverse/tree/main/notebooks) | Notebooks to get started on QA tasks                       |
| 🤗 [Model sharing and uploading](https://huggingface.co/docs/transformers/model_sharing)    | Upload and share your fine-tuned models with the community |
| ✅ [Pull Request](https://primeqa.github.io/docuverse/pull_request_template.html)            | PrimeQA Pull Request                                       |
| 📄 [Generate Documentation](https://primeqa.github.io/primeqa/README.html)                  | How Documentation works                                    |        

## ❤️ DocUVerse collaborators include: Sara Rosenthal, Parul Awasthy, Scott McCarley, Jatin Ganhotra, and Radu Florian.       
