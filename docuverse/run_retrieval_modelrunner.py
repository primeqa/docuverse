from engines import SearchEngine, SearchQueries
import json
from engines.preprocessors import ModelRunnerPreprocessor

# Test an existing engine
filepath = "/dccstor/creme_brulee/ibm_datasets/processed/dev/en/askhr_74.jsonl"
engine = SearchEngine(config_path="experiments/askhr/setup.yaml")
print(engine.get_retriever_info())
queries = SearchQueries(ModelRunnerPreprocessor(filepaths=[filepath]))

output_file = "results.jsonl"

results = []
for query in queries:
    result = engine.retriever.search(query)
    results.append(result)
print(f"{len(results)} results")

# add back to input file and dump as new output
output = []
index = 0
k = 10
with open(filepath, mode="r", encoding='utf-8') as fp:
    for line in fp:
        line = json.loads(line)
        line['contexts'] = []
        for result in results[index].results[:k]:
            results
            line['contexts'].append({'document_id': result.id, 'text': result.text, 'title': result.title, 'domain': result.productId})
        index += 1
        output.append(line)

with open(output_file, mode="w", encoding='utf-8') as fp:
    for example in output:
        fp.write(json.dumps(example) + "\n")

# scoring not implemented yet
# scores = engine.compute_score(queries, results)
# print (f"Results:\n{scores.to_string()}")