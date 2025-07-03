#%%
# !pip install -r ../requirements-milvus.txt
# !pip install ipywidgets
#%%
from docuverse import SearchEngine
import os
os.environ['DOCUVERSE_CONFIG_PATH'] = os.getcwd().replace("notebooks", "config")
#%%
os.environ['DOCUVERSE_CONFIG_PATH']
os.environ['TOKENIZER_PARALLELISM']="false"
#%% md
# ### Create the search engine
# This section initializes a SearchEngine instance and configures it using the provided configuration file or path.
# 
#%%
# Test an existing engine
engine = None
engine = SearchEngine(config_or_path="data/clapnq_small/milvus-test.yaml")
# try:
#     engine = SearchEngine(config_or_path="data/clapnq_small/milvus-test.yaml")
# except Exception as e:
#     print(f"Error: {e}")
#%% md
# ## Data Ingestion
# This cell reads and ingests the data into the SearchEngine. If the data has already been ingested, you can skip this step by typing `<enter>` or `'skip'`.
#%%
data = engine.read_data()
engine.ingest(data)
#%% md
# ## Search
# This cell searches the corpus using a SearchEngine instance. It retrieves results for the given queries and evaluates the performance based on specific scoring metrics.
#%%
queries = engine.read_questions()
results = engine.search(queries)
#%% md
# ### Evaluation
# Compute the evaluation scores for the search engine results and print them.
# The compute_score() method calculates various metrics such as precision, recall, and NDCG (Normalized Discounted Cumulative Gain) 
# to evaluate the performance of the search engine based on the queries and their corresponding results.
#%%

scores = engine.compute_score(queries, results)

# Print the evaluation results in a human-readable format.
print(f"Results:\n{scores}")