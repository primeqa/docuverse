import copy
import json
from typing import Union, Tuple, List

import yaml
from tqdm import tqdm

from docuverse import SearchEngine, SearchCorpus
from docuverse.engines.search_engine import RetrievalServers
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines import SearchData, RetrievalEngine

try:
    from pymilvus import (
        connections,
        utility,
        FieldSchema, CollectionSchema, DataType,
        Collection,
    )
except:
    print(f"You need to install pymilvus to be using Milvus functionality!")
    raise RuntimeError("fYou need to install pymilvus to be using Milvus functionality!")
from docuverse.engines.search_result import SearchResult
from docuverse.engines.search_engine_config_params import SearchEngineConfig
from docuverse.utils import get_param, DenseEmbeddingFunction, read_config_file

import os
from dotenv import load_dotenv

class MilvusEngine(SearchEngine):
    ms_servers = RetrievalServers(config="config/milvus_servers.json")

    def __init__(self, config: SearchEngineConfig, **kwargs):
        self.index = None
        self.milvus_defaults = read_config_file("../../../../config/milvus_defaults.yml")
        self.config = self.load_model_config(config)
        if 'server' in self.config:
            server = MilvusEngine.ms_servers[self.config['server']]
            self.host = server['host']
            self.port = server['port']
        self.model = DenseEmbeddingFunction(self.config.model_name)
        self.hidden_dim = len(self.model.encode('text', show_progress_bar=False))
        self.normalize_embs = get_param(kwargs, 'normalize_embs', False)
        self.init_client()

    def init_client(self):
        connections.connect("default", host=self.host, port=self.port)

    def ingest(self, corpus: SearchCorpus, update: bool = False):
        fmt = "\n=== {:30} ===\n"
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.hidden_dim)
        ]
        collection_loaded = utility.has_collection(self.config.index_name)
        if collection_loaded:
            print(fmt.format(f"Collection {self.config.index_name} exists, dropping"))
            utility.drop_collection(self.config.index_name)

        schema = CollectionSchema(fields, "Test for speed.")
        print(fmt.format(f"Create collection `f{self.config.index_name}`"))
        self.index = Collection(self.config.index_name, schema, consistency_level="Strong", index_file_size=128)

        ids = [int(row.id) for row in corpus]
        # create metadata batch
        text_vectors = [row.text for row in corpus]
        batch_size = get_param(self.config, 'bulk_batch', 40)
        passage_vectors = self.model.encode(text_vectors,
                                            _batch_size=batch_size,
                                            show_progress_bar=True)
        # create embeddings
        batch_size = 1000
        for i in tqdm(range(0, len(text_vectors), batch_size), desc="Milvus index docs:"):
            self.index.insert([ids[i:i + batch_size],
                              text_vectors[i:i + batch_size],
                              passage_vectors[i:i + batch_size]]
                            )
        # index = {"index_type": "IVF_FLAT",
        #          "metric_type": "IP",
        #          "params": {"nlist": 128}}

        index_params = get_param(self.config, 'index_params', self.milvus_defaults['default_index_params'])
        self.index.create_index("embeddings",
                              index_params)
        self.index.load()

    def search(self, question: SearchQueries.Query, **kwargs) -> SearchResult:
        search_params = get_param(self.config, 'search_params', self.milvus_defaults['search_params']["HNSW"])

                # for query_number in range(len(query_vectors)):
        query_vector = self.model.encode(question.text)

        res = self.index.search(
            [query_vector],
            "embeddings",
            search_params,
            limit=self.config.top_k,
            output_fields=["id", "text"]
        )[0]
        result = SearchResult(question=question, result=res)
        result.remove_duplicates(self.config.duplicate_removal,
                                 self.config.rouge_duplicate_threshold)
        return result