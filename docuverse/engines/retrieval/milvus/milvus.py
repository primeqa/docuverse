from typing import Union

from tqdm import tqdm

from docuverse import SearchCorpus
from docuverse.engines.retrieval.retrieval_servers import RetrievalServers
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines import SearchData, RetrievalEngine

try:
    from pymilvus import (
        # connections,
        MilvusClient,
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


class MilvusEngine(RetrievalEngine):
    ms_servers = RetrievalServers(config="config/milvus_servers.json")

    @staticmethod
    def read_servers(config: str = "config/milvus_servers.json"):
        return RetrievalServers(config="config/milvus_servers.json")

    def __init__(self, config_params: SearchEngineConfig | dict, **kwargs):
        super().__init__(config_params=config_params, **kwargs)
        self.client = None
        self.index = None
        self.milvus_defaults = read_config_file("config/milvus_default_config.yaml")
        self.load_model_config(config_params=config_params)
        self.servers = self.read_servers()
        if get_param(self.config, 'server', None):
            server = self.servers[self.config.server]
            self.host = server.host
            self.port = server.port
        self.model = DenseEmbeddingFunction(self.config.model_name)
        self.hidden_dim = len(self.model.encode('text', show_progress_bar=False))
        self.normalize_embs = get_param(kwargs, 'normalize_embs', False)
        # Milvus does not accept '-', only letters, numbers, and "_"
        self.init_client()

    def load_model_config(self, config_params: Union[dict, SearchEngineConfig]):
        super().load_model_config(config_params)
        self.config.index_name = self.config.index_name.replace("-", "_")

    def init_client(self):
        #connections.connect("default", host=self.host, port=self.port)
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")

    def ingest(self, corpus: SearchCorpus, update: bool = False):
        fmt = "\n=== {:30} ===\n"
        fields = [
            FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, description="ID", auto_id=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=False, max_length=1000, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.hidden_dim)
        ]
        # collection_loaded = utility.has_collection(self.config.index_name)
        collection_loaded = self.client.has_collection(self.config.index_name)
        if collection_loaded:
            print(fmt.format(f"Collection {self.config.index_name} exists, dropping"))
            self.client.drop_collection(self.config.index_name)
            # utility.drop_collection(self.config.index_name)
        schema = CollectionSchema(fields, self.config.index_name)
        print(fmt.format(f"Create collection `f{self.config.index_name}`"))

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="_id",
            index_type="STL_SORT"
        )

        index_params.add_index(
            field_name="embeddings",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128}
        )

        # self.index = Collection(self.config.index_name, schema, consistency_level="Strong", index_file_size=128)
        self.client.create_collection(self.config.index_name, schema=schema, index_params=index_params)

        text_vectors = [row['text'] for row in corpus]
        batch_size = get_param(self.config, 'bulk_batch', 40)
        passage_vectors = self.model.encode(text_vectors,
                                            _batch_size=batch_size,
                                            show_progress_bar=True)
        # create embeddings
        batch_size = 1000
        data = []
        for i, item in enumerate(corpus):
            data.append({"id": item["id"], "embeddings": passage_vectors[i], "text": item['text']})
        # for i in tqdm(range(0, len(text_vectors)), desc="Milvus index docs:"):
        #     # self.client.insert(collection=self.config.index_name, data={})
        #     data.append({"id": ids[i], "embeddings": passage_vectors[i], "text": text_vectors[i]})
        # self.index.insert([ids[i:i + batch_size],
        #                   text_vectors[i:i + batch_size],
        #                   passage_vectors[i:i + batch_size]]
        #                 )
        for i in tqdm(range(0, len(data), batch_size), desc="Ingesting documents"):
            self.client.insert(collection_name=self.config.index_name, data=data[i:i + batch_size])
        # index = {"index_type": "IVF_FLAT",
        #          "metric_type": "IP",
        #          "params": {"nlist": 128}}

        # index_params = get_param(self.config, 'index_params', self.milvus_defaults['default_index_params'])
        self.client.create_index(collection_name=self.config.index_name, index_params=index_params)
        # self.index.load()

    def search(self, question: SearchQueries.Query, **kwargs) -> SearchResult:
        search_params = get_param(self.config, 'search_params', self.milvus_defaults['search_params']["HNSW"])

        # for query_number in range(len(query_vectors)):
        query_vector = self.model.encode(question.text)

        res = self.client.search(
            collection_name=self.config.index_name,
            data=[query_vector],
            # "embeddings",
            search_params=search_params,
            limit=self.config.top_k,
            output_fields=["id", "text"]
        )
        result = SearchResult(question=question, result=res)
        result.remove_duplicates(self.config.duplicate_removal,
                                 self.config.rouge_duplicate_threshold)
        return result
