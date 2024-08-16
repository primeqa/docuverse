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
from docuverse.utils import get_param, read_config_file
from docuverse.utils.embedding_function import DenseEmbeddingFunction

import os
from dotenv import load_dotenv


class MilvusEngine(RetrievalEngine):
    """MilvusEngine class

    Inherits from RetrievalEngine

    MilvusEngine class provides functionality to interact with the Milvus database for indexing and searching.
    It supports ingestion of text data, creation of collection and index, and searching based on query vectors.

    Methods:
        read_servers: Read and return retrieval servers configuration from a JSON file.
        __init__: Initialize a MilvusEngine object.
        load_model_config: Load model configuration.
        init_client: Initialize a connection to Milvus server.
        ingest: Ingest text data into Milvus collection.
        search: Search Milvus collection based on query vectors.

    """
    @staticmethod
    def read_servers(config: str = "config/milvus_servers.json"):
        return RetrievalServers(config="config/milvus_servers.json")


    def __init__(self, config_params: SearchEngineConfig | dict, **kwargs):
        """
        Initializes the MilvusEngine class.

        Parameters:
        - config_params: A dictionary containing configuration parameters for the search engine.
                         Alternatively, an instance of SearchEngineConfig class can be provided.
        - **kwargs: Additional keyword arguments that can be passed to the parent class constructor.

        Returns:
        None
        """
        super().__init__(config_params=config_params, **kwargs)
        self.client = None
        self.index = None
        self.milvus_defaults = read_config_file("config/milvus_default_config.yaml")
        self.load_model_config(config_params=config_params)
        self.servers = self.read_servers()
        self.server = None
        if get_param(self.config, 'server', None):
            self.server = self.servers[self.config.server]
            # self.host = server.host
            # self.port = server.port
        self.model = DenseEmbeddingFunction(self.config.model_name)
        self.hidden_dim = len(self.model.encode('text', show_progress_bar=False))
        self.normalize_embs = get_param(kwargs, 'normalize_embs', False)

        # Milvus does not accept '-', only letters, numbers, and "_"
        self.init_client()
        # self.output_fields = ["id", "text", 'title']
        self.output_fields = [self.config.data_template.get(f"{t}_header", t) for t in ["id", "text", 'title']]
        extra = get_param(self.config.data_template, 'extra_fields', None)

        if extra is not None and len(extra) > 0:
            self.output_fields += extra

    def load_model_config(self, config_params: Union[dict, SearchEngineConfig]):
        """
        Loads the  model configuration parameters.

        Parameters:
            config_params (Union[dict, SearchEngineConfig]): The configuration parameters for the model.
                This can be either a dictionary or an instance of the SearchEngineConfig class.

        Returns:
            None

        """
        super().load_model_config(config_params)
        self.config.index_name = self.config.index_name.replace("-", "_")

    def init_client(self):
        #connections.connect("default", host=self.host, port=self.port)
        if self.server is None:
            print("MilvusEngine client is not initialized!")
            raise RuntimeError("MilvusEngine client is not initialized!")
        self.client = MilvusClient(uri=f"http://{self.server.host}:{self.server.port}",
                                   user=get_param(self.server, "user", ""),
                                   password=get_param(self.server, "password", ""),
                                   server_name=get_param(self.server, "server_name", ""),
                                   secure=get_param(self.server, "secure", False),
                                   server_pem_path = get_param(self.server, "server_pem_path", None)
                                   )

    def ingest(self, corpus: SearchCorpus, update: bool = False):
        fmt = "\n=== {:30} ==="
        fields = [
            FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, description="ID", auto_id=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=False, max_length=1000, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.hidden_dim)
        ]
        for f in self.config.data_template.extra_fields:
            fields.append(FieldSchema(name=f, dtype=DataType.VARCHAR, max_length=10000))
        # collection_loaded = utility.has_collection(self.config.index_name)
        collection_loaded = self.client.has_collection(self.config.index_name)
        if collection_loaded:
            print(fmt.format(f"Collection {self.config.index_name} exists, dropping"))
            self.client.drop_collection(self.config.index_name)
            # utility.drop_collection(self.config.index_name)
        schema = CollectionSchema(fields, self.config.index_name)
        print(fmt.format(f"Create collection `{self.config.index_name}`"))

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
            dt = {"id": item["id"], "embeddings": passage_vectors[i], "text": item['text'], 'title': item['title']}
            for f in self.config.data_template.extra_fields:
                dt[f] = str(item[f])
            data.append(dt)

        for i in tqdm(range(0, len(data), batch_size), desc="Ingesting documents"):
            self.client.insert(collection_name=self.config.index_name, data=data[i:i + batch_size])
        self.client.create_index(collection_name=self.config.index_name, index_params=index_params)

    def search(self, question: SearchQueries.Query, **kwargs) -> SearchResult:
        search_params = get_param(self.config, 'search_params',
                                  self.milvus_defaults['search_params']["HNSW"])
       # search_params['params']['group_by_field']='url'

        query_vector = self.model.encode(question.text, show_progress_bar=False)

        res = self.client.search(
            collection_name=self.config.index_name,
            data=[query_vector],
            # "embeddings",
            search_params=search_params,
            limit=self.config.top_k,
            # limit=1000,
            # group_by_field="url",
            output_fields=self.output_fields
        )
        result = SearchResult(question=question, data=res[0])
        result.remove_duplicates(self.config.duplicate_removal,
                                 self.config.rouge_duplicate_threshold)
        return result
