import json
from typing import Union

# from onnx.reference.custom_element_types importp; bfloat16
from scipy.sparse import spmatrix
from tqdm import tqdm, trange

from docuverse import SearchCorpus
from docuverse.engines.retrieval.retrieval_servers import RetrievalServers, Server
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines import SearchData, RetrievalEngine
from docuverse.utils.timer import timer

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
        self.embeddings_name = "embeddings"
        self.milvus_defaults = read_config_file("config/milvus_default_config.yaml")
        self.load_model_config(config_params=config_params)
        self.servers = self.read_servers()
        self.server = None
        if get_param(self.config, 'server', None):
            if self.config.server.find("file:") >= 0:
                self.server = Server(host=self.config.server)
            else:
                self.server = self.servers[self.config.server]
        self.model = None
        self.init_model(**kwargs)
        # Milvus does not accept '-', only letters, numbers, and "_"
        self.init_client()
        self.output_fields = ["id", "text", 'title']
        # self.output_fields = [self.config.data_template.get(f"{t}_header", t) for t in ["id", "text", 'title']]
        extra = get_param(self.config.data_template, 'extra_fields', None)
        self.ingestion_batch_size = get_param(self.config, 'bulk_batch', 40)

        if extra is not None and len(extra) > 0:
            self.output_fields += extra

    def init_model(self, **kwargs):
        pass

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
        if self.config.index_name is not None:
            # It's possible for the index_name to be None if it's a hybrid model.
            self.config.index_name = self.config.index_name.replace("-", "_")
        elif not self.config.db_engine.endswith("hybrid"):
            raise RuntimeError("The index_name cannot be null for a non-hybrid Milvus configuration!")


    def init_client(self):
        #connections.connect("default", host=self.host, port=self.port)
        if self.server is None:
            print("MilvusEngine server is not initialized - it's OK if this in a hybrid combination!")
            # raise RuntimeError("MilvusEngine server is not initialized!")
        else:
            if self.server.type == "file":
                self.client = MilvusClient(uri=self.server.server_name)
            else:
                self.client = MilvusClient(uri=f"http://{self.server.host}:{self.server.port}",
                                           user=get_param(self.server, "user", ""),
                                           password=get_param(self.server, "password", ""),
                                           server_name=get_param(self.server, "server_name", ""),
                                           secure=get_param(self.server, "secure", False),
                                           server_pem_path = get_param(self.server, "server_pem_path", None)
                                           )

    def has_index(self, index_name: str):
        return self.client.has_collection(self.config.index_name) if self.client else False

    def create_index(self, index_name: str=None, fields=None, fmt=None, **kwargs):
        if index_name is None:
            index_name = self.config.index_name
        self.check_client()
        index_params = self._create_collection(fields, fmt)
        return index_params

    def _create_collection(self, fields, fmt):
        schema = CollectionSchema(fields, self.config.index_name)
        if fmt:
            print(fmt.format(f"Create collection `{self.config.index_name}`"))
        index_params = self.prepare_index_params()
        # self.index = Collection(self.config.index_name, schema, consistency_level="Strong", index_file_size=128)
        self.client.create_collection(self.config.index_name, schema=schema, index_params=index_params,
                                      consistency_level="Eventually")
        return index_params

    def check_client(self):
        if self.client is None:
            raise RuntimeError("MilvusEngine server is not defined/initialized.")

    def delete_index(self, index_name: str|None=None, fmt=None, **kwargs):
        self.check_client()
        if index_name is None:
            index_name = self.config.index_name
        if fmt:
            print(fmt.format(f"Collection {self.config.index_name} exists, dropping"))
        self.client.drop_collection(index_name)

    def _check_index_creation_and_get_text(self, corpus, update):
        self.check_client()
        fmt = "\n=== {:30} ==="
        fields = self.create_fields()

        still_create_index = self.create_update_index(fmt=fmt, update=update, fields=fields)
        if not still_create_index:
            return None

        texts = [row['text'] for row in corpus]
        return texts

    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs):
        texts = self._check_index_creation_and_get_text(corpus, update)

        self._analyze_data(texts)

        if texts is None:
            return False
        tq = tqdm(desc="Creating data", total=len(texts), leave=True)
        tq1 = tqdm(desc="  * Encoding data", total=len(texts), leave=False)
        tq2 = tqdm(desc="  * Milvusing data", total=len(texts), leave=False)
        ingestion_batch = 5*self.ingestion_batch_size
        for i in range(0, len(texts), ingestion_batch):
            last = min(i+ingestion_batch, len(texts))
            data = self._create_data(corpus[i:last], texts[i:last], tq_instance=tq1, **kwargs)
            self._insert_data(data, tq_instance=tq2)
            tq.update(last-i)
        return True

    def _analyze_data(self, corpus):
        pass

    def _insert_data(self, data, show_progress_bar=True, tq_instance=None, **kwargs):
        if tq_instance is None:
            tq_instance = tqdm(total=len(data), desc="Ingesting documents", leave=False, disable=not show_progress_bar)
        for i in range(0, len(data), self.ingestion_batch_size):
            last = min(i+self.ingestion_batch_size, len(data))
            self.client.insert(collection_name=self.config.index_name, data=data[i:last])
            tq_instance.update(last-i)
        # self.client.flush()
        self.wait_for_ingestion(data)
        # self.client.create_index(collection_name=self.config.index_name, index_params=self.prepare_index_params())

    def wait_for_ingestion(self, data):
        tm = timer()
        import time
        ingested_items = len(data)
        while ingested_items < len(data):
            res = self.client.get_collection_stats(collection_name=self.config.index_name)
            ingested_items = res["row_count"]
            print(f"{tm.time_since_beginning()}: Currently ingested items: {ingested_items}")
            time.sleep(10)

    def _create_data(self, corpus, texts, tq_instance=None, **kwargs):
        if tq_instance is not None:
            passage_vectors = self.encode_data(texts, self.ingestion_batch_size, show_progress_bar=False)
            tq_instance.update(len(texts))
        data = []
        for i, (item, vector) in enumerate(zip(corpus, passage_vectors)):
            # if isinstance(vector, spmatrix) and vector.getnnz() == 0:
            if getattr(vector, 'count_nonzero', None) is not None and vector.count_nonzero()==0:
                continue
            dt = {key: item[key] for key in ['text', 'title', 'id']}
            dt[self.embeddings_name] = vector
            for f in self.config.data_template.extra_fields:
                if isinstance(item[f], dict|list):
                    dt[f] = json.dumps(item[f])
                else:
                    dt[f] = str(item[f])
            data.append(dt)
        return data

    def encode_data(self, texts, batch_size, **kwargs):
        passage_vectors = self.model.encode(texts,
                                            _batch_size=batch_size,
                                            **kwargs)
        # create embeddings
        return passage_vectors

    def create_fields(self, embeddings_name="embeddings", new_fields_only=False):
        fields = [
            FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, description="ID", auto_id=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=False, max_length=1000),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20000),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=10000),
        ]
        if self.config.data_template.extra_fields is not None:
            for f in self.config.data_template.extra_fields:
                fields.append(FieldSchema(name=f, dtype=DataType.VARCHAR, max_length=10000))

        return fields

    def prepare_index_params(self, embeddings_name:str="embeddings"):
        pass

    def analyze(self, text):
        pass

    def search(self, question: SearchQueries.Query, **kwargs) -> SearchResult:
        self.check_client()
        search_params = self.get_search_params()
       # search_params['params']['group_by_field']='url'
        query_vector = self.encode_query(question)
        if getattr(query_vector, 'count_nonzero', None) is not None:
            non_zero = query_vector.count_nonzero()
            if non_zero == 0:
                print(f"Query \"{question.text}\" has 0 length representation.")
                return SearchResult(question=question, data=[])
        group_by = get_param(kwargs, 'group_by', None)
        extra = {}
        if group_by is not None:
            search_params['params']['group_by_field'] = group_by
            extra = {'group_by_field': group_by}

        res = self.client.search(
            collection_name=self.config.index_name,
            data=[query_vector],
            # search_params=search_params,
            limit=self.config.top_k,
            # limit=1000,
            output_fields=self.output_fields,
            **extra
        )
        result = SearchResult(question=question, data=res[0])
        result.remove_duplicates(self.config.duplicate_removal,
                                 self.config.rouge_duplicate_threshold)
        return result

    def encode_query(self, question):
        """
        Encodes a question text using the model's encode function. By default, it uses the usual encoding mechanism,
        but it can be overridden if necessary.

        Args:
            question (Question): The question object containing the text to be encoded.

        Returns:
            list: The encoded representation of the question text.
        """
        return self.model.encode(question.text, show_progress_bar=False)

    def get_search_params(self):
        pass

    def get_search_request(self, text):
        """
        Creates a search request, based on the type of the engine - used in hybrid/multi-index search
        """
        pass