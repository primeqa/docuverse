import copy
import json
from typing import Union, Tuple, List

from elastic_transport import SerializationError

from docuverse.engines.retrieval.retrieval_servers import RetrievalServers
from docuverse.utils import get_config_dir
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines import SearchData, RetrievalEngine

try:
    from elasticsearch import Elasticsearch, RequestError, NotFoundError
except:
    print(f"You need to install elasticsearch to be using ElasticSearch functionality!")
    raise RuntimeError("fYou need to install elasticsearch to be using ElasticSearch functionality!")
from docuverse.engines.search_result import SearchResult
from docuverse.engines.search_engine_config_params import SearchEngineConfig
from docuverse.utils import get_param

import os
from dotenv import load_dotenv


class ElasticEngine(RetrievalEngine):
    """
    `ElasticEngine` class is a subclass of `RetrievalEngine` class and provides retrieval functionality
    using Elasticsearch.

    Attributes:
        - languages: A list of supported languages.
        - default_all_keys_to_index: A list of keys that are added to the stored documents

    Methods:
        - read_config(config: str): Reads and updates the ElasticSearch server configuration.
        - __init__(self, config_params, **kwargs): Initializes an instance of `ElasticEngine` class.
        - _init_connection(): Initializes the connection with ElasticSearch server.
        - _init_connection_info(server: str = None): Initializes the connection information of ElasticSearch server.
        - load_model_config(config_params: Union[dict, SearchEngineConfig]): Loads the model configuration.
        - init_client(): Initializes the ElasticSearch client.
        - info(): Returns the ElasticSearch client information.
        - search(question: SearchQueries.Query, **kwargs) -> SearchResult: Performs a search query on ElasticSearch.
        - read_results(data): Parses and returns the search results.
        - create_query(text: str, **kwargs) -> Tuple[dict[str:str], dict[str:str], dict[str:str]]: Creates a search query based on the given text.
        - _get_keys_to_index(input_passages): Returns the keys to be indexed from the input passages.
        - check_index_rebuild(): Asks for user confirmation to rebuild the index.
        - has_index(index_name): Checks if the given index exists.
        - create_update_index(do_update:bool=True) -> bool: Creates or updates the ElasticSearch index.
        - ingest(corpus: SearchData, **kwargs): Ingests the corpus into the ElasticSearch index.
    """
    # es_servers = RetrievalServers(config="config/elastic_servers.json")
    languages = ['en', 'es', 'fr', 'pt', 'ja', 'de']
    default_all_keys_to_index = ['title', 'id', 'url', 'productId',  # 'versionId',
                                 'filePath', 'deliverableLoio', 'text',
                                 'app_name', 'courseGrainedProductId']

    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)
        self.version = None
        self.pipeline_name = None
        self.rouge_duplicate_threshold = 0.7
        self.index_name = None
        self.filters = None
        self.duplicate_removal = None
        self.coga_mappings = {}
        self.settings = {}
        config = os.path.join(get_config_dir("elastic_config.json"), "elastic_config.json")
        self._read_mappings(config)
        self.config = None
        self.load_model_config(config_params)
        self.es_servers = RetrievalServers(config="config/elastic_servers.json")
        self.source_excludes = []
        self.pipeline = None
        self.client = None
        if 'all_keys_to_index' in kwargs:
            self.all_keys_to_index = kwargs['all_keys_to_index']
        else:
            self.all_keys_to_index = self.default_all_keys_to_index
        self.filters = self.config.filter_on

    def _init_connection(self):
        """
        Initializes the connection with the server.

        Parameters:
        - self (object): The object itself.

        Returns:
        - None
        """
        self._init_connection_info(self.config.get('server'))
        self.init_client()
        self._set_pipelines()

    def _init_connection_info(self, server: str = None):
        if server is None:
            load_dotenv()
            self.host = os.getenv('ES_HOST')
            self.password = os.getenv('ES_PASSWORD')
            self.user = os.getenv('ES_USER')
            self.api_key = os.getenv('ES_API_KEY')
            self.ssl_fingerprint = os.getenv('ES_SSL_FINGERPRINT')
        else:
            server_info = get_param(self.es_servers, server.lower(), None)
            if server_info is None:
                raise RuntimeError(f"ElasticSearch server {server} not found.")
            for key, val in vars(server_info).items():
                setattr(self, key, val)

    def load_model_config(self, config_params: Union[dict, SearchEngineConfig]):
        """

            Load model configuration.

            :param config_params: Configuration parameters for the model. Accepts either a dictionary or an
                                  instance of SearchEngineConfig.
            :type config_params: Union[dict, SearchEngineConfig]
            :return: None
        """
        super().load_model_config(config_params)

        # Elastic doesn't accept _ -> convert them to dashes.
        if config_params.index_name:
            config_params.index_name = config_params.index_name.replace("_", "-")

        props = self.coga_mappings[self.config.lang]['properties']
        if self.config.data_template.extra_fields is not None:
            for extra in self.config.data_template.extra_fields:
                if extra not in props:
                    props[extra] = {'type': 'keyword'}

    def init_client(self):
        """
        Initializes the Elasticsearch client, including the last accessed timer.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - None

        Example usage:
        init_client()
        """
        if self.api_key is not None:
            self.client = Elasticsearch(f"{self.host}",
                                        ssl_assert_fingerprint=self.ssl_fingerprint,
                                        api_key=self.api_key,
                                        request_timeout=60)
        elif self.password is not None:
            self.client = Elasticsearch(
                f"{self.host}",
                basic_auth=(self.user, self.password),
                ssl_assert_fingerprint=self.ssl_fingerprint
            )
        try:
            self.version = self.client.info()['version']['number']
        except Exception as e:
            print(f"Error: {e}")
            import sys
            sys.exit(101)
        self.set_accessed()

    def info(self):
        """
        Returns information about the Elasticsearch client (for display purposes and heartbeat check).

        :return: A string containing the information about the Elasticsearch client.
        """
        return f"Elasticsearch client.info(): \n{self.client.info().body}"

    def search(self, question: Union[dict, SearchQueries.Query], **kwargs) -> SearchResult:
        """

        Searches for relevant results based on the provided question, including duplicate removal.

        Parameters:
        - question: SearchQueries.Query - The question to search for.
        - **kwargs - Additional keyword arguments for creating the query.

        Returns:
        - SearchResult - The search result containing relevant information.

        """
        query, knn, rank = self.create_query(get_param(question, self.config.query_template.text_header), **kwargs)
        if self.filters:
            filters = self.create_filters(question)
            if 'bool' in query:
                query['bool'].update(**filters)
            elif 'sub-searches' in query:
                for q in query["sub-searches"]:
                    q['bool'].update(**filters)
            # for filter in self.filters:
            #     # print(filter)
            #     if query is None:
            #         query = {'bool': {}}
            #     vals = get_param(question, filter.query_field, None)
            #     if vals is not None:
            #         query = self.add_filter(query,
            #                                 type=filter.type,
            #                                 field=filter.document_field,
            #                                 terms=vals)
        index_name = self.index_name.replace("_","-")
        if 'sub_searches' in query:
            query.update(size=self.config.top_k,
                         _source={("excludes" if self.version>="8.15.0" else "exclude"): ["vector", "ml.predicted_value", "ml.tokens"]},
                         rank=rank)
            res = self.client.search(
                index=index_name,
                body=json.dumps(query),
            )
        else:
            res = self.client.search(
                index=index_name,
                knn=knn,
                query=query,
                rank=rank,
                size=self.config.top_k,
                source_excludes=['vector', 'ml.predicted_value', 'ml.tokens']
            )

        result = SearchResult(question=question, data=self.read_results(res))
        return result

    @staticmethod
    def add_filter(query, type: str = "filter", field: str = "productId", terms: str = None):
        query["bool"][type] = {"terms": {field: terms}}
        return query

    def create_filters(self, question):
        filter = {"filter": []}

        for f in self.filters:
            vals = get_param(question, f.query_field, None)
            if vals is not None:
                filter["filter"].append({f.query_field: vals})
        return filter

    # Read specific to engine, e.g. Elasticsearch results
    def read_results(self, data):
        """
        Reads the search results data into the SearchResult data structure list and returns it.

        Parameters:
            - data (dict): The search results data.

        Returns:
            - results (list): A list of SearchResult objects.

        """
        results = []
        for d in data['hits']['hits']:
            r = SearchResult.SearchDatum(score=d['_score'], data=d['_source'])
            results.append(r)
        return results

    def create_query(self, text: str, **kwargs) -> Tuple[dict[str:str], dict[str:str], dict[str:str]]:
        """
        Create Elasticsearch json query for given text and keyword arguments.

        Args:
            text (str): The text for which query needs to be created.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[dict[str:str], dict[str:str], dict[str:str]]: A tuple containing three dictionaries:
            tf-idf query , knn query, and query merge options.

        """
        pass
        # return super().create_query(text, kwargs)

    def _get_keys_to_index(self, input_passages):
        keys_to_index = []
        for k in self.all_keys_to_index:
            if k not in input_passages[0]:
                print(f"Dropping key {k} - they are not in the passages")
            else:
                keys_to_index.append(k)
        return keys_to_index

    def has_index(self, index_name, **kwargs):
        return self.client.indices.exists(index=index_name)

    def create_index(self, index_name: str, **kwargs):
        self.client.indices.create(index=index_name,
                                   mappings=self.coga_mappings[self.config.lang],
                                   settings=self.settings[self.config.lang])

    def delete_index(self, index_name: str|None = None, **kwargs):
        if index_name is None:
            index_name = self.config.index_name
        self.client.options(ignore_status=[400, 404]).indices.delete(index=self.config.index_name)



    def ingest(self, corpus: SearchData, **kwargs):
        """

        Method Name: ingest

        Parameters:
        - corpus: SearchData - The corpus of text documents to be ingested into the Elasticsearch index.
        - **kwargs - Additional keyword arguments that can be passed to the method.
                    The 'update' argument can be used to specify whether to update existing documents
                    in the index.

        Returns: None

        Description:
        This method is used to ingest a corpus of dense documents into the Elasticsearch index. It iterates over the corpus in batches and performs the bulk indexing operation using the Elasticsearch client. The method also supports updating existing documents in the index.

        Example:

        corpus = SearchData(...)
        kwargs = {'update': True}
        ingest(corpus, **kwargs)
        """
        from tqdm import tqdm
        from elasticsearch.helpers import bulk
        self.init_client()  # Redo the connection to the server
        bulk_batch = self.config.get('bulk_batch', 40)
        num_passages = len(corpus)
        # print(input_passages[0].keys())
        keys_to_index = self._get_keys_to_index(corpus)
        if self.config.data_template.extra_fields is not None:
            keys_to_index.extend(self.config.data_template.extra_fields)
        # for k in self.config.data_format.extra_fields:
        #     keys_to_index.append(k)
        actions = []
        update = kwargs.get('update', False)
        still_create_index = self.create_update_index(do_update=update)
        if not still_create_index:
            return
        t = tqdm(total=num_passages,
                 desc=f"Ingesting {self.config.db_engine.replace('es-','')} documents: ",
                 smoothing=0.05)
        for k in range(0, num_passages, bulk_batch):
            actions = [
                {
                    "_index": self.config.index_name,
                    "_id": row['id'],
                    "_source": {k: row[k] for k in keys_to_index if row[k] != ""}
                }
                for pi, row in enumerate(corpus[k:min(k + bulk_batch, num_passages)])
            ]
            self.add_fields(actions, bulk_batch, corpus, k, num_passages)
            try:
                bulk(self.client, actions=actions, pipeline=self.pipeline_name)
            except RequestError as e:
                if e.status_code == 400:
                    print("Bad request:", e.info)
                else:
                    print("Request error:", e)
            except NotFoundError as e:
                print("Index not found:", e)
            except SerializationError as e:
                print("Serialization error:", e)
            except Exception as e:
                print("Unexpected error:", e)
            t.update(bulk_batch)
        t.close()
        if len(actions) > 0:
            try:
                bulk(client=self.client, actions=actions, pipeline=self.pipeline_name)
            except RequestError as e:
                if e.status_code == 400:
                    print("Bad request:", e.info)
                else:
                    print("Request error:", e)
            except NotFoundError as e:
                print("Index not found:", e)
            except SerializationError as e:
                print("Serialization error:", e)
            except Exception as e:
                print("Unexpected error:", e)

    def add_fields(self, actions: List[dict], bulk_batch: int, corpus: SearchData, k: int, num_passages: int):
        """
        This function is used for adding fields to the indexed passage (e.g., the embedding vector
        for dense models where the model is not on the server.

        Parameters:
            actions (List[str]): List of actions to perform.
            bulk_batch (List[str]): List of bulk batches to process.
            corpus (str): The corpus to add fields to.
            k (int): The value of k.
            num_passages (int): The number of passages.

        Returns:
            None
        """
        pass

    def ingest_documents(self, documents, **kwargs):
        pass

    def _read_mappings(self, param_file):
        def union(a, b):
            if a == {}:
                return copy.deepcopy(b)
            else:
                c = copy.deepcopy(a)
                for k in b.keys():
                    if k in a:
                        c[k] = union(a[k], b[k])
                    else:
                        c[k] = copy.deepcopy(b[k])
            return c

        with open(param_file) as f:
            config = json.load(f)
            standard_mappings = config['settings']['standard']
            for lang in ElasticEngine.languages:
                self.settings[lang] = union(config['settings']['common'],
                                            config['settings'][lang if lang in config['settings'] else 'en']
                                            )
                self.coga_mappings[lang] = union(config['mappings']['common'],
                                                 config['mappings'][lang if lang in config['mappings'] else 'en']
                                                 )

    def _set_pipelines(self, **kwargs):
        pass
