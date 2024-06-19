import copy
import json
from typing import Union, Tuple, List

import yaml

from docuverse.engines.search_queries import SearchQueries
from docuverse.engines import SearchData, RetrievalEngine

try:
    from elasticsearch import Elasticsearch
except:
    print(f"You need to install elasticsearch to be using ElasticSearch functionality!")
    raise RuntimeError("fYou need to install elasticsearch to be using ElasticSearch functionality!")
from docuverse.engines.search_result import SearchResult
from docuverse.engines.search_engine_config_params import SearchEngineConfig
from docuverse.utils import get_param

import os
from dotenv import load_dotenv


class ElasticServers:
    def __init__(self, config="../../../../config/elastic_servers.json"):
        self.servers = {}
        if os.path.exists(config):
            if config.endswith(".json"):
                self.servers = json.load(open(config))
            elif config.endswith(".yaml"):
                self.servers = yaml.safe_load(open(config))

    def get(self, name: str):
        return self.servers.get(name, (None, None, None))


class ElasticEngine(RetrievalEngine):
    es_servers = ElasticServers("config/elastic_servers.json")
    languages = ['en', 'es', 'fr', 'pt', 'ja', 'de']
    default_all_keys_to_index = ['title', 'id', 'url', 'productId',  # 'versionId',
                         'filePath', 'deliverableLoio', 'text',
                         'app_name', 'courseGrainedProductId']

    @staticmethod
    def read_config(config: str):
        es_servers = ElasticServers(config)

    def __init__(self, config_params, **kwargs):
        # super().__init__(**kwargs)
        self.pipeline_name = None
        self.rouge_duplicate_threshold = 0.7
        self.index_name = None
        self.filters = None
        self.duplicate_removal = None
        self.coga_mappings = {}
        self.settings = {}
        self._read_mappings("config/elastic_config.json")
        self.config = None
        self._init_config(config_params)
        self.source_excludes = []
        self.pipeline = None
        self.client = None
        if 'all_keys_to_index' in kwargs:
            self.all_keys_to_index = kwargs['all_keys_to_index']
        else:
            self.all_keys_to_index = self.default_all_keys_to_index


    def _init_connection(self):
        self._init_connection_info(self.config.get('server'))
        self._init_client()
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
            server_info = self.es_servers.get(server.lower())
            if server_info is None:
                raise RuntimeError(f"ElasticSearch server {server} not found.")
            for key, val in server_info.items():
                setattr(self, key, val)
            # self.host, self.api_key, self.ssl_fingerprint = \
            #     [server_info.get(key) for key in ['host', 'api_key', 'ssl_fingerprint']]

    def _init_config(self, config_params: Union[dict, SearchEngineConfig]):
        if isinstance(config_params, dict):
            config_params = SearchEngineConfig(config=config_params)

        # Elastic doesn't accept _ -> convert them to dashes.
        if config_params.index_name:
            config_params.index_name = config_params.index_name.replace("_", "-")
        PARAM_NAMES = ["index_name", "title_field", "text_field", "n_docs", "filters", "duplicate_removal",
                       "rouge_duplicate_threshold"]

        for param_name in PARAM_NAMES:
            setattr(self, param_name, get_param(config_params, param_name))

        self.config = config_params

    def _init_client(self):
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
            _ = self.client.info()
        except Exception as e:
            print(f"Error: {e}")
            import sys
            sys.exit(101)

    def info(self):
        return f"Elasticsearch client.info(): \n{self.client.info().body}"

    def search(self, question: SearchQueries.Query, **kwargs) -> SearchResult:
        query, knn, rank = self.create_query(question['text'], **kwargs)
        if self.filters:
            for filter in self.filters:
                print(filter)
                query = self.add_filter(query, type=self.filters[filter]['type'],
                                        field=self.filters[filter]['field'],
                                        terms=self.filters[filter]['terms'])

        res = self.client.search(
            index=self.index_name,
            knn=knn,
            query=query,
            rank=rank,
            # TODO - top_k and n_docs is the same argument or am I missing something?
            size=self.config.top_k,
            # TODO - This should be specified in the retrieval config too
            source_excludes=['vector', 'ml.predicted_value', 'ml.tokens']
        )

        result = SearchResult(question=question, data=self.read_results(res))
        result.remove_duplicates(self.duplicate_removal,
                                 self.rouge_duplicate_threshold)
        return result

    # Read specific to engine, e.g. Elasticsearch results
    def read_results(self, data):
        results = []
        for d in data['hits']['hits']:
            r = SearchResult.SearchDatum(score=d['_score'],data=d['_source'])
            results.append(r)
        return results

    def create_query(self, text: str, **kwargs) -> Tuple[dict[str:str], dict[str:str], dict[str:str]]:
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

    def check_index_rebuild(self):
        index_name = self.config.index_name
        import sys
        while True:
            r = input(
                f"Are you sure you want to recreate the index {index_name}? It might take a long time!! Say 'yes' or 'no':").strip()
            if r == 'no':
                print("OK - exiting. Run with '--actions r'")
                sys.exit(0)
            elif r == 'yes':
                break
            else:
                print(f"Please type 'yes' or 'no', not {r}!")

    def create_update_index(self, do_update=True):
        if self.client.indices.exists(index=self.config.index_name):
            if not do_update:
                self.check_index_rebuild()
                self.client.options(ignore_status=[400, 404]).indices.delete(index=self.config.index_name)
            else:
                print(f"Using existent index {self.config.index_name}.")
        else:
            if do_update:
                print("You are trying to update an index that does not exist "
                      "- will ignore your command and create the index.")
        if not self.client.indices.exists(index=self.config.index_name):
            self.client.indices.create(index=self.config.index_name,
                                       mappings=self.coga_mappings[self.config.lang],
                                       settings=self.settings[self.config.lang])

    def ingest(self, corpus: SearchData, **kwargs):
        from tqdm import tqdm
        from elasticsearch.helpers import bulk
        self._init_client() # Redo the connection to the server
        bulk_batch = self.config.get('bulk_batch', 40)
        num_passages = len(corpus)
        # print(input_passages[0].keys())
        keys_to_index = self._get_keys_to_index(corpus)
        actions = []
        update=kwargs.get('update', False)
        self.create_update_index(do_update=update)
        t = tqdm(total=num_passages, desc="Ingesting dense documents: ", smoothing=0.05)
        for k in range(0, num_passages, bulk_batch):
            actions = [
                {
                    "_index": self.config.index_name,
                    "_id": row['id'],
                    "_source": {k: row[k] for k in keys_to_index}
                }
                for pi, row in enumerate(corpus[k:min(k + bulk_batch, num_passages)])
            ]
            self.add_fields(actions, bulk_batch, corpus, k, num_passages)
            try:
                bulk(self.client, actions=actions, pipeline=self.pipeline_name)
            except Exception as e:
                print(f"Got an error in indexing: {e}")
            t.update(bulk_batch)
        t.close()
        if len(actions) > 0:
            try:
                bulk(client=self.client, actions=actions, pipeline=self.pipeline_name)
            except Exception as e:
                print(f"Got an error in indexing: {e}, {len(actions)}")

    def add_fields(self, actions:List[dict], bulk_batch:int, corpus:SearchData, k:int, num_passages:int):
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

    def add_filter(self, query, type: str = "filter", field: str = "productId", terms: list = None):
        query["bool"][type] = {"terms": {field: [term for term in terms]}}
        return query

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
