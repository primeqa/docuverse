import json
import yaml

try:
    from elasticsearch import Elasticsearch
except:
    print(f"You need to install elasticsearch to be using ElasticSearch functionality!")
    raise RuntimeError("fYou need to install elasticsearch to be using ElasticSearch functionality!")
from docuverse.engines.search_result import SearchResult
from docuverse.engines.search_engine_config_params import SearchEngineConfig

import os
from dotenv import load_dotenv


class ElasticServers:
    def __init__(self, config="servers.json"):
        self.servers = {}
        if os.path.exists(config):
            if config.endswith(".json"):
                self.servers = json.load(open(config))
            elif config.endswith(".yaml"):
                self.servers = yaml.safe_load(open(config))

    def get(self, name: str):
        return self.servers.get(name, (None, None, None))


class ElasticEngine:
    es_servers = ElasticServers("servers.json")

    @staticmethod
    def read_config(config: str):
        es_servers = ElasticServers(config)

    def __init__(self, config_params, **kwargs):
        super().__init__(**kwargs)
        self._init_connection_info(config_params.get('server'))
        self._init_config(config_params)
        self._init_client()

    def _init_connection_info(self, server:str=None):
        if server is None:
            load_dotenv()
            self.host = os.getenv('ES_HOST')
            self.password = os.getenv('ES_PASSWORD')
            self.user = os.getenv('ES_USER')
            self.api_key = os.getenv('ES_API_KEY')
            self.ssl_fingerprint = os.getenv('ES_SSL_FINGERPRINT')
        else:
            server_info = self.es_servers.get(server)
            if server_info is None:
                raise RuntimeError(f"ElasticSearch server {server} not found.")
            self.host, self.api_key, self.ssl_fingerprint = \
                [server_info.get(key) for key in ['host', 'api_key', 'ssl_fingerprint']]

    def _init_config(self, config_params):
        if isinstance(config_params, dict):
            config_params = SearchEngineConfig(config=config_params)

        self.index_name = config_params.index_name
        self.title_field = config_params.title_field
        self.text_field = config_params.text_field
        self.fields = config_params.fields
        self.n_docs = config_params.n_docs
        self.config = config_params
        self.filters = config_params.filters
        self.duplicate_removal = config_params.duplicate_removal
        self.rouge_duplicate_threshold = config_params.rouge_duplicate_threshold

    def _init_client(self):
        if self.password is not None:
            self.client = Elasticsearch(
                f"{self.host}:9200",
                basic_auth=(self.user, self.password),
                ssl_assert_fingerprint=self.ssl_fingerprint
            )
        elif self.api_key is not None:
            self.client = Elasticsearch(f"{self.host}",
                                        ssl_assert_fingerprint=self.ssl_fingerprint,
                                        api_key=self.api_key,
                                        request_timeout=60)
        try:
            _ = self.client.info()
        except Exception as e:
            print(f"Error: {e}")
            raise e

    def info(self):
        return f"Elasticsearch client.info(): \n{self.client.info().body}"

    def search(self, text, **kwargs) -> SearchResult:
        query, knn, rank = self.create_query(text, **kwargs)
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
            size=self.n_docs,
            # TODO - This should be specified in the retrieval config too
            source_excludes=['vector', 'ml.predicted_value']
        )

        result = SearchResult(data=self.read_results(res))
        result.remove_duplicates(self.duplicate_removal,
                                 self.rouge_duplicate_threshold)
        return result

    # Read specific to engine, e.g. Elasticsearch results
    def read_results(self, data):
        results = []
        for d in data['hits']['hits']:
            r = SearchResult.SearchDatum(d['_source'])
            results.append(r)
        return results

    def create_query(self, text, **kwargs):
        return super().create_query(text, kwargs)

    def ingest(self, corpus, **kwargs):
        from tqdm import tqdm
        for index, record in tqdm(enumerate(corpus)):
            resp = self.client.index(index=self.index_name, id=1, document=record)
            if index > 10:
                return None

            # TODO: Find out the exact format of the records to be indexed
            # TODO: Move to bulk API
            # keys_to_index = self.fields
            # actions = [
            #     {
            #         "_index": self.index_name,
            #         "_id": record['id'],
            #         "_source": {k: record[k] for k in keys_to_index}
            #     }
            # ]
            # try:
            #     response = bulk(client=self.client, actions=actions)
            # except Exception as e:
            #     print(f"Got an error in indexing: {e}, {len(actions)}")

    def add_filter(self, query, type: str = "filter", field: str = "productId", terms: list = None):
        query["bool"][type] = {"terms": {field: [term for term in terms]}}
        return query

    def ingest_documents(self, documents, **kwargs):
        pass
