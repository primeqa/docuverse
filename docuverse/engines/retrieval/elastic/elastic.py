try:
    from elasticsearch import Elasticsearch
except:
    print(f"You need to install elasticsearch to be using ElasticSearch functionality!")
    raise RuntimeError("fYou need to install elasticsearch to be using ElasticSearch functionality!")
from docuverse.engines.search_result import SearchResult
from docuverse.engines.retrieval_engine import RetrievalEngine
from docuverse.utils import get_param


class ElasticEngine(RetrievalEngine):
    def __init__(self, config_params, **kwargs):
        super().__init__(**kwargs)
        self.host = config_params['connection']['host']
        if 'password' in config_params['connection']:
            self.password = config_params['connection']['password']
            self.user = config_params['connection']['user']
            self.client = Elasticsearch(basic_auth=f"{self.user}:{self.password}", hosts=[self.host])
        elif 'es_api_key' in config_params['connection']:
            self.api_key = config_params['connection']['es_api_key']
            self.ssl_fingerprint = config_params['connection']['es_ssl_fingerprint']
            self.client = Elasticsearch(f"{self.host}",
                                        ssl_assert_fingerprint=(self.ssl_fingerprint),
                                        api_key=self.api_key,
                                        request_timeout=60)
        self.index_name = config_params['index']
        self.title_field = get_param(config_params['connection'],'title_field','title')
        self.text_field = get_param(config_params['connection'],'text_field','text')
        # self.fields = config_params['fields']
        self.config = config_params
        # self.filter_fields = config_params['filter-fields']

        try:
            _ = self.client.info()
        except Exception as e:
            print(f"Error: {e}")
            raise e

    def search(self, text, **kwargs) -> SearchResult:
        query, knn, rank = self.create_query(text, **kwargs)
        if 'filter' in self.config:
            for filter in self.config['filter']:
                query = self.add_filter(query, type=self.config['filter'][filter]['type'], 
                                        field=self.config['filter'][filter]['field'], 
                                        terms=self.config['filter'][filter]['terms'])

        res = self.client.search(
            index=self.index_name,
            knn=knn,
            query=query,
            rank=rank,
            size=get_param(self.config, 'top_k', 10),
            source_excludes=['vector', 'ml.predicted_value']
        )
        result = SearchResult(results=self.read_results(res))

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

    def add_filter(self, query, type: str = "filter", field: str = "productId", terms: list = None):
        query["bool"][type] = {"terms": {field: [term for term in terms]}}
        return query
    
    def ingest_documents(self, documents, **kwargs):
        pass
