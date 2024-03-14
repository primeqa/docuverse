try:
    from elasticsearch import Elasticsearch
except:
    print(f"You need to install elasticsearch to be using ElasticSearch functionality!")
    raise RuntimeError("fYou need to install elasticsearch to be using ElasticSearch functionality!")
from docuverse.engines.search_result import SearchResult

class ElasticEngine:
    def __init__(self, config_params, **kwargs):
        self.host = config_params.connection.host
        if 'password' in config_params.connection:
            self.password = config_params.connection.password
            self.user = config_params.connection.user
        elif 'es_api_key' in config_params.connection:
            self.api_key = config_params.connection.es_api_key
            self.ssl_fingerprint = config_params.connection.es_ssl_fingerprint
        self.index_name = config_params.connection.index
        self.title_field = config_params.connection.title_field
        self.text_field = config_params.connection.text_field
        self.fields = config_params.fields
        self.config = config_params
        self.filter_fields = config_params['filter-fields']

        if self.password is not None:
            self.client = Elasticsearch(basic_auth=f"{self.user:self.password}", hosts=[self.host])
        elif self.api_key is not None:
            self.client = Elasticsearch(f"{self.host}",
                            ssl_assert_fingerprint=(self.ssl_fingerprint),
                            api_key=self.api_key,
                            request_timeout=60)
        try:
            _ = self.client.info()
        except Exception as e:
            print(f"Error: {e}")
            raise e

    def search(self, text, **kwargs) -> SearchResult:
        query, knn, rank = self.create_query(text, **kwargs)
        res = self.client.search(
            index=self.index_name,
            knn=knn,
            query=query,
            rank=rank,
            size=self.get_param(kwargs, 'top_k'),
            source_excludes=['vector', 'ml.predicted_value']
        )
        res = SearchResult(result=res)

        res = self.remove_duplicates(res._body['hits']['hits'],
                                self.duplicate_removal,
                                self.rouge_duplicate_threshold)
        result.append({'qid': qid, 'text': query_text,
                       "answers": extract_answers(res)})

    def _create_query(self, text="", **kwargs):
        pass

    def ingest_documents(self, documents, **kwargs):
        pass