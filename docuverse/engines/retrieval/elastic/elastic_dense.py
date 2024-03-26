from .elastic import ElasticEngine
from docuverse.utils import get_param, DenseEmbeddingFunction


class ElasticDenseEngine(ElasticEngine):
    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)
        self.vector_field_name = get_param(kwargs, 'vector_field_name', 'vector')
        self.model_on_server = get_param(kwargs, 'model_on_server', False)
        if not self.model_on_server:
            self.model = DenseEmbeddingFunction(self.model_name)
        self.normalize_embs = get_param(kwargs, 'normalize_embs', False)

    def create_query(self, text, **kwargs):
        _knn = None
        _query = None
        _rank = None
        _knn = {
            "field": self.vector_field_name,
            "k": get_param(kwargs, 'top_k'),
            "num_candidates": get_param('num_candidates', 1000),
        }
        _query = {"bool": {
            "must": {
                "multi_match": {
                    "query": text,
                    "fields": [self.text_field, self.title_field]
                }
            }
        }}
        if self.model_on_server:
            _knn["query_vector_builder"] = {
                "text_embedding": {
                    "model_id": self.model_name,
                    "model_text": text
                }
            }
        else:
            _knn['query_vector'] = self.model.encode(text, self.normalize_embs)

        return _query, _knn, _rank
