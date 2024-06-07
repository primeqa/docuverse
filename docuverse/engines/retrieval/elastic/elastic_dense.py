from typing import Union

from .elastic import ElasticEngine
from docuverse.utils import get_param, DenseEmbeddingFunction
from ...search_engine_config_params import SearchEngineConfig


class ElasticDenseEngine(ElasticEngine):
    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)
        self.vector_field_name = get_param(kwargs, 'vector_field_name', 'vector')
        self.model_on_server = get_param(kwargs, 'model_on_server', False)
        self.hidden_dim = 384
        if not self.model_on_server:
            self.model = DenseEmbeddingFunction(config_params.model_name)
        self.normalize_embs = get_param(kwargs, 'normalize_embs', False)
        self._init_connection()

    def _init_connection(self):
        self._init_connection_info(self.config.get('server'))
        self._init_client()

        if self.config.model_on_server:
            if 'ml' in self.client.__dict__:
                r = self.client.ml.get_trained_models(model_id=self.config.model_name)
                self.hidden_dim = r['trained_model_configs'][0]['inference_config']['text_embedding']['embedding_size']
            else:
                self.hidden_dim = 384  # Some default value, the system might crash if it's wrong.
        else:
            print("Encoding corpus documents:")
            self.hidden_dim = len(self.model.encode('text'))

        self._set_pipelines()

    def create_query(self, text, **kwargs):
        _knn = None
        _query = None
        _rank = None
        _knn = {
            "field": self.vector_field_name,
            "k": int(get_param(kwargs, 'top_k', self.config.top_k)),
            "num_candidates": int(get_param(kwargs, 'num_candidates', self.config.num_candidates)),
        }
        if self.config.hybrid == "rrf":
            _query = {"bool": {
                "must": {
                    "multi_match": {
                        "query": text,
                        "fields": [self.config.text_field, self.config.title_field]
                    }
                }
            }}
            _rank ={"rrf": {"window_size": 200}}
        if self.model_on_server:
            _knn["query_vector_builder"] = {
                "text_embedding": {
                    "model_id": self.config.model_name,
                    "model_text": text
                }
            }
        else:
            _knn['query_vector'] = self.model.encode(text, self.normalize_embs, show_progress_bar=False)

        return _query, _knn, _rank

    def _set_pipelines(self, **kwargs):
        mappings = self.coga_mappings[self.config.lang]
        processors = []
        if self.config.model_on_server:
            vector_field_name = "ml"
            pipeline_name = f"{self.config.model_name}-test"
            processors = [{
                "inference": {
                    "model_id": self.config.model_name,
                    "target_field": "ml",
                    "field_map": {
                        "text": "text_field"
                    }
                }
            }]
            on_failure = [{
                "set": {
                    "description": "Index document to 'failed-<index>'",
                    "field": "_index",
                    "value": "failed-{{{_index}}}"
                }
            },
                {
                    "set": {
                        "description": "Set error message",
                        "field": "ingest.failure",
                        "value": "{{_ingest.on_failure_message}}"
                    }
                }]
            vector_field_name = f"ml.predicted_value"
        else:
            vector_field_name = "vector"
            pipeline_name = None
            on_failure = None

        mappings['properties'][vector_field_name] = {
            "type": "dense_vector",
            "similarity": "cosine",
            "dims": self.hidden_dim,
            "index": "true"
        }
        self.pipeline_name = self.config.model_name + "-test"
        self.client.ingest.put_pipeline(processors=processors, id=self.pipeline_name)

    def add_fields(self, actions, bulk_batch, corpus, k, num_passages):
        if not self.config.model_on_server:
            passage_vectors = self.model.encode([d['text'] for d in corpus[k:k+bulk_batch]], show_progress_bar=False)
            for pi, (action, row) in enumerate(zip(actions, corpus[k:min(k + bulk_batch, num_passages)])):
                action["_source"]['vector'] = passage_vectors[pi]
