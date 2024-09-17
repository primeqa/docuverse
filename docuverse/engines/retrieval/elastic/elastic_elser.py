from docuverse.engines.retrieval.elastic import ElasticEngine

class ElasticElserEngine(ElasticEngine):
    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)
        self.output_name = kwargs.get('elser_name', 'ml')
        self.output_feature_name = kwargs.get('elser_feature_name', 'tokens')
        self.elser_name = f'{self.output_name}.{self.output_feature_name}'
        self._init_connection()

    def create_query(self, text, **kwargs):
        _knn = None
        _rank = None
        _query = {
            "bool": {
                "must": {
                    "text_expansion": {
                        self.elser_name: {
                            "model_id": self.config.model_name,
                            "model_text": text
                        }
                    }
                }
            }
        }
        if self.config.hybrid == "rrf":
            _query1 = {"bool": {
                "must": {
                    "multi_match": {
                        "query": text,
                        "fields": [self.config.text_field, self.config.title_field]
                    }
                }
            }}
            _query = {"sub_searches": [{"query":_query}, {"query":_query1}]}
            if self.version >= "8.15.0":
                _rank = {"rrf": {"rank_window_size": 200}}
            else:
                _rank = {"rrf": {"window_size": 200}}
        return _query, _knn, _rank

    def _set_pipelines(self, **kwargs):
        mappings = self.coga_mappings[self.config.lang]
        mappings['properties'][self.elser_name] = {"type": "rank_features"}
        processors = [
                {
                    "inference": {
                        "model_id": self.config.model_name,
                        "target_field": self.output_name,
                        "field_map": {
                            "text": "text_field"
                        },
                        "inference_config": {
                            "text_expansion": {
                                "results_field": self.output_feature_name
                            }
                        }
                    }}
            ]
        self.pipeline_name = self.config.model_name + "-test"
        self.client.ingest.put_pipeline(processors=processors, id=self.pipeline_name)
