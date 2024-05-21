from docuverse.engines.retrieval.elastic import ElasticEngine

class ElasticElserEngine(ElasticEngine):
    def __init__(self, config_params, **kwargs):
        self.output_name = kwargs.get('elser_name', 'ml')
        self.output_feature_name = kwargs.get('elser_feature_name', 'tokens')
        self.elser_name = f'{self.output_name}.{self.output_feature_name}'
        super().__init__(config_params, **kwargs)

    def create_query(self, text, **kwargs):
        return {
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
        }, None, None

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
        self.client.ingest.put_pipeline(processors=processors, id=self.config.model_name + "-test")
