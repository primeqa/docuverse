from docuverse.engines.retrieval.elastic import ElasticEngine

class ElasticElserEngine(ElasticEngine):
    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)

    def create_query(self, text, **kwargs):
        return {
            "bool": {
                "must": {
                    "text_expansion": {
                        "ml.tokens": {
                            "model_id": ".elser_model_1",
                            "model_text": text
                        }
                    }
                }
            }
        }, None, None