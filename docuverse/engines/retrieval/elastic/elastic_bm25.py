from typing import Tuple, Dict
from docuverse.engines.retrieval.elastic import ElasticEngine


class ElasticBM25Engine(ElasticEngine):
    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)

    def create_query(self, text, **kwargs) -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
        return {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": text,
                        "fields": [self.text_field, self.title_field]
                    }
                },
            }
        }, \
            None, None