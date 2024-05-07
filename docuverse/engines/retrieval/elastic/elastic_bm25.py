from typing import Tuple, Dict, List, Any
from docuverse.engines.retrieval.elastic import ElasticEngine


class ElasticBM25Engine(ElasticEngine):
    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)

    def create_query(self, text, **kwargs) -> tuple[
        dict[str, dict[str, dict[str, dict[str, list[Any] | Any]]]], Any, Any]:
        return {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": text,
                        "fields": [self.text_field, self.title_field]
                    }
                }
            }
        }, None, None