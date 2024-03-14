from docuverse.engines.retrieval.elastic import ElasticEngine

class ElasticDenseEngine(ElasticEngine):
    def __init__(self, config, **kwargs):
        super.__init__(config)

    def create_query(self, text, **kwargs):
        return ""