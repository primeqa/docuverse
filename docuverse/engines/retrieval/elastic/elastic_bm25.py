from elastic import ElasticEngine

class ElasticBM25(ElasticEngine):
    def __init__(self, config, **kwargs):
        super.__init__(config)

    def query(self, text, **kwargs):