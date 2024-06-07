class Reranker(object):
    def __init__(self, reranking_config, **kwargs):
        self.config = reranking_config
        self.name = reranking_config['name']

    def rerank(self, documents, combine_score=None):
        return None