from typing import Union

from docuverse import SearchResult
from docuverse.utils import DenseEmbeddingFunction
from reranker import Reranker

class DenseReranker(Reranker):
    def __init(self, reranking_config, **kwargs):
        super().__init__(reranking_config, **kwargs)
        self.model =  DenseEmbeddingFunction(reranking_config.reranker_model)

    def rerank(self, documents:SearchResult|list[SearchResult], combine_score=None) \
            -> SearchResult|list[SearchResult]:
        if isinstance(documents, list):
            return [self.rerank(d, combine_score=combine_score) for d in documents]

        output = SearchResult()
        texts = [datum.text for datum in documents]
        embeddings = self.model.encode(texts, _batch_size=self.config.reranker_batch_size)
        for datum in documents:
            text = datum.get_text()
            embedding = self.model.embed(text)
            datum["embedding"] = embedding
            output.append(datum)