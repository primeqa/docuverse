import math
from copy import deepcopy

import numpy as np
from torch.nn.functional import embedding

from docuverse import SearchResult
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction
from .reranker import Reranker
from sentence_transformers import util as st_util
from docuverse.engines.search_engine_config_params import RerankerConfig as RerankerConfig
from ...utils.embeddings.sparse_embedding_function import SparseEmbeddingFunction


class SpladeReranker(Reranker):
    def __init__(self, reranking_config: RerankerConfig|dict, **kwargs):
        super().__init__(reranking_config, **kwargs)
        self.model = SparseEmbeddingFunction(reranking_config.reranker_model)

    def similarity(self, embedding1, embedding2):
        emb1, emb2 = (embedding1, embedding2) if len(embedding1) > len(embedding2) else (embedding2, embedding1)
        keys = {k:v for (k,v) in emb1}


        norm1 = np.linalg.norm([e[1] for e in emb1], ord=2)
        norm2 = np.linalg.norm([e[1] for e in emb2], ord=2)

        val = sum(keys[key]*value for (key, value) in emb2 if key in keys)
        return val / (norm1*norm2) if norm1+norm2>1e-7 else 0


    # def rerank(self, documents: SearchResult | list[SearchResult]) \
    #         -> SearchResult | list[SearchResult]:
    #         # Computing RRF is as follows (taken from
    #         #                      https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)
    #         # score = 0.0
    #         # for q in queries:
    #         #     if d in result(q):
    #         #         score += 1.0 / ( k + rank( result(q), d ) )
    #         # return score
    #         #
    #         # # where
    #         # # k is a ranking constant
    #         # # q is a query in the set of queries
    #         # # d is a document in the result set of q
    #         # # result(q) is the result set of q
    #         # # rank( result(q), d ) is d's rank within the result(q) starting from 1
    #         k = len(documents)
    #         for i in range(len(documents)):
    #             hybrid_similarities[i] = 1.0/(k+i+1) + 1.0/(k+idx_to_rerank_idx[i]+1)
    #
    #     sorted_similarities = sorted(zip(documents, hybrid_similarities),
    #                                  key=lambda pair: pair[1], reverse=True)
    #
    #     output = SearchResult(documents.question, [])
    #     for doc, sim in sorted_similarities:
    #         doc1 = deepcopy(doc)
    #         doc1.score = sim
    #         output.append(doc1)
    #     return output