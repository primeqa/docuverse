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
        self.model = SparseEmbeddingFunction(reranking_config.reranker_model,
                                             batch_size=reranking_config.reranker_gpu_batch_size)

    def similarity(self, embedding1, embedding2):
        emb1, emb2 = (embedding1, embedding2) if len(embedding1) > len(embedding2) else (embedding2, embedding1)
        keys = {k:v for (k,v) in emb1}


        norm1 = np.linalg.norm([e[1] for e in emb1], ord=2)
        norm2 = np.linalg.norm([e[1] for e in emb2], ord=2)

        val = sum(keys[key]*value for (key, value) in emb2 if key in keys)
        return val / (norm1*norm2) if norm1+norm2>1e-7 else 0
