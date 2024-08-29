import math
from copy import deepcopy

import torch

from docuverse import SearchResult
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction
from .reranker import Reranker
from sentence_transformers import util as st_util
from docuverse.engines.search_engine_config_params import RerankerConfig as RerankerConfig
from ...utils.embeddings.sparse_embedding_function import SparseEmbeddingFunction
from ...utils.timer import timer


class SpladeReranker(Reranker):
    def __init__(self, reranking_config: RerankerConfig|dict, **kwargs):
        super().__init__(reranking_config, **kwargs)
        self.model = SparseEmbeddingFunction(reranking_config.reranker_model,
                                             batch_size=reranking_config.reranker_gpu_batch_size)

    def similarity(self, embedding1, embedding2):
#         norm1 = float(torch.linalg.norm(embedding1[:,1], ord=2))
#         norm2 = float(torch.linalg.norm(embedding2[:,1], ord=2))
        # print (norm1)
        tm=timer("reranking::cosine")
        device = self.model.device
        embedding1 = embedding1.to(device)
        vocab_size = self.model.vocab_size

        emb1 = torch.zeros(vocab_size, dtype=torch.bfloat16, device=device)
        emb1[embedding1[:, 0].int()] = embedding1[:, 1]
        tm.add_timing("copy")

        if isinstance(embedding2, list):
            num_vectors = len(embedding2)
            emb2 = torch.zeros((num_vectors,vocab_size), dtype=torch.bfloat16, device=device)
            for i in range(len(embedding2)):
                embedding2[i] = embedding2[i].to(device)
                emb2[i][embedding2[i][:, 0].int()] = embedding2[i][:, 1]
            tm.add_timing("copy")
            prod = torch.matmul(emb1.reshape(1, vocab_size), emb2.reshape(num_vectors, vocab_size, 1))
            tm.add_timing("dotproduct")
        else:
            embedding2 = embedding2.to(device)

            emb2 = torch.zeros(vocab_size, dtype=torch.bfloat16, device=device)
            emb2[embedding2[:,0].int()] = embedding2[:,1]

            prod =  torch.dot(emb1, emb2)

        return prod #/ (norm1*norm2) if norm1+norm2>1e-7 else 0

