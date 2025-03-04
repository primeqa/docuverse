import math
from copy import deepcopy

import torch
from .reranker import Reranker
from docuverse.engines.search_engine_config_params import RerankerConfig as RerankerConfig
from ...utils.embeddings.sparse_embedding_function import SparseEmbeddingFunction
from ...utils.timer import timer


class SpladeReranker(Reranker):
    def __init__(self, reranking_config: RerankerConfig|dict, **kwargs):
        super().__init__(reranking_config, **kwargs)
        self.model = SparseEmbeddingFunction(reranking_config.reranker_model,
                                             batch_size=reranking_config.reranker_gpu_batch_size,
                                             process_name="reranking")

    def pair_similarity(self, pair, device='cpu'):
        return self.similarity(pair[0], pair[1], device)

    def similarity(self, embedding1, embedding2, device='cuda'):
        def make_zeros(arg, dtype=float):
            if device == 'cuda':
                return torch.zeros(arg, dtype=torch.bfloat16, device=device)
            else:
                return torch.zeros(arg, dtype=dtype, device=device)
        tm=timer("reranking::cosine")
        # device = self.model.device
        if not isinstance(embedding1, torch.Tensor):
            embedding1 = torch.tensor(embedding1)
            ee2 = []
            for e in embedding2:
                ee2.append(torch.tensor(e))
            embedding2 = ee2
        embedding1 = embedding1.to(device)
        vocab_size = self.model.vocab_size
        max1 = int(torch.max(embedding1[:, 0]))+1
        emb1 = make_zeros(max1, dtype=embedding1.dtype)
        emb1[embedding1[:, 0].int()] = embedding1[:, 1]
        tm.add_timing("sim::valcopy")

        if isinstance(embedding2, list):
            maxind =  max([int(torch.max(emb[:, 0])) for emb in embedding2])+1
            num_vectors = len(embedding2)
            emb2 = make_zeros((num_vectors, maxind), embedding2[0].dtype)
            for i in range(len(embedding2)):
                embedding2[i] = embedding2[i].to(device)
                emb2[i][embedding2[i][:, 0].int()] = embedding2[i][:, 1]
            tm.add_timing("sim::valcopy")
            final_max = min(max1, maxind)
            prod = torch.matmul(emb1[:final_max].reshape(1, final_max), emb2[:,:final_max].reshape(num_vectors, final_max, 1))
            tm.add_timing("sim::dotproduct")
        else:
            maxind = int(torch.max(torch.max(embedding1[:, 0]), torch.max(embedding2[:, 0])))
            embedding2 = embedding2.to(device)

            emb2 = make_zeros(maxind)
            emb2[embedding2[:,0].int()] = embedding2[:,1]
            tm.add_timing("sim::valcopy")

            prod =  torch.dot(emb1, emb2)
            tm.add_timing("sim::dotproduct")

        return prod #/ (norm1*norm2) if norm1+norm2>1e-7 else 0

