from copy import deepcopy

from docuverse import SearchResult
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction
from .bi_encoder_reranker import BiEncoderReranker
from sentence_transformers import util as st_util
from docuverse.engines.search_engine_config_params import RerankerConfig as RerankerConfig


class DenseReranker(BiEncoderReranker):
    def __init__(self, reranking_config: RerankerConfig|dict, **kwargs):
        super().__init__(reranking_config, **kwargs)
        self.model = DenseEmbeddingFunction(reranking_config.reranker_model)

    def similarity(self, embedding1, embedding2):
        res = st_util.pytorch_cos_sim(embedding1, embedding2).detach()
        return res[0]
