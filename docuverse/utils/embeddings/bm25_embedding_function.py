import os

from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import numpy as np
from typing import Union, List

from docuverse.utils import get_param, get_config_dir
from docuverse.utils.embeddings.embedding_function import EmbeddingFunction
try:
    from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
    from pymilvus.model.sparse import BM25EmbeddingFunction
except:
    print(f"You need to install pymilvus to be using Milvus functionality!"
          f" Run `pip install -r requirements-milvus.txt` from the top directory")
    raise RuntimeError("fYou need to install pymilvus to be using Milvus functionality!")


class BM25EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_or_directory_name, batch_size=128, **kwargs):
        super().__init__(model_or_directory_name=model_or_directory_name, batch_size=batch_size, **kwargs)
        self.analyzer = build_default_analyzer(get_param(kwargs, 'language', "en"))
        print('=== done BM25 initializing model')

    def __call__(self, texts: Union[List[str], str], **kwargs) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        return self.encode(texts)

    def create_model(self, model_or_directory_name: str=None, device: str='cpu'):
        from sentence_transformers import SentenceTransformer

    @property
    def tokenizer(self):
        return None

    def encode(self, texts: Union[str, List[str]], _batch_size: int = -1, show_progress_bar=None, **kwargs) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        embs = []
        if _batch_size == -1:
            _batch_size = self.batch_size
        if show_progress_bar is None:
            show_progress_bar = not (isinstance(texts, str) or max(len(texts), _batch_size) <= 1)

        if isinstance(texts, str):
            embs = self.analyzer(texts)
        elif isinstance(texts, list):
            embs = [self.analyzer(text) for text in tqdm(texts, desc='Tokenizing texts', disable=not show_progress_bar)]

        return embs
