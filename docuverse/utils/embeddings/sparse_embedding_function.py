import os

from torch.nn import Embedding
from transformers import AutoTokenizer
import torch
import numpy as np
from typing import Union, List

from docuverse.utils import get_param, get_config_dir
from docuverse.utils.embeddings.embedding_function import EmbeddingFunction
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from collections import Counter
import tqdm


class SpladeSentenceTransformer:
    def __init__(self, model_name_or_path, device:str= 'cpu', max_terms=500):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = self.model.cuda()
        self.model = self.model.bfloat16()
        self.model.eval()
        self.max_terms = max_terms

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size=-1, **kwargs):
        # input_dict=self.tokenizer(sentences, max_length=512,padding='max_length',return_tensors='pt')
        input_dict = self.tokenizer(sentences, max_length=512, padding=True, return_tensors='pt', truncation=True)

        num_toks = (input_dict['input_ids'] != 1).sum(dim=1)
        input_dict['input_ids'] = input_dict['input_ids'].cuda()
        input_dict['attention_mask'] = input_dict['attention_mask'].cuda()
        if 'token_type_ids' in input_dict:
            input_dict['token_type_ids'] = input_dict['token_type_ids'].cuda()
        outputs = self.model(**input_dict)  # , return_dict=True)

        hidden_state = outputs[0]
        attention_mask = input_dict['attention_mask']  # (bs,seqlen)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).type(hidden_state.dtype)

        maxarg = torch.log(1.0 + torch.relu(hidden_state))  # bs * seqlen * voc
        maxdim1 = torch.max(maxarg * input_mask_expanded, dim=1).values  # bs * voc
        s1 = torch.sort(maxdim1, descending=True, dim=1)  # values = bs * voc,  indices = bs * voc

        lengths = attention_mask.sum(dim=1)
        expansions = []
        for idoc, length in enumerate(lengths):
            expanded_toks = self.tokenizer.convert_ids_to_tokens(s1.indices[idoc, 0:self.max_terms])
            expanded_weights = s1.values[idoc, 0:self.max_terms]
            expansion = [(t, float(w)) for n, (t, w) in enumerate(zip(expanded_toks, expanded_weights)) if w > 0]
            expansions.append(expansion)

        return expansions, num_toks

    def make_vector_from_expansions(expansion, scale):
        eps = 1.0 / scale
        vec = Counter({t: int(scale * w) for n, (t, w) in enumerate(expansion)})  # if w>eps } )
        return {t: w for t, w in vec.most_common()}


class SparseEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_or_directory_name, batch_size=128, **kwargs):
        super().__init__(model_or_directory_name, model_or_directory_name=model_or_directory_name, **kwargs)
        self.model = None
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # "mps" if torch.backends.mps.is_available() else 'cpu'
        if device == 'cpu':
            print(f"You are using {device}. This is much slower than using "
                  "a CUDA-enabled GPU. If on Colab you can change this by "
                  "clicking Runtime > Change runtime type > GPU.")
            self.num_devices = 0
        else:
            self.num_devices = torch.cuda.device_count()
        dmf_loaded = False
        if get_param(kwargs, 'from_dmf', None) is not None:
            model_or_directory_name = self.pull_from_dmf(model_or_directory_name)
            dmf_loaded = True

        # from sentence_transformers import SentenceTransformer
        try:
            self.create_model(model_or_directory_name=model_or_directory_name, device=device)
        except Exception as e:
            # Try once more, from dmf
            if not dmf_loaded:
                model_or_directory_name = self.pull_from_dmf(model_or_directory_name)
                self.create_model(model_or_directory_name=model_or_directory_name, device=device)
            else:
                print(f"Model not found: {model_or_directory_name}")
                raise RuntimeError(f"Model not found: {model_or_directory_name}")
        print('=== done initializing model')

    def create_model(self, model_or_directory_name:str=None, device:str="cpu"):
        self.model = SpladeSentenceTransformer(model_or_directory_name, device)

    def __call__(self, texts: Union[List[str], str], **kwargs) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        return self.encode(texts)

    def encode(self, texts: Union[str, List[str]], _batch_size: int = -1, show_progress_bar=None, **kwargs) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        embs = []
        if _batch_size == -1:
            _batch_size = self.batch_size
        if show_progress_bar is None:
            show_progress_bar = isinstance(texts, str) or max(len(texts), _batch_size) <= 1

        res, num_tokens = self.model.encode(texts, show_progress_bar=show_progress_bar, **kwargs)
        return res
