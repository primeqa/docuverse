import os

from click.core import batch
from torch.nn import Embedding
from transformers import AutoTokenizer
import torch
import numpy as np
from typing import Union, List, Dict

from triton.language.extra.cuda import num_threads

from docuverse.utils import get_param, get_config_dir
from docuverse.utils.embeddings.embedding_function import EmbeddingFunction
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from collections import Counter
from tqdm import tqdm
from docuverse.utils.timer import timer


class SpladeSentenceTransformer:
    def __init__(self, model_name_or_path, device:str= 'cpu', max_terms=500):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = device
        self.model.to(device)
        if device == "cuda":
            self.model = self.model.cuda()
            self.model = self.model.bfloat16()
        self.model.eval()
        self.max_terms = max_terms

    @torch.no_grad()
    def encode(self, sentences: List[str], _batch_size=-1, show_progress_bar=False, **kwargs):
        # input_dict=self.tokenizer(sentences, max_length=512,padding='max_length',return_tensors='pt')
        tm = timer("ingest_and_test::run::encode")
        expansions = []
        num_toks = 0
        num_sents = len(sentences)
        sorted_sents_inds = sorted(range(0, len(sentences)), key=lambda x: len(sentences[x]), reverse=True)
        sorted_sents = [sentences[i] for i in sorted_sents_inds]

        with tqdm(desc="Processed sents", disable=not show_progress_bar, total=num_sents, leave=False) as tk:
            for b in range(0, num_sents, _batch_size):
                this_batch_size = min(b+_batch_size, num_sents)-b
                input_dict = self.tokenizer(sorted_sents[b:b+this_batch_size],
                                            max_length=512, padding=True, return_tensors='pt', truncation=True)
                tm.add_timing("tokenizer")
                num_toks += (input_dict['input_ids'] != 1).sum().item()
                attention_mask = input_dict['attention_mask']  # (bs,seqlen)
                if self.device == "cuda":
                    input_dict['input_ids'] = input_dict['input_ids'].cuda()
                    input_dict['attention_mask'] = input_dict['attention_mask'].cuda()
                    if 'token_type_ids' in input_dict:
                        input_dict['token_type_ids'] = input_dict['token_type_ids'].cuda()
                tm.add_timing("copy_to_gpu")
                outputs = self.model(**input_dict)  # , return_dict=True)
                tm.add_timing("bert_encoding")
                hidden_state = outputs[0]#.cpu()
                tm.add_timing("copy_to_cpu")
                maxarg = torch.log(1.0 + torch.relu(hidden_state))
                tm.add_timing("relu")
                #hidden_state = hidden_state.cpu()

                input_mask_expanded = attention_mask.unsqueeze(-1).to(maxarg.device)# .expand(hidden_state.size()).type(hidden_state.dtype)

                 # bs * seqlen * voc
                maxdim1 = torch.max(maxarg * input_mask_expanded, dim=1).values  # bs * voc
                tm.add_timing("attention_mask_filter")
                s1 = torch.sort(maxdim1, descending=True, dim=1)  # values = bs * voc,  indices = bs * voc
                lengths = attention_mask.sum(dim=1)
                tm.add_timing("sort")
                sizes = torch.sum(s1.values>0, dim=1)
                tm.add_timing("compute_expansion_sizes")

                for idoc in range(this_batch_size):
                    expanded_toks = self.tokenizer.convert_ids_to_tokens(s1.indices[idoc, 0:sizes[idoc]])
                    tm.add_timing("expansion::convert_ids_to_tokens")
                    expanded_weights = s1.values[idoc, 0:sizes[idoc]]
                    tm.add_timing("expansion::get_weights")
                    expansion = [(t, float(w)) for t, w in zip(expanded_toks, expanded_weights)]
                    tm.add_timing("expansion::create_expansion")
                    expansions.append(expansion)
                    tm.add_timing("expansion::add_expansion")
                # tm.add_timing("expansion")
                tk.update(min(_batch_size, num_sents-b))
        #tk.close()

        tmp = [[]] * len(expansions)
        for i, e in enumerate(expansions):
            tmp[sorted_sents_inds[i]] = e
        return tmp, num_toks

    def make_vector_from_expansions(expansion, scale):
        eps = 1.0 / scale
        vec = Counter({t: int(scale * w) for n, (t, w) in enumerate(expansion)})  # if w>eps } )
        return {t: w for t, w in vec.most_common()}


class SparseEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_or_directory_name, batch_size=128, **kwargs):
        super().__init__(model_or_directory_name=model_or_directory_name, batch_size=batch_size, **kwargs)
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
            Union[Dict[str, float|int], List[Dict[str, float|int]]]:
        return self.encode(texts)

    def encode(self, texts: Union[str, List[str]], _batch_size: int = -1, show_progress_bar=None, **kwargs) -> \
            Union[Dict[str, float|int], List[Dict[str, float|int]]]:
        if _batch_size == -1:
            _batch_size = self.batch_size
        if show_progress_bar is None:
            show_progress_bar = not (isinstance(texts, str) or max(len(texts), _batch_size) <= 1)

        res, num_tokens = self.model.encode(texts, _batch_size=_batch_size,
                                            show_progress_bar=show_progress_bar, **kwargs)
        return res
