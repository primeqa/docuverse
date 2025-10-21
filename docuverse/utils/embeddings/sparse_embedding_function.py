import os

from click.core import batch
from scipy.sparse import csr_matrix
from torch.nn import Embedding
from transformers import AutoTokenizer
import torch
import numpy as np
from typing import Union, List, Dict

# from triton.language.extra.cuda import num_threads

from docuverse.utils import get_param, get_config_dir
from docuverse.utils.embeddings.embedding_function import EmbeddingFunction
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from collections import Counter
from tqdm import tqdm
from docuverse.utils.timer import timer


class SparseSentenceTransformer:
    def __init__(self, model_name_or_path, device:str= 'cpu', doc_max_tokens=None, query_max_tokens=None,
                 process_name="ingest_and_test::search", **kwargs):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = device
        self.model.to(device)
        if device == "cuda":
            self.model = self.model.cuda()
            self.model = self.model.bfloat16()
        self.model.eval()
        self.doc_max_tokens = doc_max_tokens
        self.query_max_tokens = query_max_tokens
        self.process_name = process_name

    @torch.no_grad()
    def encode(self, sentences: List[str], _batch_size=16, show_progress_bar=False,
               tqdm_instance=None,
               process_name=None,
               prompt_name=None,
               **kwargs):
        # input_dict=self.tokenizer(sentences, max_length=512,padding='max_length',return_tensors='pt')
        tm = timer(f"{process_name if process_name is not None else self.process_name}::encode", disable=False)
        expansions = []
        num_toks = 0
        max_num_expansion = 500
        num_sents = len(sentences)
        sorted_sents_inds = sorted(range(0, len(sentences)), key=lambda x: len(sentences[x]), reverse=True)
        tm.add_timing("sorting sentences")
        # sorted_sents = [sentences[i] for i in sorted_sents_inds]
        message = get_param(kwargs, 'message', "Processed candidates")
        max_terms = get_param(kwargs, 'max_terms', self.doc_max_tokens)
        if max_terms is None:
            max_terms = 1000000000 # all terms
        if tqdm_instance is not None:
            tk = tqdm_instance
        else:
            if show_progress_bar:
                tk = tqdm(desc=message, disable=not show_progress_bar, total=num_sents, leave=True, smoothing=1)
            else:
                tk = None

        encode_question = get_param(kwargs, 'encode_question', True)
        for b in range(0, num_sents, _batch_size):
            this_batch_size = min(b+_batch_size, num_sents)-b
            input_dict = self.tokenizer([sentences[sorted_sents_inds[i]] for i in range(b,b+this_batch_size)],
                                        max_length=512, padding=True, return_tensors='pt', truncation=True)
            tm.add_timing("tokenizer")
            if encode_question:
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
                hidden_state = outputs[0]
                tm.add_timing("copy_to_cpu")
                maxarg = torch.log(1.0 + torch.relu(hidden_state))
                tm.add_timing("relu")

                input_mask_expanded = attention_mask.unsqueeze(-1).to(maxarg.device)# .expand(hidden_state.size()).type(hidden_state.dtype)
                 # bs * seqlen * voc
                maxdim1 = torch.max(maxarg * input_mask_expanded, dim=1).values  # bs * voc
                tm.add_timing("attention_mask_filter")
                # get topk high weights

                max_size = maxdim1.shape[1]
                topk, indices = torch.topk(maxdim1, k=self.doc_max_tokens) # (weight - (bs * max_terms), index - (bs * max_terms))
                tm.add_timing("get_topk_weights")
                topk_n = topk.tolist()
                inds = indices.tolist()

                embeddings = [
                    csr_matrix((vals, (np.zeros(len(ind)), ind)), shape=(1, max_size))
                    for vals, ind in zip(topk_n, inds)
                ]
            else: # No query encoding, just map the tokens to 1.0
                max_size = self.tokenizer.vocab_size
                keys = [torch.unique(input_dict['input_ids'][i], sorted=True).tolist() for i in range(b,b+this_batch_size)]

                embeddings = [csr_matrix((np.ones(len(k)), (np.zeros(len(k), dtype=int), k)), shape=(1, max_size))
                              for k in keys]
            tm.add_timing("expansion::create_expansion")
            expansions.extend(embeddings)
            tm.add_timing("expansion::add_expansion")
            if tk:
                tk.update(min(_batch_size, num_sents - b))
        unsorted_expansions = [[]] * len(expansions)

        for i, e in enumerate(expansions):
            unsorted_expansions[sorted_sents_inds[i]] = e
        tm.add_timing("unsort sentences")
        return unsorted_expansions


    @staticmethod
    def make_vector_from_expansions(expansion, scale):
        eps = 1.0 / scale
        vec = Counter({t: int(scale * w) for n, (t, w) in enumerate(expansion)})  # if w>eps } )
        return {t: w for t, w in vec.most_common()}

    def convert_token_ids_to_tokens(self, res):
        expansions = [ [(self.tokenizer.convert_ids_to_tokens(int(tok[0])), float(tok[1])) for tok in elem ] for elem in res ]
        return expansions

class SparseEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_or_directory_name, batch_size=128, **kwargs):
        super().__init__(model_or_directory_name=model_or_directory_name, batch_size=batch_size, **kwargs)
        self.model = None
        import torch
        device = 'cuda' if torch.cuda.is_available() else "cpu" # "mps" if torch.backends.mps.is_available() else 'cpu'
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
            self.create_model(model_or_directory_name=model_or_directory_name, device=device, **kwargs)
        except Exception as e:
            # Try once more, from dmf
            if not dmf_loaded:
                model_or_directory_name = self.pull_from_dmf(model_or_directory_name)
                self.create_model(model_or_directory_name=model_or_directory_name, device=device)
            else:
                print(f"Model not found: {model_or_directory_name}")
                raise RuntimeError(f"Model not found: {model_or_directory_name}")
        print('=== done initializing model')

    @property
    def tokenizer(self):
        return self.model.tokenizer


    def create_model(self, model_or_directory_name:str=None, device:str="cpu", **kwargs):
        self.model = SparseSentenceTransformer(model_or_directory_name, device, **kwargs)

    def __call__(self, texts: Union[List[str], str], **kwargs) -> \
            Union[Dict[str, float|int], List[Dict[str, float|int]]]:
        return self.encode(texts)

    def encode_documents(self, *args, **kwargs) -> Union[Dict[str, float|int], List[Dict[str, float|int]]]:
        return self.encode(*args, **kwargs)

    def encode(self, texts: Union[str, List[str]], _batch_size: int = -1,
               show_progress_bar=None,
               tqdm_instance=None,
               **kwargs) -> \
            Union[Dict[str, float|int], List[Dict[str, float|int]]]:
        if _batch_size == -1:
            _batch_size = self.batch_size
        if show_progress_bar is None:
            show_progress_bar = not (isinstance(texts, str) or max(len(texts), _batch_size) <= 1)

        res = self.model.encode(texts, _batch_size=_batch_size,
                                show_progress_bar=show_progress_bar,
                                tqdm_instance=tqdm_instance,
                                **kwargs)
        
        if kwargs.get("create_vector_for_ingestion", False):
            res = self.model.convert_token_ids_to_tokens(res)

        return res

    def encode_query(self, query: str, show_progress_bar=False, tqdm_instance=None, prompt_name=None, **kwargs):
        # return self.encode(query, max_terms=self.model.query_max_tokens)
        return self.model.encode([query], max_terms=self.model.query_max_tokens,
                                 show_progress_bar=show_progress_bar, prompt_name=prompt_name, **kwargs)[0]

