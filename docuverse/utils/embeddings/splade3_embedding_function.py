from typing import Union, Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer

from docuverse.utils import get_param, convert_to_single_vectors
from docuverse.utils.embeddings.embedding_function import EmbeddingFunction
import torch
try:
    from pymilvus import model
except ImportError as e:
    print("You need to install pymilvus to be able to use this function")


class SpladeEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_or_directory_name, batch_size=128, **kwargs):
        super().__init__(model_or_directory_name=model_or_directory_name, batch_size=batch_size, **kwargs)
        self.tokenizer = None
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

    def create_model(self, model_or_directory_name: str = None, device: str = "cpu", **kwargs):
        self.model = model.sparse.SpladeEmbeddingFunction(model_name=model_or_directory_name,
                                                          device=device,
                                                          k_tokens_query=get_param(kwargs, 'query_max_tokens', None),
                                                          k_tokens_document = get_param(kwargs, 'doc_max_tokens', None)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_or_directory_name)

    def __call__(self, texts: Union[List[str], str], **kwargs) -> \
            Union[Dict[str, float | int], List[Dict[str, float | int]]]:
        return self.encode(texts, **kwargs)

    # def encode(self, texts: Union[str, List[str]], _batch_size: int = -1, show_progress_bar=None, **kwargs) -> \
    #         Union[Dict[str, float | int], List[Dict[str, float | int]]]:
    #     if _batch_size == -1:
    #         _batch_size = self.batch_size
    #     if show_progress_bar is None:
    #         show_progress_bar = not (isinstance(texts, str) or max(len(texts), _batch_size) <= 1)
    #
    #     # res = [self.model.encode(text, is_q=False) for text in texts]
    #     res = []
    #     num_sents = len(texts)
    #     sorted_sents_inds = sorted(range(0, len(texts)), key=lambda x: len(texts[x]), reverse=True)
    #     # sorted_sents = [sentences[i] for i in sorted_sents_inds]
    #     _batch_size = 1
    #     for i in range(0, len(texts), _batch_size):
    #         end = min(i + _batch_size, num_sents)
    #         _input = self.tokenizer([texts[sorted_sents_inds[pi]] for pi in range(i,end)],
    #                                 add_special_tokens=True,
    #                                 padding="longest",  # pad to max sequence length in batch
    #                                 truncation="longest_first",  # truncates to self.max_length
    #                                 return_attention_mask=True)
    #         _input = {k: torch.tensor(v) for k, v in _input.items()}
    #         # _input['id'] = torch.tensor([int(i) for i in range(i,end)], dtype=torch.long)
    #         out = self.model(d_kwargs=_input)
    #         res.extend(out)
    #
    #     if kwargs.get("create_vector_for_ingestion", False):
    #         res = self.model.convert_token_ids_to_tokens(res)
    #
    #     return res

    @staticmethod
    def _encode_data(self, method, _batch_size, show_progress_bar, texts, tqdm_instance):
        embs = []
        if tqdm_instance is not None:
            tq = tqdm_instance
        else:
            tq = tqdm(desc=f"Encoding texts w/ SPLADE", total=len(texts),
                      show_progress_bar=False if show_progress_bar is None else show_progress_bar)
        if _batch_size == -1:
            _batch_size = len(texts)/10
        for i in range(0, len(texts), _batch_size):
            last = min(len(texts), i + _batch_size)
            embs.extend(convert_to_single_vectors(method(texts[i:i + _batch_size])))
            tq.update(last - i)
        return embs

    def encode(self, texts: Union[str, List[str]], _batch_size: int = -1, show_progress_bar=None,
               tqdm_instance=None, **kwargs) -> \
            Union[Dict[str, float | int], List[Dict[str, float | int]]]:
        return self._encode_data(self.model.encode_documents,
                                 _batch_size, show_progress_bar, texts, tqdm_instance)
        # return [self.model.encode_documents(t) for t in texts] if type(texts) is list else self.model.encode_documents(texts)


    def encode_query(self, texts: Union[str, List[str]], _batch_size: int = -1, show_progress_bar=None,
                     tqdm_instance=None, **kwargs) -> \
            Union[Dict[str, float | int], List[Dict[str, float | int]]]:
        return self._encode_data(self.model.encode_queries,
                                 _batch_size, show_progress_bar, texts, tqdm_instance)
        # return convert_to_single_vectors(self.model.encode_queries(texts))
