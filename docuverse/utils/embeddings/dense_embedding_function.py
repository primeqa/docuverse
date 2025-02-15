import os

from transformers import AutoTokenizer
import torch
import numpy as np
from typing import Union, List

from docuverse.utils import get_param, get_config_dir
from docuverse.utils.embeddings.embedding_function import EmbeddingFunction
import simple_colors


class DenseEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_or_directory_name, batch_size=128, **kwargs):
        super().__init__(model_or_directory_name=model_or_directory_name, batch_size=batch_size, **kwargs)
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # "mps" if torch.backends.mps.is_available() else 'cpu'
        if device == 'cpu':
            print(f"You are using {device}. This is much slower than using "
                  "a CUDA-enabled GPU. If on Colab you can change this by "
                  "clicking Runtime > Change runtime type > GPU.")
            self.num_devices = 0
        else:
            self.num_devices = torch.cuda.device_count()
            print("Running on the gpus: ",
                  simple_colors.red([torch.cuda.get_device_name(i) for i in range(self.num_devices)], ['bold']))
        self.pqa = False
        self.emb_pool = None
        # if os.path.exists(name):
        #     raise NotImplemented
        #     # from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoConfig
        #     # self.queries_to_vectors = None # queries_to_vectors
        #     # self.model = DPRQuestionEncoder.from_pretrained(
        #     #     pretrained_model_name_or_path=name,
        #     #     from_tf = False,
        #     #     cache_dir=None,)
        #     # self.model.eval()
        #     # self.model = self.model.half()
        #     # self.model.to(device)
        #     # self.pqa = True
        # else:
        dmf_loaded = False
        if get_param(kwargs, 'from_dmf', None) is not None:
            model_or_directory_name = self.pull_from_dmf(model_or_directory_name)
            dmf_loaded = True

        model_loaded = False
        try:
            self.create_model(model_or_directory_name, device)
            model_loaded = True
        except Exception as e:
            raise e

        if dmf_loaded and not model_loaded:
            print(f"Model not found in DMF: {model_or_directory_name}")
            raise RuntimeError(f"Model not found in DMF: {model_or_directory_name}")

        if not model_loaded:
            try:
                model_or_directory_name = self.pull_from_dmf(model_or_directory_name)
                self.create_model(model_or_directory_name, device)
            except Exception as e:
                print(f"Model not found: {model_or_directory_name}")
                raise RuntimeError(f"Model not found: {model_or_directory_name}")

        print('=== done initializing model')

    def __call__(self, texts: Union[List[str], str], **kwargs) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        return self.encode(texts, **kwargs)


    def __del__(self):
        if self.emb_pool is not None:
            self.stop_pool()
            self.emb_pool = None

    def create_model(self, model_or_directory_name: str=None, device: str='cpu'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_or_directory_name, device=device)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def start_pool(self):
        self.emb_pool = self.model.start_multi_process_pool()

    def stop_pool(self):
        self.model.stop_multi_process_pool(self.emb_pool)
        self.emb_pool = None

    def encode(self, texts: Union[str, List[str]], _batch_size: int = -1,
               show_progress_bar=None,
               tqdm_instance=None,
               prompt_name=None,
               **kwargs) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        embs = []
        if _batch_size == -1:
            _batch_size = self.batch_size
        if show_progress_bar is None:
            show_progress_bar = not (isinstance(texts, str) or max(len(texts), _batch_size) <= 1)

        if not self.pqa:
            sorted_inds = sorted(range(0, len(texts)), key=lambda x: len(texts[x]), reverse=True)
            stexts = [texts[sorted_inds[i]] for i in range(len(texts))]
            if tqdm_instance is not None:
                # stexts = tqdm_instance(stexts, desc="Encoding texts", disable=not show_progress_bar)
                for i in range(0, len(texts), _batch_size):
                    i_end = min(i + _batch_size, len(texts))
                    tems = self._encode_data(texts=stexts[i:i_end],
                                             _batch_size=_batch_size,
                                             show_progress_bar=False,
                                             prompt_name=prompt_name)
                    embs.extend(tems)
                    tqdm_instance.update(i_end - i)
            else:
                embs = self._encode_data(texts=stexts, _batch_size=_batch_size,
                                         show_progress_bar=show_progress_bar,
                                         prompt_name=prompt_name)
            tmp_embs = [None] * len(texts)
            for i in range(len(texts)):
                tmp_embs[sorted_inds[i]] = embs[i]
            embs = tmp_embs
        else:
            raise NotImplemented
            # if batch_size < 0:
            #     batch_size = self.batch_size
            # if len(texts) > batch_size:
            #     embs = []
            #     for i in tqdm(range(0, len(texts), batch_size)):
            #         i_end = min(i + batch_size, len(texts))
            #         tems = self.queries_to_vectors(self.tokenizer,
            #                                        self.model,
            #                                        texts[i:i_end],
            #                                        max_query_length=500).tolist()
            #         embs.extend(tems)
            # else:
            #     embs = self.queries_to_vectors(self.tokenizer, self.model, texts, max_query_length=500).tolist()
        return embs

    def _encode_data(self, texts, _batch_size, show_progress_bar, prompt_name=None, ):
        if isinstance(texts, list) and len(texts) > 30 and self.num_devices > 1:
            if self.emb_pool is None:
                self.start_pool()
            embs = self.model.encode_multi_process(pool=self.emb_pool,
                                                   sentences=texts,
                                                   batch_size=_batch_size,
                                                   prompt_name=prompt_name)
            embs = self.normalize(embs)
        else:
            embs = self.model.encode(texts,
                                     show_progress_bar=show_progress_bar,
                                     normalize_embeddings=True,
                                     batch_size=_batch_size,
                                     prompt_name=prompt_name
                                     ).tolist()
        return embs

    @staticmethod
    def normalize(passage_vectors):
        return [v / np.linalg.norm(v) for v in passage_vectors if np.linalg.norm(v) > 0]

