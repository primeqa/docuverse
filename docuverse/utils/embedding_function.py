from transformers import AutoTokenizer
import torch
import numpy as np
from typing import Union, List


class DenseEmbeddingFunction:
    def __init__(self, name, batch_size=128):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # "mps" if torch.backends.mps.is_available() else 'cpu'
        if device == 'cpu':
            print(f"You are using {device}. This is much slower than using "
                  "a CUDA-enabled GPU. If on Colab you can change this by "
                  "clicking Runtime > Change runtime type > GPU.")
            self.num_devices = 0
        else:
            self.num_devices = torch.cuda.device_count()
        self.pqa = False
        self.batch_size = batch_size
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
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)
        print('=== done initializing model')

    def __call__(self, texts: Union[List[str], str]) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        return self.encode(texts)

    def __del__(self):
        if self.emb_pool is not None:
            self.stop_pool()

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def start_pool(self):
        self.emb_pool = self.model.start_multi_process_pool()

    def stop_pool(self):
        self.model.stop_multi_process_pool(self.emb_pool)
        self.emb_pool = None

    def encode(self, texts: Union[str, List[str]], _batch_size: int = -1, show_progress_bar=None) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        embs = []
        if _batch_size == -1:
            _batch_size = self.batch_size
        if show_progress_bar is None:
            show_progress_bar = isinstance(texts, str) or max(len(texts), _batch_size) <= 1

        if not self.pqa:
            if isinstance(texts, list) and len(texts) > 30 and self.num_devices > 1:
                if self.emb_pool is None:
                    self.start_pool()
                embs = self.model.encode_multi_process(pool=self.emb_pool,
                                                       sentences=texts,
                                                       batch_size=_batch_size)
                embs = self.normalize(embs)
            else:
                embs = self.model.encode(texts,
                                         show_progress_bar=show_progress_bar,
                                         normalize_embeddings=True
                                         ).tolist()
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

    @staticmethod
    def normalize(passage_vectors):
        return [v / np.linalg.norm(v) for v in passage_vectors if np.linalg.norm(v) > 0]
