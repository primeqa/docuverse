import numpy as np
import torch
from typing import Union, List, Any

from docuverse.utils import get_param, detect_device, convert_to_type
from docuverse.utils.embeddings.embedding_function import EmbeddingFunction
import simple_colors


class DenseEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_or_directory_name, batch_size=128, **kwargs):
        super().__init__(model_or_directory_name=model_or_directory_name, batch_size=batch_size, **kwargs)
        self.model = None
        device = self.detect_current_device(kwargs)

        self.pqa = False
        self.emb_pool = None
        dmf_loaded = False
        if get_param(kwargs, 'from_dmf', None) is not None:
            model_or_directory_name = self.pull_from_dmf(model_or_directory_name)
            dmf_loaded = True

        model_loaded = False
        try:
            self.create_model(model_or_directory_name, device,
                              attn_implementation=get_param(kwargs, 'attn_implementation', "sdpa"))
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

    def detect_current_device(self, kwargs: dict[str, Any]) -> str:
        import torch
        device = detect_device()
        if device == 'cpu':
            print(f"You are using {device}. This is much slower than using "
                  "a CUDA-enabled GPU. If on Colab you can change this by "
                  "clicking Runtime > Change runtime type > GPU.")
            self.num_devices = 0
        else:
            self.num_devices = torch.cuda.device_count()
            if torch.cuda.is_available():
                gpus = simple_colors.red([torch.cuda.get_device_name(i) for i in range(self.num_devices)], ['bold'])
                attn = simple_colors.yellow(get_param(kwargs, 'attn_implementation', "sdpa"))
                print(f"Running on the gpus:{gpus}, attention: {attn}")
            elif torch.backends.mps.is_available():
                print(f"Running on the {simple_colors.red('mps')} ")
        return device

    def __call__(self, texts: Union[List[str], str], **kwargs) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        return self.encode(texts, **kwargs)


    def __del__(self):
        # Clean up multiprocess pool
        if self.emb_pool is not None:
            try:
                self.stop_pool()
            except:
                pass
            self.emb_pool = None

        # Clean up model and free GPU memory
        if hasattr(self, 'model') and self.model is not None:
            try:
                # Move model to CPU to free GPU memory
                if hasattr(self.model, 'to'):
                    self.model = self.model.to('cpu')
                # Delete model reference
                del self.model
                self.model = None
                # Clear CUDA cache
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def create_model(self, model_or_directory_name: str=None, device: str='cpu',
                     attn_implementation="sdpa"):
        from sentence_transformers import SentenceTransformer

        if attn_implementation is not None:
            model_args: dict[str, Any] = {"attn_implementation": attn_implementation}
            if attn_implementation.find("flash") >= 0:
                import torch
                model_args["dtype"] = torch.bfloat16
            else:
                model_args['dtype'] = self.torch_dtype

            self.model = SentenceTransformer(model_or_directory_name,
                                             device=device,
                                             trust_remote_code=True,
                                             model_kwargs=model_args)
        else:
            self.model = SentenceTransformer(model_or_directory_name, device=device, trust_remote_code=True)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def start_pool(self):
        self.emb_pool = self.model.start_multi_process_pool()

    def stop_pool(self):
        self.model.stop_multi_process_pool(self.emb_pool)
        self.emb_pool = None

    def encode(self,
               texts: Union[str, List[str]], _batch_size: int = -1,
               show_progress_bar=None,
               tqdm_instance=None,
               prompt_name=None,
               tm=None,
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
                    tems = self._encode_data(texts=stexts[i:i_end], _batch_size=_batch_size,
                                             show_progress_bar=False, tm=tm)
                    embs.extend(tems)
                    del tems  # Free memory immediately
                    tqdm_instance.update(i_end - i)
            else:
                embs = self._encode_data(texts=stexts, _batch_size=_batch_size,
                                         show_progress_bar=show_progress_bar,
                                         prompt_name=prompt_name, tm=tm)
            # Optimize memory by reordering in-place where possible
            result_embs = [None] * len(texts)
            for i in range(len(texts)):
                result_embs[sorted_inds[i]] = embs[i]
            del embs  # Free original list memory
            embs = result_embs
        else:
            raise NotImplemented

        if self.matryoshka_dim > 0:
            embs = self._truncate_and_renormalize(embs)

        return embs

    def _truncate_and_renormalize(self, embs):
        """Truncate embeddings to matryoshka_dim and re-normalize."""
        d = self.matryoshka_dim
        truncated = []
        for e in embs:
            t = e[:d]
            norm = np.linalg.norm(t)
            if norm > 0:
                t = (np.array(t) / norm).tolist()
            truncated.append(t)
        return truncated

    def _encode_data(self, texts, _batch_size,
                     show_progress_bar,
                     prompt_name=None, tm=None):
        embs = []
        if isinstance(texts, list) and len(texts) > 30 and self.num_devices > 1:
            try:
                if self.emb_pool is None:
                    self.start_pool()
                embs = self.model.encode_multi_process(pool=self.emb_pool,
                                                       sentences=texts,
                                                       batch_size=_batch_size,
                                                       prompt_name=prompt_name)
                embs = self.normalize(embs)
            except Exception as e:
                # Ensure pool is cleaned up on error
                if self.emb_pool is not None:
                    try:
                        self.stop_pool()
                    except:
                        pass
                raise e
        else:
            _with_torch = torch.cuda.is_available()
            while _batch_size >= 1:
                try:
                    embs = self._tokenize_and_encode(
                        texts, _batch_size, show_progress_bar,
                        normalize_embeddings=True, prompt_name=prompt_name,
                        tm=tm
                    )
                    return embs
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f"Got an error: {e}, reducing batch size to {_batch_size // 2}")
                    if "out of memory" in str(e):
                        print(f"GPU out of memory, reducing batch size from {_batch_size} to {_batch_size // 2}")
                        if _with_torch:
                            torch.cuda.empty_cache()
                        if _batch_size <= 1:
                            print("Using CPU for current batch only, will continue on GPU")
                            original_device = self.model.device
                            cpu_model = self.model.to('cpu')
                            embs = cpu_model.encode(texts,
                                                     show_progress_bar=show_progress_bar,
                                                     normalize_embeddings=True,
                                                     batch_size=1,
                                                     prompt_name=prompt_name
                                                     ).tolist()
                            self.model = cpu_model.to(original_device)
                            del cpu_model
                            if _with_torch:
                                torch.cuda.empty_cache()
                            return embs
                    _batch_size = _batch_size // 2
            raise RuntimeError("Out of memory, you're out of luck...")
        return embs

    def _tokenize_and_encode(self, texts, batch_size, show_progress_bar,
                             normalize_embeddings=True, prompt_name=None, tm=None):
        """Encode texts with separate tokenization and forward-pass timing."""
        from docuverse.utils.timer import timer
        from sentence_transformers.util import batch_to_device
        from tqdm.auto import tqdm

        model = self.model
        device = model.device

        # Resolve prompt from prompt_name
        prompt = None
        if prompt_name is not None:
            prompt = model.prompts.get(prompt_name)

        # Prepend prompt to texts if needed
        if prompt is not None:
            sentences = [prompt + text for text in texts]
        else:
            sentences = list(texts)

        # Sort by length for efficient batching (same as SentenceTransformer.encode)
        length_sorted_idx = np.argsort([-model._text_length(s) for s in sentences])
        sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]

        # Build extra_features for prompt_length
        extra_features = {}
        if prompt is not None and hasattr(model, 'tokenize'):
            prompt_tok = model.tokenize([prompt])
            if "input_ids" in prompt_tok:
                extra_features["prompt_length"] = prompt_tok["input_ids"].shape[-1] - 1

        # Create sub-timer for encode phases
        if tm is None:
            tm = timer(timer.get_top_method("encode"))

        # Phase 1: Tokenize all batches
        all_features = []
        for start_index in range(0, len(sentences_sorted), batch_size):
            batch = sentences_sorted[start_index:start_index + batch_size]
            features = model.tokenize(batch)
            all_features.append(features)
        tm.add_timing("encode::tokenize")

        # Phase 2: Forward pass on all batches
        all_embeddings = []
        for features in tqdm(all_features, desc="Encoding", disable=not show_progress_bar):
            features = batch_to_device(features, device)
            features.update(extra_features)
            with torch.no_grad():
                out_features = model.forward(features)
                embeddings = out_features["sentence_embedding"].detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu())
        if device.type == 'cuda':
            torch.cuda.synchronize()
        tm.add_timing("encode::model_forward")

        # Unsort back to original order
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        # Convert to list of lists
        return np.asarray([emb.float().numpy() if emb.dtype == torch.bfloat16
                           else emb.numpy() for emb in all_embeddings]).tolist()

    @staticmethod
    def normalize(passage_vectors):
        return [v / np.linalg.norm(v) for v in passage_vectors if np.linalg.norm(v) > 0]

