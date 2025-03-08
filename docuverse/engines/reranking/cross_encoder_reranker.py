import torch
from sentence_transformers import SentenceTransformerModelCardData, SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .reranker import Reranker
from docuverse.engines.search_engine_config_params import RerankerConfig as RerankerConfig
from sentence_transformers.cross_encoder import CrossEncoder
from docuverse.engines.search_result import SearchResult

class CrossEncoderModel:
    def __init__(self, model_name, device='cuda'):
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, pairs, device='cuda'):
        scores = None
        with torch.no_grad():
            enc = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt').to(device)
            res = self.model(**enc, return_dict=True)
            # scores = res.logits.view(-1, ).float()
            # scores = res.last_hidden_state[:,0,0].float()
            scores = res.last_hidden_state[:,0,:].softmax(dim=1)[:,0].tolist()
            # pred_ids = scores.argsort(descending=True)
        return scores

class CrossEncoderReranker(Reranker):
    def __init__(self, reranking_config: RerankerConfig|dict, **kwargs):
        super().__init__(reranking_config, **kwargs)
        self.model = CrossEncoderModel(reranking_config.reranker_model)

    def _rerank(self, answer_list, show_progress):
        num_docs = len(answer_list)
        output = []
        for answer in tqdm(answer_list, desc="Computing cross-encodings",
                                total=num_docs, disable=not show_progress):
            similarity_scores = self.model.predict([[answer.question.text, t.text] for t in answer])
            sorted_similarities = sorted(zip(answer, similarity_scores),
                                         key=lambda pair: pair[1], reverse=True)
            self.tm.add_timing("cosine::reorder")
            output.append(self._build_sorted_list(answer, sorted_similarities))
        return output

