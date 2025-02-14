from copy import deepcopy
from tqdm import tqdm

from docuverse.engines.search_result import SearchResult
from docuverse.utils.timer import timer
import torch

class Reranker(object):
    def __init__(self, reranking_config, **kwargs):
        self.model = None
        self.config = reranking_config
        self.name = reranking_config['name']

    def similarity(self, embedding1, embedding2, device='cuda'):
        pass

    def pair_similarity(self, embedding_pair, device='cuda'):
        return self.similarity(embedding_pair[0], embedding_pair[1], device=device)

    def rerank(self, answer_list, show_progress=True):
        tm = timer("reranking")
        is_single_instance = isinstance(answer_list, SearchResult)
        if is_single_instance:
            answer_list = [answer_list]
        output = []
        _batch_size =  self.config.reranker_batch_size
        counter = None
        num_docs = len(answer_list)

        texts = [d.question.text for d in answer_list]
        id2pos = {}

        for answer in answer_list:
            for doc in answer:
                if doc.id not in id2pos:
                    id2pos[doc.id] = len(texts)
                    texts.append(doc.text)
        tm.add_timing("preparation")
        if self.config.reranker_lowercase:
            embeddings = self.model.encode([l.lower() for l in texts], show_progress_bar=True)
        else:
            embeddings = self.model.encode(texts, show_progress_bar=True, message="Reranking answers")

        for qid, answer in tqdm(enumerate(answer_list), desc="Computing Cosine: ",
                                total=len(answer_list), disable=not show_progress):
            num_examples = len(answer)
            if num_examples == 0:
                output.append(answer)
                continue
            qembed = embeddings[qid]
            similarity_scores = self.similarity(qembed, [embeddings[id2pos[doc.id]] for doc in answer])
            # similarity_scores = embedding_scores[qid]
            tm.mark() # make sure the time does not include the time spent in computing similarity_scores.
            hybrid_similarities = [0] * num_examples
            if self.config.reranker_combination_type == 'none':
                hybrid_similarities = similarity_scores
                if isinstance(hybrid_similarities, torch.Tensor):
                    hybrid_similarities = hybrid_similarities.tolist()
            elif self.config.reranker_combination_type == 'weight':
                weight = self.config.reranker_combine_weight
                for result, similarity in zip(answer, similarity_scores):
                    hybrid_similarities.append(similarity * weight + result.score * (1 - weight))
            elif self.config.reranker_combination_type == 'rrf':
                # Inverse rank combination appears to work better than simple addition or multiplication of scores.
                # For this, we combine the ranks, not the separate scores (which may not be normalized to each other).
                simple_rerank_order = sorted(range(0, len(similarity_scores)),
                                             key=lambda i: similarity_scores[i],
                                             reverse=True)
                idx_to_rerank_idx = {simple_rerank_order[idx]: idx for idx in range(0, len(simple_rerank_order))}
                k = num_docs
                for i in range(num_examples):
                    hybrid_similarities[i] = 1.0 / (k + i + 1) + 1.0 / (k + idx_to_rerank_idx[i] + 1)

            sorted_similarities = sorted(zip(answer, hybrid_similarities),
                                         key=lambda pair: pair[1], reverse=True)
            tm.add_timing("cosine::reorder")
            op = SearchResult(answer.question, [])
            for _doc, sim in sorted_similarities:
                doc1 = deepcopy(_doc)
                doc1.score = sim
                op.append(doc1)
            output.append(op)
            tm.add_timing("cosine::copy_data")
        #tm.add_timing("cosine")
        total_docs = (len(answer_list[0]) + 1) * len(answer_list)
        saved = total_docs - len(embeddings)
        print(f"Saved {saved} embedding computations ({saved/total_docs:.1%}).")
        return output[0] if is_single_instance else output
