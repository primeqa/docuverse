from tqdm import tqdm

from docuverse.engines.reranking.reranker import Reranker
import torch


class BiEncoderReranker(Reranker):
    def pair_similarity(self, embedding_pair, device='cuda'):
        return self.similarity(embedding_pair[0], embedding_pair[1], device=device)

    def _rerank(self, answer_list, show_progress):
        texts = [d.question.text for d in answer_list]
        id2pos = {}
        output = []

        for answer in answer_list:
            for doc in answer.top_k(self.top_k):
                if doc.id not in id2pos:
                    id2pos[doc.id] = len(texts)
                    texts.append(doc.text)
        self.tm.add_timing("preparation")

        num_docs = len(answer_list)
        if self.config.reranker_lowercase:
            embeddings = self.model.encode([l.lower() for l in texts], show_progress_bar=True)
        else:
            embeddings = self.model.encode(texts, show_progress_bar=True, message="Reranking answers")
        for qid, answer in tqdm(enumerate(answer_list), desc="Computing Cosine: ",
                                total=num_docs, disable=not show_progress):
            num_examples = len(answer)
            if num_examples == 0:
                output.append(answer)
                continue
            qembed = embeddings[qid]
            similarity_scores = self.similarity(qembed, [embeddings[id2pos[doc.id]] for doc in answer])

            self.tm.mark()  # make sure the time does not include the time spent in computing similarity_scores.
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
                k = num_examples
                for i in range(num_examples):
                    hybrid_similarities[i] = 1.0 / (k + i + 1) + 1.0 / (k + idx_to_rerank_idx[i] + 1)
            reranked_list = self._build_sorted_list(answer, similarity_scores)
            output.append(reranked_list)
            # self._build_sorted_list(answer, hybrid_similarities, output)

        total_docs = (len(answer_list[0]) + 1) * len(answer_list)
        saved = total_docs - len(embeddings)
        print(f"Saved {saved} embedding computations ({saved / total_docs:.1%}).")
        return output
