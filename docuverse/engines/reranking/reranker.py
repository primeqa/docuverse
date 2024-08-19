from copy import deepcopy

from docuverse.engines.search_result import SearchResult


class Reranker(object):
    def __init__(self, reranking_config, **kwargs):
        self.model = None
        self.config = reranking_config
        self.name = reranking_config['name']

    def similarity(self, embedding1, embedding2):
        pass

    def rerank(self, documents):
        if isinstance(documents, list):
            return [self.rerank(d) for d in documents]

        texts = [documents.question.text] + [datum.text for datum in documents]
        vals = self.model.encode(texts, _batch_size=self.config.reranker_batch_size)
        question_embedding = vals[0]
        embeddings = vals[1:]
        similarity_scores = [self.similarity(question_embedding, embedding) for embedding in embeddings]
        # similarity_scores = [self.cosine(question_embedding, embedding).detach().item()\
        #                      for embedding in embeddings]

        # Convert a list of nearness scores to a list of document IDs
        hybrid_similarities = []
        if self.config.reranker_combination_type == 'weight':
            weight = self.config.reranker_combine_weight
            for result, similarity in zip(documents, similarity_scores):
                hybrid_similarities.append(similarity*weight + result.score*(1-weight))
        elif self.config.combination_type == 'rrf':
            # Inverse rank combination appears to work better than simple addition or multiplication of scores.
            # For this, we combine the ranks, not the separate scores (which may not be normalized to each other).
            simple_rerank_order = sorted(range(0, len(similarity_scores)),
                                         key=lambda i: similarity_scores[i],
                                         reverse=True)
            idx_to_rerank_idx = {simple_rerank_order[idx]: idx for idx in range(0, len(simple_rerank_order))}
            k = len(documents)
            for i in range(len(documents)):
                hybrid_similarities[i] = 1.0/(k+i+1) + 1.0/(k+idx_to_rerank_idx[i]+1)

        sorted_similarities = sorted(zip(documents, hybrid_similarities),
                                     key=lambda pair: pair[1], reverse=True)

        output = SearchResult(documents.question, [])
        for doc, sim in sorted_similarities:
            doc1 = deepcopy(doc)
            doc1.score = sim
            output.append(doc1)
        return output