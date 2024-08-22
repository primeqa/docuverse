from copy import deepcopy

from tqdm import tqdm

from docuverse.engines.search_result import SearchResult


class Reranker(object):
    def __init__(self, reranking_config, **kwargs):
        self.model = None
        self.config = reranking_config
        self.name = reranking_config['name']

    def similarity(self, embedding1, embedding2):
        pass

    def rerank(self, documents, show_progress=True):
        is_single_instance = isinstance(documents, SearchResult)
        if is_single_instance:
            documents = [documents]
        output = []
        _batch_size =  self.config.reranker_batch_size
        counter = None
        num_docs = len(documents)
        if show_progress:
            counter = tqdm(desc="Reranking documents: ", total=num_docs)

        for b in range(0, num_docs, _batch_size):
            texts = []
            for doc in documents[b:min(num_docs,b+_batch_size)]:
                texts.append(doc.question.text)
                texts.extend([datum.text for datum in doc])
            vals = self.model.encode(texts, _batch_size=self.config.reranker_gpu_batch_size)
            #continue
            pos = 0
            for bi, doc in enumerate(documents[b:b+_batch_size]):
                question_embedding = vals[pos]
                num_examples = len(doc)
                embeddings = vals[pos+1:pos+1+num_examples]
                similarity_scores = [self.similarity(question_embedding, embedding) for embedding in embeddings]
                # Convert a list of nearness scores to a list of document IDs
                hybrid_similarities = [0] * num_examples
                if self.config.reranker_combination_type == 'weight':
                    weight = self.config.reranker_combine_weight
                    for result, similarity in zip(documents, similarity_scores):
                        hybrid_similarities.append(similarity*weight + result.score*(1-weight))
                elif self.config.reranker_combination_type == 'rrf':
                    # Inverse rank combination appears to work better than simple addition or multiplication of scores.
                    # For this, we combine the ranks, not the separate scores (which may not be normalized to each other).
                    simple_rerank_order = sorted(range(0, len(similarity_scores)),
                                                 key=lambda i: similarity_scores[i],
                                                 reverse=True)
                    idx_to_rerank_idx = {simple_rerank_order[idx]: idx for idx in range(0, len(simple_rerank_order))}
                    k = num_docs
                    for i in range(num_examples):
                        hybrid_similarities[i] = 1.0/(k+i+1) + 1.0/(k+idx_to_rerank_idx[i]+1)

                sorted_similarities = sorted(zip(doc, hybrid_similarities),
                                             key=lambda pair: pair[1], reverse=True)

                op = SearchResult(doc.question, [])
                for _doc, sim in sorted_similarities:
                    doc1 = deepcopy(_doc)
                    doc1.score = sim
                    op.append(doc1)
                output.append(op)
                pos += len(doc)+1
            if show_progress:
                counter.update(_batch_size)

        if show_progress:
            counter.close()
            counter.clear()
        if is_single_instance:
            return output[0]
        else:
            return output