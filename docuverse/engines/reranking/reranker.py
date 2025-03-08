from copy import deepcopy

from docuverse.engines.search_result import SearchResult
from docuverse.utils.timer import timer


class Reranker(object):
    def __init__(self, reranking_config, **kwargs):
        self.model = None
        self.config = reranking_config
        self.name = reranking_config['name']
        self.tm = timer("reranking")


    def similarity(self, embedding1, embedding2, device='cuda'):
        pass

    def rerank(self, answer_list, show_progress=True):
        is_single_instance = isinstance(answer_list, SearchResult)
        if is_single_instance:
            answer_list = [answer_list]
        _batch_size =  self.config.reranker_batch_size

        output = self._rerank(answer_list, show_progress=show_progress)
        return output[0] if is_single_instance else output

    def _rerank(self, answer_list, show_progress):
        pass

    def _build_sorted_list(self, answer, similarities):
        sorted_similarities = sorted(zip(answer, similarities),
                                     key=lambda pair: pair[1], reverse=True)
        self.tm.add_timing("cosine::reorder")
        op = SearchResult(answer.question, [])
        for _doc, sim in sorted_similarities:
            doc1 = deepcopy(_doc)
            doc1.score = sim
            op.append(doc1)
        self.tm.add_timing("cosine::copy_data")
        return op