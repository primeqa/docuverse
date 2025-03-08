from copy import deepcopy

from docuverse.engines.search_result import SearchResult
from docuverse.utils.timer import timer


class Reranker(object):
    """
    Handles reranking of search results using similarity models.

    This class provides functionality to rerank search results based on provided similarity
    measures. It supports batch processing of search results and maintains timing
    metrics associated with the reranking process.

    Attributes:
        model: The similarity model used for reranking.
        config: The configuration details for the reranking process, including batch size
            and model specifics as defined in the `reranking_config`.
        name: The name of the reranker, as specified in the `reranking_config`.
        tm: An instance of the timer utility to track timing information.
    """
    def __init__(self, reranking_config, **kwargs):
        self.model = None
        self.config = reranking_config
        self.name = reranking_config['name']
        self.tm = timer("reranking")


    def similarity(self, embedding1, embedding2, device='cuda'):
        """
        Calculates similarity between two embeddings using a specified device.

        This method computes the similarity between two embedding vectors
        using a specified computational device. It can be used in various
        applications such as measuring distance or relationship between
        two feature representations.

        Args:
            embedding1 (list[float] | torch.Tensor): First embedding vector.
            embedding2 (list[float] | torch.Tensor): Second embedding vector.
            device (str): The computational device used for operation. Defaults
                to 'cuda'.

        Returns:
            float: Similarity score between the two embeddings.
        """
        pass

    def rerank(self, answer_list, show_progress=True):
        """
        Reranks the given list of search results using the configured reranker.

        This method processes a list of SearchResult objects (or a single SearchResult object)
        and applies a reranking mechanism, returning a re-ordered or prioritized list based on
        the reranking model's criteria. It optionally allows the display of progress during
        the reranking process.

        Parameters:
            answer_list (SearchResult | list[SearchResult]): A single instance
                of SearchResult or a list of SearchResult objects to be reranked.
            show_progress (bool): Whether to display progress during the reranking
                operation. Defaults to True.

        Returns:
            list[SearchResult] or SearchResult: The reranked list of SearchResult
                objects if input is a list, or a single reranked SearchResult object
                if the input was a single instance.
        """
        is_single_instance = isinstance(answer_list, SearchResult)
        if is_single_instance:
            answer_list = [answer_list]
        _batch_size =  self.config.reranker_batch_size

        output = self._rerank(answer_list, show_progress=show_progress)
        return output[0] if is_single_instance else output

    def _rerank(self, answer_list, show_progress):
        """
        Reranks a list of answers based on specified criteria.

        To be implemented by subclasses.
        """
        pass

    def _build_sorted_list(self, answer: SearchResult, similarities: list[float]):
        """
        Builds a sorted list of documents based on their similarity scores and updates the resulting search result.

        The function takes in a list of answers and their corresponding similarity scores,
        sorts them in descending order of similarity, and creates a new SearchResult object
        containing the sorted documents. It also updates timing information during execution.

        Args:
            answer (list): List of document objects or answers.
            similarities (list): List of similarity scores corresponding to the documents.

        Returns:
            SearchResult: An updated search result object containing the sorted documents.
        """
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