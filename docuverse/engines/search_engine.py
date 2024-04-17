import yaml
import os
from docuverse.engines import SearchResult, SearchCorpus, SearchQueries, EvaluationOutput, RetrievalEngine
from docuverse.utils import get_param

class SearchEngine(object):
    DEFAULT_CACHE_DIR = os.path.join(f"{os.getenv('HOME')}", ".local", "share", "elastic_ingestion")

    def __init__(self, retrieval_params, reranking_params=None, **kwargs):
        self.retrieval_params = retrieval_params
        self.reranking_params = reranking_params
        self.cache_dir = get_param(kwargs, 'cache_dir', self.DEFAULT_CACHE_DIR)
        self.cache_policy = get_param(kwargs,'cache_policy', 'always')
        self.rouge_duplicate_threshold = get_param(kwargs,'rouge_duplicate_threshold', -1)
        self.duplicate_removal_ = get_param(kwargs, 'duplicate_removal', "none")


    @staticmethod
    def read_config(config_path):
        """
        Reads the configuration file at the specified path and returns the retrieved values for retrieval
        and reranking. The configuration file is consistent to the mf-coga config file (if that doesn't make sense
        don't worry about it :) ).
        The format of the yaml file is as follows:
          * The file should contain 'retrieval' and 'reranking' parameters; if no 'retrieval' is present, then the
          file is assumed to be flat.
          * The 'retrieval' parameter should have a value 'name' which will be used to decide the engine type.

        :param config_path: The path to the configuration file.
        :type config_path: str

        :return: A tuple containing the retrieved values for retrieval and reranking.
                 If the configuration file does not contain values for retrieval or reranking,
                 the corresponding value in the tuple will be None.
        :rtype: tuple

        :raises yaml.YAMLError: If there is an error while loading the configuration file.
        """
        try:
            vals = yaml.safe_load(open(config_path))
            return vals['retrieval'] if 'retrieval' in vals else vals, vals[
                'reranking'] if 'reranking' in vals else None

        except yaml.YAMLError as exc:
            raise exc

    @staticmethod

    @staticmethod
    def create(config_path, **kwargs):
        retrieval_config, reranking_config = SearchEngine.read_config(config_path)

        reranker = None
        if reranking_config is not None:
            reranker = SearchEngine._create_reranker(reranking_config)
        retriever = SearchEngine._create_retriever(retrieval_config)
        return retriever, reranker

    @staticmethod
    def _create_retriever(retrieval_config) -> RetrievalEngine:
        return RetrievalEngine.create_engine(retrieval_config)
    @classmethod
    def _create_reranker(cls, reranking_config):
        if reranking_config == None:
            return None

        from docuverse.engines.reranking.Reranker import Reranker
        reranker = Reranker(reranking_config)
        return reranker

    def ingest(self, corpus: SearchCorpus, output_index:str):
        pass

    def search(self, corpus: SearchQueries) -> SearchResult:
        pass

    def set_index(self, index=None):
        pass

    def compute_score(self, queries: SearchQueries, results: SearchResult) -> EvaluationOutput:
        pass
