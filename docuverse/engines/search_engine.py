from typing import List

import yaml
import os

from tqdm import tqdm

from docuverse.engines.search_engine_config_params import DocUVerseConfig, SearchEngineArguments
from docuverse.engines.search_result import SearchResult
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.utils.evaluation_output import EvaluationOutput
from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.reranking.Reranker import Reranker

class SearchEngine:
    DEFAULT_CACHE_DIR = os.path.join(f"{os.getenv('HOME')}", ".local", "share", "elastic_ingestion")

    # def __init__(self, retrieval_params, reranking_params=None, **kwargs):
    #     self.retrieval_params = retrieval_params
    #     self.reranking_params = reranking_params
    #     self.cache_dir = get_param('cache_dir', self.DEFAULT_CACHE_DIR)
    #     self.cache_policy = get_param('cache_policy', 'always')
    #     self.rouge_duplicate_threshold = get_param('rouge_duplicate_threshold', -1)
    #     self.duplicate_removal_ = get_param('duplicate_removal', "none")
    def __init__(self, config_path: str = None, **kwargs):
        self.retriever = None
        self.reranker = None
        self.reranking_config = None
        self.retrieval_config = None
        self.create(config_or_path=config_path)


    @staticmethod
    def read_config(config_or_path):
        """
        Reads the configuration file at the specified path and returns the retrieved values for retrieval
        and reranking. The configuration file is consistent to the mf-coga config file (if that doesn't make sense
        don't worry about it :) ).
        The format of the yaml file is as follows:
          * The file should contain 'retrieval' and 'reranking' parameters; if no 'retrieval' is present, then the
          file is assumed to be flat.
          * The 'retrieval' parameter should have a value 'name' which will be used to decide the engine type.

        :param config_or_path: The path to the configuration file.
        :type config_or_path: str

        :return: A tuple containing the retrieved values for retrieval and reranking.
                 If the configuration file does not contain values for retrieval or reranking,
                 the corresponding value in the tuple will be None.
        :rtype: tuple

        :raises yaml.YAMLError: If there is an error while loading the configuration file.
        """
        if isinstance(config_or_path, str):
            if os.path.exists(config_or_path):
                with open(config_or_path) as stream:
                    try:
                        vals = yaml.safe_load(stream=stream)
                        return vals['retrieval'] if 'retrieval' in vals else vals, vals[
                            'reranking'] if 'reranking' in vals else None

                    except yaml.YAMLError as exc:
                        raise exc
            else:
                print(f"The configuration file '{config_or_path}' does not exist.")
                raise FileNotFoundError(f"The configuration file '{config_or_path}' does not exist.")
        elif isinstance(config_or_path, DocUVerseConfig|SearchEngineArguments):
            return config_or_path, None

    def create(self, config_or_path, **kwargs):
        self.retrieval_config, self.reranking_config = SearchEngine.read_config(config_or_path)

        if self.reranking_config is not None:
            self.reranker = SearchEngine._create_reranker(self.reranking_config)
        self.retriever = SearchEngine._create_retriever(self.retrieval_config)

    @staticmethod
    def _create_retriever(retrieval_config) -> RetrievalEngine:
        return RetrievalEngine(retrieval_config)

    @classmethod
    def _create_reranker(cls, reranking_config=None) -> Reranker:
        if reranking_config is None:
            return None

        reranker = Reranker(reranking_config)
        return reranker

    def ingest(self, corpus: SearchCorpus):
        self.retriever.ingest(corpus=corpus)
    
    def get_retriever_info(self):
        return self.retriever.info()

    def search(self, queries: SearchQueries) -> List[SearchResult]:
        answers = [self.retriever.search(query) for query in tqdm(queries, desc="Processing queries: ")]
        return answers

    def set_index(self, index=None):
        pass

    def compute_score(self, queries: SearchQueries, results: SearchResult) -> EvaluationOutput:
        pass

