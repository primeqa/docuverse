from typing import List, Tuple

import yaml
import os

from tqdm import tqdm

import docuverse.utils
from docuverse.engines import SearchData
from docuverse.engines.search_engine_config_params import DocUVerseConfig, SearchEngineArguments, RerankerArguments, \
    GenericArguments
from docuverse.engines.search_result import SearchResult
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.utils.evaluation_output import EvaluationOutput
from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.reranking.reranker import Reranker
from docuverse.utils.text_tiler import TextTiler


class SearchEngine:
    DEFAULT_CACHE_DIR = os.path.join(f"{os.getenv('HOME')}", ".local", "share", "elastic_ingestion")

    # def __init__(self, retrieval_params, reranking_params=None, **kwargs):
    #     self.retrieval_params = retrieval_params
    #     self.reranking_params = reranking_params
    #     self.cache_dir = get_param('cache_dir', self.DEFAULT_CACHE_DIR)
    #     self.cache_policy = get_param('cache_policy', 'always')
    #     self.rouge_duplicate_threshold = get_param('rouge_duplicate_threshold', -1)
    #     self.duplicate_removal_ = get_param('duplicate_removal', "none")
    def __init__(self, search_config: str = None, reranker_config: str = None, **kwargs):
        self.retriever = None
        self.reranker = None
        self.reranking_config = None
        self.retriever_config = None
        if isinstance(search_config, DocUVerseConfig):
            self.create(search_config_or_path=search_config.search_config,
                        reranker_config_or_path=search_config.reranker_config)
        else:
            self.create(search_config_or_path=search_config, reranker_config_or_path=reranker_config)
        self.tiler = None

    @staticmethod
    def read_configs(search_config_or_path: str, reranker_config_or_path: str) -> \
            Tuple[GenericArguments, GenericArguments]:
        return SearchEngine._read_config(search_config_or_path, SearchEngineArguments), \
            SearchEngine._read_config(reranker_config_or_path, RerankerArguments)

    @staticmethod
    def _read_config(config_or_path: str, TYPE) -> GenericArguments:
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
        elif isinstance(config_or_path, TYPE):
            return config_or_path

    def create(self, search_config_or_path, reranker_config_or_path, **kwargs):
        self.retriever_config, self.reranking_config = \
            SearchEngine.read_configs(search_config_or_path, reranker_config_or_path)

        self.reranker = self._create_reranker()
        self.retriever = self._create_retriever()

    def _create_retriever(self) -> RetrievalEngine:
        from docuverse.utils.retrievers import create_retrieval_engine
        return create_retrieval_engine(self.retriever_config)

    def _create_reranker(self) -> Reranker | None:
        if self.reranking_config is None:
            return None

        from docuverse.utils.retrievers import create_reranker_engine
        return create_reranker_engine(self.reranking_config)

    def ingest(self, corpus: SearchCorpus, update: bool = False):
        self.retriever.ingest(corpus=corpus, update=update)

    def get_retriever_info(self):
        return self.retriever.info()

    def search(self, queries: SearchQueries) -> List[SearchResult]:
        answers = [self.retriever.search(query) for query in tqdm(queries, desc="Processing queries: ")]
        if self.reranker is not None:
            answers = [self.reranker.rerank(answer) for answer in tqdm(answers, desc="Reranking queries: ")]
        return answers

    def set_index(self, index=None):
        pass

    def compute_score(self, queries: SearchQueries, results: SearchResult) -> EvaluationOutput:
        pass

    def read_data(self, file):
        if self.tiler is None:
            if getattr(self.retriever, 'model', None) is not None:
                tokenizer = self.retriever.model.tokenizer
            else:
                tokenizer = self.retriever_config.model_name
                if tokenizer == "" or tokenizer.startswith("."):
                    tokenizer = "sentence-transformers/all-MiniLM-L6-v2"
            self.tiler = TextTiler(max_doc_size=self.retriever_config.max_doc_length,
                                   stride=self.retriever_config.stride,
                                   tokenizer=tokenizer)
        return SearchData.read_data(input_files=file, tiler=self.tiler, **vars(self.retriever_config))
