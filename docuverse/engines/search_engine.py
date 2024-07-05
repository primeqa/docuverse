from typing import List, Tuple

import yaml
import os

from tqdm import tqdm

import docuverse.utils
from docuverse.engines import SearchData
from docuverse.engines.search_engine_config_params import DocUVerseConfig
from docuverse.engines.search_result import SearchResult
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.utils.evaluation_output import EvaluationOutput
from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.reranking.reranker import Reranker
from docuverse.utils.text_tiler import TextTiler


class SearchEngine:
    DEFAULT_CACHE_DIR = os.path.join(f"{os.getenv('HOME')}", ".local", "share", "elastic_ingestion")

    def __init__(self, config_or_path: DocUVerseConfig | str = None, **kwargs):
        self.config = None
        self.retriever = None
        self.reranker = None
        self.scorer = None
        self.create(config_or_path=config_or_path)
        self.tiler = None

    # @staticmethod
    # def read_configs(config_or_path: str) -> GenericArguments:
    #     if isinstance(config_or_path, str):
    #         if os.path.exists(config_or_path):
    #             with open(config_or_path) as stream:
    #                 try:
    #                     vals = yaml.safe_load(stream=stream)
    #                     if 'retrieval' not in vals: # By default, all parameters are assumed to be retriever params
    #                         vals['retrieval'] = vals
    #                         vals['reranker'] = None
    #                 except yaml.YAMLError as exc:
    #                     raise exc
    #         else:
    #             print(f"The configuration file '{config_or_path}' does not exist.")
    #             raise FileNotFoundError(f"The configuration file '{config_or_path}' does not exist.")
    #     elif isinstance(config_or_path, GenericArguments):
    #         return config_or_path

    def create(self, config_or_path, **kwargs):
        if isinstance(config_or_path, str | dict):
            self.config = DocUVerseConfig(config_or_path)
        elif isinstance(config_or_path, DocUVerseConfig):
            self.config = config_or_path

        self.reranker = self._create_reranker()
        self.retriever = self._create_retriever()

    def _create_retriever(self) -> RetrievalEngine:
        from docuverse.utils.retrievers import create_retrieval_engine
        return create_retrieval_engine(self.config.retriever_config)

    def _create_reranker(self) -> Reranker | None:
        if self.config.reranker_config is None or self.config.reranker_config.reranker_model is None:
            return None

        from docuverse.utils.retrievers import create_reranker_engine
        return create_reranker_engine(self.config.reranker_config)

    def ingest(self, corpus: SearchCorpus, update: bool = False):
        self.retriever.ingest(corpus=corpus, update=update)

    def get_retriever_info(self):
        return self.retriever.info()

    def search(self, queries: SearchQueries) -> List[SearchResult]:
        answers = [self.retriever.search(query) for query in tqdm(queries, desc="Processing queries: ")]
        if self.reranker is not None:
            answers = [self.reranker.rerank(answer) for answer in tqdm(answers, desc="Reranking queries: ")]
        return answers

    def write_output(self, output, overwrite=False):
        """
        Writes the output of the system, saving copiees.
        """
        import json
        if not overwrite and os.path.exists(self.config.output_file):
            # Make a copy before writing over
            import shutil
            i = 1
            template = self.config.output_file.replace(".json", "")
            while os.path.exists(f"{template}.bak{i}.json"):
                i += 1
            shutil.copy2(self.config.output_file, f"{template}.bak{i}.json")
        with open(self.config.output_file, "w") as outfile:
            outp = [r.as_list() for r in output]
            outfile.write(json.dumps(outp, indent=2))

    def set_index(self, index=None):
        pass

    def compute_score(self, queries: SearchQueries, results: List[SearchResult]) -> EvaluationOutput:
        pass

    def read_data(self, file):
        retriever_config = self.config.retriever_config
        if self.tiler is None:
            tokenizer = None
            if getattr(self.retriever, 'model', None) is not None:
                tokenizer = self.retriever.model.tokenizer
            else:
                tokenizer = retriever_config.model_name
                if tokenizer == "" or tokenizer.startswith("."):
                    tokenizer = "sentence-transformers/all-MiniLM-L6-v2"
            self.tiler = TextTiler(max_doc_size=retriever_config.max_doc_length,
                                   stride=retriever_config.stride,
                                   tokenizer=tokenizer,
                                   aligned_on_sentences=retriever_config.aligned_on_sentences,
                                   count_type=retriever_config.count_type)
        return SearchData.read_data(input_files=file,
                                    tiler=self.tiler,
                                    **vars(retriever_config))

    def read_questions(self, file):
        return SearchQueries.read(file, **vars(self.config.retriever_config))
