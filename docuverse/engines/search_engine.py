import json
from typing import List, Tuple, Union

import os

from tqdm import tqdm
from copy import deepcopy

from docuverse.utils import get_param
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
        self.create(config_or_path=config_or_path, **kwargs)
        self.tiler = None

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

    def load_model_config(self, config_params: Union[dict, SearchEngineConfig]):
        if isinstance(config_params, dict):
            config_params = SearchEngineConfig(config=config_params)

        # Elastic doesn't accept _ -> convert them to dashes.
        if config_params.index_name:
            config_params.index_name = config_params.index_name.replace("_", "-")
        PARAM_NAMES = ["index_name", "title_field", "text_field", "n_docs", "filters", "duplicate_removal",
                       "rouge_duplicate_threshold"]

        for param_name in PARAM_NAMES:
            setattr(self, param_name, get_param(config_params, param_name))

        self.config = config_params

    def ingest(self, corpus: SearchCorpus, update: bool = False):
        self.retriever.ingest(corpus=corpus, update=update)

    def has_index(self, index_name):
        return self.retriever.has_index(index_name=index_name)

    def get_retriever_info(self):
        return self.retriever.info()

    def search(self, queries: SearchQueries) -> List[SearchResult]:
        # self.retriever.init_client()
        self.retriever.reconnect_if_necessary()
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

    def read_output(self, filename):
        import json

        with open(filename, "r") as inp:
            output = json.load(inp)
        res = [SearchResult(o['question'], o['retrieved_passages']) for o in output]
        return res

    def set_index(self, index=None):
        pass

    def compute_score(self, queries: SearchQueries, results: List[SearchResult]) -> EvaluationOutput:
        pass

    def read_data(self, file, no_cache: bool | None = None):
        if no_cache is not None:
            retriever_config = deepcopy(self.config.retriever_config)
            retriever_config.no_cache = no_cache
        else:
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
