import json
import pickle
from multiprocessing import Pool
from typing import List, Tuple, Union

import os

from pydantic_core.core_schema import LiteralSchema
from tqdm import tqdm
from copy import deepcopy

from triton.language.extra.cuda import num_threads

from docuverse.utils import (
    open_stream,
    file_is_of_type,
    parallel_process,
    ask_for_confirmation
)
from docuverse.engines import SearchData

from docuverse.engines.search_engine_config_params import DocUVerseConfig, SearchEngineConfig
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
        self.write_necessary = False
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

    def ingest(self, corpus: SearchCorpus|list[SearchCorpus], update: bool = False):
        self.retriever.ingest(corpus=corpus, update=update)

    def has_index(self, index_name):
        return self.retriever.has_index(index_name=index_name)

    def get_retriever_info(self):
        return self.retriever.info()

    def search(self, queries: Union[SearchQueries, list[SearchQueries.Query]]) -> List[SearchResult]:
        self.write_necessary = False
        answers, cache_file = self.read_cache_file(extension=".retrieve.pkl.bz2")
        if answers is None:
            if len(queries) == 0:
                 print(f"No queries to seaarch. Check {self.config.input_queries}")
            self.retriever.reconnect_if_necessary()
            answers = parallel_process(self.retriever.search, queries,
                                       num_threads=self.config.num_search_threads,
                                       msg=f"Searching documents:")
            self.write_necessary = True
            self.write_cache_file(answers, cache_file)
        if self.reranker is not None:
            ranswers, cache_file = self.read_cache_file(extension=".rerank.pkl.bz2")
            if ranswers is None:
                answers = self.reranker.rerank(answers)
                self.write_necessary = True
                # cache_file.replace(".retrieve.pkl.bz2", ".rerank.pkl.bz2")
                self.write_cache_file(answers, cache_file)
            else:
                answers = ranswers

        return answers

    def read_cache_file(self, extension):
        answers = None
        cache_file = None
        if self.config.cache_dir is not None and not self.config.no_cache:
            # Read the results if available, don't search again
            base_cache_file = os.path.basename(self.config.output_file.replace(".json", extension))
            cache_file = os.path.join(self.config.cache_dir, base_cache_file)
            if os.path.exists(cache_file):
                r = ask_for_confirmation(f"File {cache_file} exists, read?",
                                         answers=['yes', 'no'],
                                         default='no')
                if r=='no':
                    return None, None
                print(f"Reading cached search results from {cache_file}")
                try:
                    answers = self.read_output(cache_file)
                except Exception as e:
                    print(f"Failed to read cache file {cache_file}: {e}")
        return answers, cache_file

    def write_cache_file(self, values, cache_file):
        if cache_file is not None:
            if not os.path.exists(self.config.cache_dir):
                os.makedirs(self.config.cache_dir)
            try:
                self.write_output(values, cache_file)
            except Exception as e:
                print(f"Failed to write cache file {cache_file}: {e}")

    def write_output(self, output, output_file:str|None|bytes=None, overwrite=False):
        """
        Writes the output of the system, saving copiees.
        """
        import json
        if not self.write_necessary:
            return
        if not overwrite and os.path.exists(self.config.output_file):
            # Make a copy before writing over
            import shutil
            i = 1
            template = self.config.output_file.replace(".json", "")
            while os.path.exists(f"{template}.bak{i}.json"):
                i += 1
            shutil.copy2(self.config.output_file, f"{template}.bak{i}.json")
        if output_file is None:
            output_file = self.config.output_file
        if file_is_of_type(output_file, extensions=".json"):
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as outfile:
                outp = [r.as_dict() for r in output]
                outfile.write(json.dumps(outp, indent=2))
        elif file_is_of_type(output_file, extensions=".pkl"):
            with open_stream(output_file, write=True, binary=True) as outfile:
                pickle.dump(output, outfile)

    def read_output(self, filename):
        output = None
        import orjson
        res = []
        if file_is_of_type(filename, ".json"):
            with open(filename, "r") as inp:
                output = orjson.loads("".join(inp.readlines()))
                res = [SearchResult(SearchQueries.Query(template=self.config.query_template, **o['question']),
                                    o['retrieved_passages']) for o in output]
        elif file_is_of_type(filename, ".pkl"):
            res = pickle.load(open_stream(filename, binary=True))
        return res

    def set_index(self, index=None):
        pass

    def compute_score(self, queries: SearchQueries, results: List[SearchResult]) -> EvaluationOutput:
        pass

    def read_data(self, file, no_cache: bool | None = None):
        if self.config.db_engine in ['milvus-hybrid', 'milvus_hybrid']:
            if not self.config.hybrid['shared_tokenizer']:
                data = []
                for m in self.retriever.models:
                    tiler = self.create_tiler(m.config.retriever_config)
                    data.append(self._read_data(file, no_cache=no_cache,
                                                tiler=tiler,
                                                retriever_config=m.config.retriever_config)
                                )
            else: # Use the first hybrid model
                return self._read_data(file, no_cache=no_cache,
                                       retriever_config=self.retriever.models[0].config)
        else:
            return self._read_data(file, no_cache=no_cache)
        return data

    def _read_data(self, file, no_cache: bool | None = None, tiler=None, retriever_config=None):
        if retriever_config is None:
            retriever_config = self.config.retriever_config
        if no_cache is not None:
            retriever_config = deepcopy(self.config.retriever_config)
            retriever_config.no_cache = no_cache

        tiler = self.tiler if self.tiler is not None else self.create_tiler(retriever_config)
        return SearchData.read_data(input_files=file,
                                    tiler=tiler,
                                    **vars(retriever_config))

    def create_tiler(self, retriever_config):
        if self.tiler is None:
            tokenizer = None
            if getattr(self.retriever, 'model', None) is not None:
                tokenizer = self.retriever.model.tokenizer
            else:
                tokenizer = retriever_config.model_name
                if tokenizer == "" or tokenizer.startswith("."):
                    tokenizer = "sentence-transformers/all-MiniLM-L6-v2"
            return TextTiler(max_doc_size=retriever_config.max_doc_length,
                                   stride=retriever_config.stride,
                                   tokenizer=tokenizer,
                                   aligned_on_sentences=retriever_config.aligned_on_sentences,
                                   count_type=retriever_config.count_type)
        else:
            return self.tiler

    def read_questions(self, file):
        return SearchQueries.read(file, **vars(self.config.retriever_config))

    def get_output_name(self):
        if self.config.output_name is None:
            return self.config.index_name
        else:
            return self.config.output_name
