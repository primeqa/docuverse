import json
import pickle
from typing import List, Union

import os

from copy import deepcopy

from tqdm.auto import tqdm

from docuverse.utils import (
    open_stream,
    file_is_of_type,
    parallel_process,
    ask_for_confirmation, prepare_for_save_and_backup, get_param
)
from docuverse.engines import SearchData

from docuverse.engines.search_engine_config_params import DocUVerseConfig, SearchEngineConfig
from docuverse.engines.search_result import SearchResult
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.utils.evaluation_output import EvaluationOutput
from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.reranking.bi_encoder_reranker import BiEncoderReranker
from docuverse.utils.text_tiler import TextTiler
from docuverse.utils.timer import timer


class SearchEngine:
    DEFAULT_CACHE_DIR = os.path.join(f"{os.getenv('HOME')}", ".local", "share", "elastic_ingestion")
    __name = "SearchEngine"

    def __init__(self, config_or_path: DocUVerseConfig | str = None, **kwargs):
        self.write_necessary = False
        self.config = None
        self.retriever = None
        self.reranker = None
        self.scorer = None
        self.name = get_param(kwargs, "name", self.__name)
        SearchEngine.__name = self.name
        self.tm = timer(f"{self.name}")
        self.create(config_or_path=config_or_path, **kwargs)
        self.tiler = None

    @staticmethod
    def get_name():
        return SearchEngine.__name

    # ------------------------------------------------------------------
    # Factory classmethods (PR 2 — `from_preset` is the headline surface).
    # All three converge on the existing ``__init__(config_or_path=...)``
    # path, so behavior is identical once the dict is built.
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "SearchEngine":
        """Build a ``SearchEngine`` from a named recipe.

        Example::

            engine = SearchEngine.from_preset(
                "milvus-dense",
                input_passages="data/passages.jsonl",
                input_queries="data/queries.jsonl",
            )

        Override keys may be flat (``top_k=10``), dotted
        (``**{"retriever.top_k": 10}``), or nested via a single dict
        argument (use ``from_dict`` for that case). See
        :func:`docuverse.presets.list_presets` for available recipe names.
        """
        from docuverse.presets import deep_merge_overrides, load_preset

        merged = deep_merge_overrides(load_preset(name), overrides)
        return cls(config_or_path=merged)

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "SearchEngine":
        """Build a ``SearchEngine`` from a YAML/JSON config file, with optional overrides.

        Equivalent to ``SearchEngine(config_or_path=path)`` when no overrides
        are supplied. With overrides, the file is loaded into a dict and the
        same deep-merge as ``from_preset`` is applied on top.
        """
        from docuverse.presets import deep_merge_overrides
        from docuverse.utils import read_config_file

        if not overrides:
            return cls(config_or_path=path)
        base = read_config_file(path) or {}
        if not isinstance(base, dict):
            raise ValueError(f"{path} did not parse to a dict")
        merged = deep_merge_overrides(base, overrides)
        return cls(config_or_path=merged)

    @classmethod
    def from_dict(cls, config: dict, **overrides) -> "SearchEngine":
        """Build a ``SearchEngine`` from an in-memory config dict, with optional overrides."""
        from docuverse.presets import deep_merge_overrides

        merged = deep_merge_overrides(config, overrides)
        return cls(config_or_path=merged)

    @classmethod
    def list_presets(cls) -> list[str]:
        """Return the sorted list of available preset names."""
        from docuverse.presets import list_presets

        return list_presets()

    def with_reranker(
        self, reranker_model: str, reranker_engine: str = "dense", **kwargs
    ) -> "SearchEngine":
        """Attach (or replace) a reranker on this engine, in place.

        Returns ``self`` so calls can chain::

            engine = SearchEngine.from_preset("milvus-dense").with_reranker(
                "cross-encoder/ms-marco-MiniLM-L-12-v2",
                reranker_engine="cross-encoder",
            )

        Composition avoids the 11×4 retrieval×reranker preset cross-product.
        """
        if self.config is None or self.config.reranker_config is None:
            raise RuntimeError(
                "with_reranker requires an initialized config; "
                "build the engine first then attach a reranker."
            )
        self.config.reranker_config.reranker_model = reranker_model
        self.config.reranker_config.reranker_engine = reranker_engine
        for key, value in kwargs.items():
            setattr(self.config.reranker_config, key, value)
        self.reranker = self._create_reranker()
        return self

    def create(self, config_or_path, **kwargs):
        if isinstance(config_or_path, str | dict):
            self.config = DocUVerseConfig(config_or_path)
        elif isinstance(config_or_path, DocUVerseConfig):
            self.config = config_or_path

        self.reranker = self._create_reranker()
        if self.reranker is not None:
            self.tm.add_timing("initialization::reranker")
        self.retriever = self._create_retriever()
        if self.retriever is not None:
            self.tm.add_timing("initialization::retriever")


    def _create_retriever(self) -> RetrievalEngine:
        from docuverse.utils.retrievers import create_retrieval_engine
        return create_retrieval_engine(self.config.retriever_config)

    def _create_reranker(self) -> BiEncoderReranker | None:
        if self.config.reranker_config is None or self.config.reranker_config.reranker_model is None:
            return None

        from docuverse.utils.retrievers import create_reranker_engine
        return create_reranker_engine(self.config.reranker_config)

    def ingest(self, corpus: SearchCorpus|list[SearchCorpus], **kwargs):
        self.retriever.ingest(corpus=corpus, **kwargs)

    # ------------------------------------------------------------------
    # One-call facade: index / search / evaluate. Each is a thin wrapper
    # over read_data/ingest, search, and compute_score so a new user never
    # has to plumb intermediate objects between calls.
    # ------------------------------------------------------------------

    def index(self, documents=None, update: bool = False, **kwargs) -> "SearchEngine":
        """Read, tile and ingest documents in one call.

        ``documents`` may be a file path (jsonl/tsv/csv, optionally
        compressed), a glob, a ``ds:<dataset>`` HuggingFace spec, a list of
        dicts with ``id``/``text`` (and optional extra) fields, or ``None``
        to use ``config.input_passages``. Returns ``self`` so calls chain.
        """
        corpus = self.read_data(documents)
        self.ingest(corpus, update=update, **kwargs)
        return self

    def evaluate(self, results: List[SearchResult], queries=None) -> EvaluationOutput:
        """Score ``results`` against gold judgments.

        ``queries`` defaults to the queries used in the last ``search()``
        call, falling back to ``config.input_queries``.
        """
        if queries is None:
            queries = getattr(self, "_last_queries", None)
        if queries is None:
            queries = self.read_questions()
        return self.compute_score(queries, results)

    def _coerce_queries(self, queries):
        """Accept a path, a list of strings, a list of dicts, or ready Query objects."""
        if queries is None or isinstance(queries, str):
            return self.read_questions(queries)
        if isinstance(queries, list) and queries:
            if isinstance(queries[0], str):
                return self.read_questions(
                    [{"id": str(i), "text": q} for i, q in enumerate(queries)])
            if isinstance(queries[0], dict):
                return self.read_questions(queries)
        return queries

    def has_index(self, index_name):
        return self.retriever.has_index(index_name=index_name)

    def get_retriever_info(self):
        return self.retriever.info()

    def search(self, queries: Union[SearchQueries, list[SearchQueries.Query], list[str], list[dict], str, None] = None) -> List[SearchResult]:
        queries = self._coerce_queries(queries)
        self._last_queries = queries
        self.write_necessary = False
        answers, cache_file = self.read_cache_file(extension=".retrieve.pkl.bz2")
        if answers is None:
            if len(queries) == 0:
                 print(f"No queries to search. Check {self.config.input_queries}")
            self.retriever.reconnect_if_necessary()
            # batch_query_encoding=False preserves the per-query encode path so
            # per-query latency benchmarks measure encoding inside search().
            batch_queries = getattr(self.config, "batch_query_encoding", True)
            # If the retriever can run the whole batch itself, let it: it can
            # batch-encode all queries in one GPU pass (in this process, no
            # fork-after-CUDA) and then parallelize the CPU search with threads.
            if batch_queries and hasattr(self.retriever, "search_all"):
                answers = self.retriever.search_all(
                    queries, num_threads=self.config.num_search_threads)
            else:
                # Let a retriever batch-encode all queries up front (one GPU pass)
                # instead of encoding once per query inside parallel_process.
                if batch_queries and hasattr(self.retriever, "precompute_query_embeddings"):
                    self.retriever.precompute_query_embeddings(queries)
                answers = parallel_process(self.retriever.search, queries,
                                           num_threads=self.config.num_search_threads,
                                           msg=f"Searching documents:")
            self.write_necessary = True
            self.write_cache_file(answers, cache_file)
        if self.reranker is not None:
            if self.write_necessary: # The retriever just got run, therefore force running the reranker
                ranswers, cache_file = None, self._get_cache_file(extension=".rerank.pkl.bz2")
            else:
                ranswers, cache_file = self.read_cache_file(extension=".rerank.pkl.bz2")
            if ranswers is None:
                answers = self.reranker.rerank(answers)
                self.write_necessary = True
                # cache_file.replace(".retrieve.pkl.bz2", ".rerank.pkl.bz2")
                self.write_cache_file(answers, cache_file)
            else:
                answers = ranswers
        # tm = timer.subtimer_from_top("search", default_parent="ingest_and_test")
        for answer in answers:
            answer.remove_duplicates(self.config.duplicate_removal, self.config.rouge_duplicate_threshold)
        # tm.add_timing("remove duplicates")
        return answers

    def read_cache_file(self, extension):
        answers = None
        cache_file = None
        if self.config.cache_dir is not None and not self.config.no_cache:
            # Read the results if available, don't search again
            cache_file = self._get_cache_file(extension)
            if os.path.exists(cache_file):
                r = ask_for_confirmation(f"File {cache_file} exists, read?",
                                         answers=['yes', 'no'],
                                         default='no')
                if r=='no':
                    return None, cache_file
                print(f"Reading cached search results from {cache_file}")
                try:
                    answers = self.read_output(cache_file)
                except Exception as e:
                    print(f"Failed to read cache file {cache_file}: {e}")
        return answers, cache_file

    def _get_cache_file(self, extension):
        # Strip only a trailing .json/.jsonl extension. A bare str.replace of
        # ".json" would also corrupt ".jsonl" names and any path containing
        # ".json" in the middle.
        base = os.path.basename(self.config.output_file)
        for ext in (".jsonl", ".json"):
            if base.endswith(ext):
                base = base[: -len(ext)]
                break
        return os.path.join(self.config.cache_dir, base + extension)

    def write_cache_file(self, values, cache_file):
        if cache_file is not None:
            if not os.path.exists(self.config.cache_dir):
                os.makedirs(self.config.cache_dir)
            try:
                self.tm.mark()
                self.write_output(values, cache_file)
                self.tm.add_timing("write_output")
            except Exception as e:
                print(f"Failed to write cache file {cache_file}: {e}")

    def write_output(self, output, output_file:str|None|bytes=None, overwrite=False):
        """
        Writes the output of the system, backing up any existing file first.
        Always writes: results served from cache are written too (a method
        named write_output that silently does nothing is a footgun).
        """
        import json
        if output_file is None:
            output_file = self.config.output_file
        if output_file is None:
            return
        prepare_for_save_and_backup(output_file, overwrite)
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if file_is_of_type(output_file, extensions=".json"):
            # with open(output_file, "w") as outfile:
            with open_stream(output_file, write=True) as outfile:
                outp = [r.as_dict() for r in output]
                outfile.write(json.dumps(outp, indent=2))
        elif file_is_of_type(output_file, extensions=".jsonl"):
            with open_stream(output_file, write=True) as outfile:
                for r in output:
                    outfile.write(json.dumps(r.as_dict()) + "\n")
        elif file_is_of_type(output_file, extensions=".pkl"):
            with open_stream(output_file, write=True, binary=True) as outfile:
                pickle.dump(output, outfile)


    @staticmethod
    def read_output_(filename: str, query_template) -> List[SearchResult]:
        output = None
        import orjson
        res = []
        if file_is_of_type(filename, ".json"):
            with open(filename, "r") as inp:
                try:
                    output = orjson.loads("".join(inp.readlines()))
                    res = [SearchResult(SearchQueries.Query(template=query_template, **o['question']),
                                        o['retrieved_passages']) for o in output]
                except orjson.JSONDecodeError as e:
                    print(f"Error parsing JSON file {filename}: {str(e)}")
                    raise
        elif file_is_of_type(filename, ".jsonl"):
            with open(filename, "r") as inp:
                for i, line in tqdm(enumerate(inp), desc="Reading output"):
                    try:
                        o = json.loads(line)
                        res.append(SearchResult(SearchQueries.Query(template=query_template, **o['question']),
                                                o['retrieved_passages']))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSONL line {i}, error: {str(e)}, json: {line}")
                        continue

        elif file_is_of_type(filename, ".pkl"):
            res = pickle.load(open_stream(filename, binary=True))
        return res

    def read_output(self, filename=None) -> list[SearchResult]:
        filename = self.config.output_file if filename is None else filename
        return SearchEngine.read_output_(filename, self.config.query_template)

    def set_index(self, index=None):
        pass

    def compute_score(self, queries: SearchQueries, results: List[SearchResult]) -> EvaluationOutput:
        from docuverse.utils.evaluator import EvaluationEngine
        scorer = EvaluationEngine(self.config)
        results = scorer.compute_score(queries, results, model_name=self.get_output_name())
        return results

    def read_data(self, file=None, no_cache: bool | None = None):
        if file is None:
            file = self.config.input_passages
        if self.config.db_engine in ['milvus-hybrid', 'milvus_hybrid']:
            if not self.config.hybrid['shared_tokenizer']:
                data = []
                for m in self.retriever.models:
                    tiler = self.create_tiler(m.config)
                    data.append(self._read_data(file, no_cache=no_cache,
                                                tiler=tiler,
                                                retriever_config=m.config)
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
            retriever_config = deepcopy(retriever_config)
            retriever_config.no_cache = no_cache

        if tiler is None:
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

            return TextTiler(
                **(
                    {"tokenizer":tokenizer} | retriever_config.__dict__
                ))
                             # max_doc_size=retriever_config.max_doc_length,
                             # stride=retriever_config.stride,
                             #
                             # aligned_on_sentences=retriever_config.aligned_on_sentences,
                             # count_type=retriever_config.count_type
        else:
            return self.tiler

    def read_questions(self, file=None):
        return SearchQueries.read(file if file else self.config.input_queries,
                                  **vars(self.config.retriever_config))

    def get_output_name(self):
        if self.config.output_name is None:
            return self.config.index_name
        else:
            return self.config.output_name
