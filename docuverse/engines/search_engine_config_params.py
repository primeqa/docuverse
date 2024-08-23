import json
import os
from typing import Optional, List, Literal

from optimum.utils.runs import RunConfig, Run

from docuverse.utils import read_config_file
from docuverse.engines.retrieval.search_filter import SearchFilter
from docuverse.utils import get_param
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from docuverse.engines.data_template import (
    DataTemplate,
    default_query_template,
    default_data_template,
    read_doc_query_format
)


class GenericArguments:
    def get(self, key: str, default=None):
        return self.__dict__[key] if key in self.__dict__ else default

    def __getitem__(self, item, default=None):
        return self.get(item, default)


@dataclass
class RetrievalArguments(GenericArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    project_dir: str = field(
        default=None,
        metadata={
            "help": "The root dir for the configuration files."
        }
    )
    _argument_group_name = "Search Arguments"

    model_name: str = field(
        default="",
        metadata={"help": "Pre-trained model name or path if not the same as model_name"}
    )

    input_passages: Optional[str] | None = field(
        default=None,
        metadata={
            "nargs": "+",
            "help": "The input passages, if any (can be empty for retrieval only)."
        }
    )

    input_queries: Optional[str] | None = field(
        default=None,
        metadata={
            "help": "The test queries, if any (can be empty for ingestion only)."
        }
    )

    hybrid: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["rrf", "none"],
            "help": "The type of hybrid combination to use (default is RRF)."
        }
    )

    top_k: Optional[str] = field(
        default=10,
        metadata={
            "help": "The maximum number of results to return."
        }
    )

    index_name: str = field(
        default=None,
        metadata=
        {
            "help": "Defines the index name to use. If not specified, it is built as "
                    "{args.data}_{args.db_engine}_{args.model_name if args.db_engine=='es-dense' else 'elser'}_index"}
    )

    db_engine: Optional[str] = field(
        default="es-bm25",
        metadata={
            "choices": ['es-dense', 'es-elser', 'es-bm25', 'es-dense'],
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )

    model_on_server: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If present, the given model is assumed to exist on the ES server."
        }
    )

    doc_based: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If present, the document text will be ingested, otherwise the ingestion will be done"
                    " at passage level."
        }
    )

    max_doc_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "If provided, the documents will be split into chunks of <max_doc_length> "
                    "*word pieces* (in regular English text, about 2 word pieces for every word). "
                    "If not provided, the passages in the file will be ingested truncated at 512 tokens."
        }
    )

    stride: Optional[int] = field(
        default=None,
        metadata={
            "help": "Argument that works in conjunction with --max_doc_length: it will define the "
                    "increment of the window start while tiling the documents."
        }
    )

    aligned_on_sentences: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If present, the documents will be split into tiles such that they break only "
                    "in-between sentences."
        }
    )

    tile_overlap: Optional[int] = field(
        default=None,
        metadata={
            "help": "The counterargument to stride - it's defined as max_doc_length-stride. Sometimes"
                    "it's easier to think in terms of overlap than in terms of stride."
        }
    )

    ingestion_batch_size: Optional[int] = field(
        default=40,
        metadata={
            "help": "For elastic search only, sets the ingestion batch size (default 40)."
        }
    )

    title_handling: Optional[str] = field(
        default="all",
        metadata={
            "help": "Defines the policy of adding titles to the passages: can be 'all', 'first', or 'none'."
                    "'all' will add the document title to every tile, 'first' will add to the first split tile, "
                    "and 'none' will not add it at all."
        }
    )

    count_type: Literal['char', 'token'] = field(
        default="token",
        metadata={
            "help": "Defines what the measure for max_doc_length and stride are - can be either 'token' or 'char'. (default 'token')."
        }
    )

    text_field: Optional[str] = field(
        default="text",
        metadata={
            "help": "Defines the text field name in the document json line (default: 'text')."
        }
    )

    title_field: Optional[str] = field(
        default="title",
        metadata={
            "help": "Defines the title field name in the document json line (default: 'title')."
        }
    )

    filters: Optional[str] | None = field(
        default=None,
        metadata={
            "help": "Defines the fields to filter on when searching with ElasticSearch (default: None)."
        }
    )

    filter_on: Optional[List[SearchFilter]] | None = field(
        default=None,
        metadata={
            "help": "Specifies a map from question attributes to document attributes for filtering."
        }
    )

    server: Optional[str] | None = field(
        default=None,
        metadata={
            "help": "The server to use (convai, resconvai, ailang)."
        }
    )

    lang: Optional[str] = field(
        default="en",
        metadata={
            "help": "The language of the documents (default: en)."
        }
    )

    max_num_documents: Optional[str] | None = field(
        default=None,
        metadata={
            "help": "The maximum number of documents to ingest (for testing purposes)."
        }
    )

    bulk_batch: Optional[int] = field(
        default=40,
        metadata={
            "help": "If provided, the documents will be ingested with the provided bulk batch size."
        }
    )

    num_candidates: Optional[int] = field(
        default=1000,
        metadata={
            "help": "If provided, it specifies the provided number of candidates nodes to "
                    "search in the HNSW algorithm."
        }
    )

    num_preprocessor_threads: Optional[int] = field(
        default=-1,
        metadata={
            "help": "If provided, it specifies the number of threads to use for preprocessing data"
        }
    )

    num_search_threads: Optional[int] = field(
        default=1,
        metadata={
            "help": "If provided, it will search with multiple threads."
        }
    )

    no_cache: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If provided, the cache will be ignored when reading the documents."
        }
    )

    duplicate_removal: Optional[bool]|None = field(
        default=None,
        metadata={
            "help": "Defines the strategy for removing duplicates (default: don't remove). It can be 'rouge' (based on "
                    "rouge similarity) or 'exact' (exact match)"
        }
    )

    rouge_duplicate_threshold: Optional[float] = field(
        default=0.9,
        metadata={
            "help": "Defines the threshold for the rouge score when removing duplicates."
        }
    )

    data_format: Optional[str] | None = field(
        default=None,
        metadata={
            "help": "Defines the configuration file associated with the data and query format of the documents/queries "
                    "(default: None)."
        }
    )

    query_header_format: Optional[str] = field(
        default=None,
        metadata={
            "help": "Defines the query header format (default: None; it should have the following fields defined:"
                    "* id_header: str"
                    "* text_header: str"
                    "* relevant_header: str"
                    "* answers_header: str"
                    "* keep_fields: str"
                    "Example: 'query_id_header:id,query_text_header:text,query_relevant_header:relevant_docs,query_answers_header:text_answers')."
        }
    )

    data_header_format: Optional[str] = field(
        default=None,
        metadata={
            "help": "Defines the data header (default: None; it should have the following fields defined:"
                    "* id_header: str"
                    "* text_header: str"
                    "* keep_fields: str"
                    "Example: 'query_id_header:id,query_text_header:text,query_relevant_header:relevant_docs,query_answers_header:text_answers')."
        }
    )

    verbose: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If provided, information about statistics on the data read will be displayed."
        }
    )

    query_template: DataTemplate = default_query_template
    data_template: DataTemplate = default_data_template

    def __post_init__(self):
        # parse the query_header_template

        if self.data_format is not None:
            self.data_template, self.query_template = read_doc_query_format(self.data_format)
        else:
            def create_template(param):
                items = param.split(',')
                val = {}
                for item in items:
                    a = item.split(':')
                    if len(a) == 2:
                        val[a[0]] = a[1]
                return DataTemplate(**val)

            if self.query_header_format is not None:
                self.query_template = create_template(self.query_header_format)

            if self.data_header_format is not None:
                self.data_template = create_template(self.data_header_format)
        if self.filter_on is not None:
            res = []
            for name, _filter in self.filter_on.items():
                f = SearchFilter(name=name, **_filter)
                if f.query_field not in self.query_template.extra_fields:
                    self.query_template.extra_fields.append(f.query_field)
                if f.document_field not in self.data_template.extra_fields:
                    self.data_template.extra_fields.append(f.document_field)
                res.append(f)
            self.filter_on = res


@dataclass
class EngineArguments(GenericArguments):
    _argument_group_name = "Engine Arguments"
    output_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The output rank file. This file will also be used to create an output filename for the metrics, "
                    "if specified."
        }
    )
    actions: Optional[str] = field(
        default="ir",
        metadata={
            "help": "The actions that can be done: i(ingest), r(retrieve), R(rerank), u(update), e(evaluate)"
        }
    )

    config: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration file associated with the configuration; command line arguments will override"
                    "values in this configuration."
        }

    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The cache directory to use for storing intermediate results."
        }
    )

    action_flags = {
        "i": "ingest",
        "u": "update",
        "r": "retrieve",
        "e": "evaluate",
        "R": "rerank"
    }

    ingest: Optional[bool] = False
    update: Optional[bool] = False
    retrieve: Optional[bool] = False
    evaluate: Optional[bool] = False
    rerank: Optional[bool] = False

    def __post_init__(self):
        for _, val in self.action_flags.items():
            setattr(self, val, False)
        for a in self.actions:
            action_flag = self.action_flags.get(a)
            if action_flag is not None:
                setattr(self, action_flag, True)


@dataclass
class RerankerArguments(GenericArguments):
    _argument_group_name = "Reranker Arguments"
    reranker_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model to use for reranking - can be a HuggingFace model name or a path."
        }
    )

    reranker_batch_size: Optional[int] = field(
        default=32,
        metadata={
            "help": "The batch size to use for reranking."
        }
    )

    reranker_gpu_batch_size: Optional[int] = field(
        default=128,
        metadata={
            "help": "The gpu batch size to use."
        }
    )

    reranker_combine_weight: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The weight to use in combining the previous scores with the current scores (default is 1.0 "
                    "- only use the reranker scores, 0.0 is keep the original scores, ignore the reranker)."
        }
    )

    reranker_combination_type: Literal['rrf', 'weight'] = field(
        default="rrf",
        metadata={
            "help": "The combination type to use for reranking."
        }
    )

    reranker_engine: Literal["dense", "splade", "none"] = field(
        default="dense",
        metadata={
            "help": "The model type to use for reranking."
        }
    )


@dataclass
class EvaluationArguments(GenericArguments):
    _argument_group_name = "Evaluation Arguments"

    compute_rouge: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If provided, will compute the ROUGE score between the answers and the gold passage "
                    "(note: it will be pretty slow for long documents)."
        }
    )

    ranks: Optional[str] = field(
        default="1,3,5",
        metadata={
            "help": "Defines the R@i evaluation ranks."
        }
    )

    iranks: List[int] | None = None

    eval_measure: Optional[str] = field(
        default="match",
        metadata={
            "help": "Defines the evaluation measure to use (default: match). Can be 'match' or 'ndcg' "
                    "- more to be added"
        }
    )

    def __post_init__(self):
        self.iranks = [int(r) for r in self.ranks.split(",")]


@dataclass
class SearchEngineConfig:
    index: str
    title_field: str
    text_field: str
    productId_field: str
    n_docs: int
    search_type: str
    fields: list
    filters: dict
    duplicate_removal: str
    rouge_duplicate_threshold: float
    hidden_dim: float = 384

    def __init__(self, config: dict):
        self.index = get_param(config, 'index|index_name')
        self.title_field = get_param(config, 'title_field', None)
        self.text_field = get_param(config, 'text_field', None)
        self.productId_field = get_param(config, 'productId_field', None)
        self.fields = get_param(config, 'fields', None)
        self.n_docs = get_param(config, 'n_docs|top_k', 30)
        self.search_type = get_param(config, 'search-type', None)
        self.filters = get_param(config, 'filters', None)
        # "Defines the strategy for removing duplicates (default: don't remove). It can be 'rouge' (based on rouge similarity) or 'exact' (exact match)")
        # choices=["none", "rouge", "exact"], default="none",
        self.duplicate_removal = get_param(config, 'duplicate_removal', None)
        # "The rouge-l F1 similarity for dropping duplicated in the result (default 0.7)"
        # default=-1, type=float,
        self.rouge_duplicate_threshold = get_param(config, 'rouge_duplicate_threshold', 0.7)
        self.model_on_server = config.get("model-on-server", False)
        self.normalize_embeddings = config.get("normalize-embeddings", True)
        self.data_type = config.get("data-type", "auto")
        self.ingestion_batch_size = config.get("ingestion-batch-size", 40)
        self.stride = config.get("stride", 100)
        self.max_num_documents = config.get("max-num-documents", None)
        self.max_doc_length = int(config.get("max-doc-length")) if 'max-doc-length' in config else None
        self.doc_based = config.get("doc-based", False)
        self.db_engine = config.get("db-engine", "es-dense")
        self.server = config.get("server", None)
        self.data_template = config.get("data_template", None)
        self.query_template = config.get("query_template", None)


class RunConfig(GenericArguments):
    pass


class RerankerConfig(RerankerArguments):
    pass


class EvaluationConfig:
    def __init__(self, config, **kwargs):
        data = {}
        if isinstance(config, str):
            if os.path.exists(config):
                if config.endswith(".json"):
                    data = json.load(open(config))
            if 'evaluation' in data:
                data = data['evaluation']
        elif isinstance(config, EvaluationArguments):
            data = vars(config)
        elif isinstance(config, dict):
            if 'evaluation' in config:
                data = config['evaluation']
            else:
                data = config

        data.update(kwargs)

        self.__dict__.update(data)


class DocUVerseConfig(GenericArguments):
    """
    class DocUVerseConfig:
        Represents the configuration for DocUVerse.

        Attributes:
            params (HfArgumentParser): The argument parser for parsing the configuration.
            eval_config (EvaluationConfig): The evaluation configuration.
            retriever_config (SearchEngineConfig): The search engine configuration.
            run_config (RunConfig): The run configuration.

        Methods:
            __init__(self, config: dict): Initializes the DocUVerseConfig object.
            read_dict(self, **kwargs): Reads the configuration from a dictionary.
            read_args(self): Reads the configuration from command line arguments.
            read_json(self, json_file): Reads the configuration from a JSON file.
            _process_params(self, parse_method, *args, **kwargs): Processes the parameters using the specified parse method.
            ingest_params(self): Ingests the parameters from the parsed configurations.
            get_stdargs(): Returns the standard arguments configuration.

        """

    def __init__(self, config: dict | str = None):
        self.evaluate = None
        self.output_file = None
        self.input_queries = None
        self.retrieve = None
        self.input_passages = None
        self.update = None
        self.ingest = None
        self.params = HfArgumentParser((RetrievalArguments, RerankerArguments, EvaluationArguments, EngineArguments))
        if isinstance(config, str | dict):
            self.read_configs(config)
        else:
            self.reranker_config: RerankerConfig | None = None
            self.eval_config: EvaluationConfig | None = None
            self.retriever_config: SearchEngineConfig | None = None
            self.run_config: RunConfig | None = None

    def read_dict(self, kwargs):
        self._process_params(self.params.parse_dict, kwargs, allow_extra_keys=True)

    def read_args(self):
        self._process_params(self.params.parse_args_into_dataclasses, return_remaining_strings=True)

    def read_json(self, json_file):
        self._process_params(self.params.parse_json_file, json_file)

    def _process_params(self, parse_method, *args, **kwargs):
        # self.config = kwargs
        result = parse_method(*args, **kwargs)
        (self.retriever_config, self.reranker_config, self.eval_config, self.run_config) = (
            result[0], result[1], result[2], result[3]
        )

        self.ingest_params()

    def ingest_params(self):
        for _dict in [self.retriever_config, self.reranker_config, self.eval_config, self.run_config]:
            for key, value in _dict.__dict__.items():
                self.__setattr__(key, value)

    def read_configs(self, config_or_path: str) -> GenericArguments:
        if isinstance(config_or_path, str):
            if os.path.exists(config_or_path):
                try:
                    vals = read_config_file(config_or_path)
                    self._flatten_and_read_dict(vals)
                except Exception as exc:
                    raise exc
            else:
                print(f"The configuration file '{config_or_path}' does not exist.")
                raise FileNotFoundError(f"The configuration file '{config_or_path}' does not exist.")
        elif isinstance(config_or_path, dict):
            self._flatten_and_read_dict(config_or_path)
        elif isinstance(config_or_path, GenericArguments):
            return config_or_path

    def _flatten_and_read_dict(self, vals):
        if get_param(vals, "retrieval|retriever"):  # By default, all parameters are assumed to be retriever params
            vals1 = {}
            for k, v in vals.items():
                if v and v != "None":
                    vals1.update(v)
            vals = vals1
        self.read_dict(vals)

    def update(self, other_config):
        if isinstance(other_config, DocUVerseConfig):
            DocUVerseConfig._update(self.retriever_config, other_config.retriever_config)
            DocUVerseConfig._update(self.reranker_config, other_config.reranker_config)
            DocUVerseConfig._update(self.eval_config, other_config.eval_config)
            DocUVerseConfig._update(self.run_config, other_config.run_config)
            self.ingest_params()

    @staticmethod
    def _update(output_class, input_class, default):
        if input_class is not None:
            for key, value in input_class.__dict__.items():
                if value != default.__dict__[key]:
                    output_class.__dict__[key] = value

    default_retriever_config = RetrievalArguments()
    default_reranker_config = RerankerArguments()
    default_eval_config = EvaluationArguments()
    default_run_config = EngineArguments()

    @staticmethod
    def get_stdargs_config():
        config = DocUVerseConfig()
        config.read_args()
        if get_param(config.run_config, 'config'):
            config1 = DocUVerseConfig(config.run_config.config)
            # config1.update(config)
            DocUVerseConfig._update(config1.retriever_config, config.retriever_config,
                                    DocUVerseConfig.default_retriever_config)
            DocUVerseConfig._update(config1.reranker_config, config.reranker_config,
                                    DocUVerseConfig.default_reranker_config)
            DocUVerseConfig._update(config1.eval_config, config.eval_config, DocUVerseConfig.default_eval_config)
            DocUVerseConfig._update(config1.run_config, config.run_config, DocUVerseConfig.default_run_config)
            config = config1
            config.ingest_params()
        if config.retriever_config.num_preprocessor_threads > 1:
            os.environ['TOKENIZERS_PARALLELISM'] = "true"
        return config
