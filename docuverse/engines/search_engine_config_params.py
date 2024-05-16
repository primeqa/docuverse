import json
import os
from typing import Optional, List

from optimum.utils.runs import RunConfig

from docuverse.utils import get_param
from dataclasses import dataclass, field
from transformers import HfArgumentParser


class GenericArguments:
    def get(self, key:str, default=None):
        return self.__dict__[key] if key in self.__dict__ else None


@dataclass
class SearchEngineArguments(GenericArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
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

    config: Optional[str] = field(
        default=None,
        metadata={
            "help": "The config file to use."
        }
    )

    hybrid: Optional[str] = field(
        default="rrf",
        metadata={
            "choices": ["rrf"],
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
        default="es-dense",
        metadata={
            "choices": ['es-dense', 'es-elser', 'es-bm25', 'es-dense'],
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    # output_file: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "The output rank file. This file will also be used to create an output filename"
    #                 "for metrics."
    #     }
    # )

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

    def __post_init__(self):
        pass


@dataclass
class EngineArguments(GenericArguments):
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
        for a in self.actions:
            action_flag = self.action_flags.get(a)
            if action_flag is not None:
                setattr(self, action_flag, True)

@dataclass
class EvaluationArguments(GenericArguments):
    compute_rouge: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If provided, will compute the ROUGE score between the answers and the gold passage "
                    "(note: it will be pretty slow for long documents)."
        }
    )

    duplicate_removal: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Defines the strategy for removing duplicates (default: don't remove). It can be 'rouge' (based on "
                    "rouge similarity) or 'exact' (exact match)"
        }
    )

    ranks: Optional[str] = field(
        default="1,3,5",
        metadata={
            "help": "Defines the R@i evaluation ranks."
        }
    )

    server: Optional[str]|None = field(
        default=None,
        metadata={
            "help": "The server to use (convai, resconvai, ailang)."
        }
    )

    iranks: List[int]|None = None

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


class RunConfig(GenericArguments):
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
            search_config (SearchEngineConfig): The search engine configuration.
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
    def __init__(self, config: dict=None):
        self.params = HfArgumentParser((SearchEngineArguments, EvaluationArguments, EngineArguments))
        self.config = config
        self.eval_config: EvaluationConfig|None = None
        self.search_config: SearchEngineConfig|None = None
        self.run_config: RunConfig|None = None

    def read_dict(self, **kwargs):
        self._process_params(self.params.parse_dict, **kwargs)

    def read_args(self):
        self._process_params(self.params.parse_args_into_dataclasses)

    def read_json(self, json_file):
        self._process_params(self.params.parse_json_file, json_file)

    def _process_params(self, parse_method, *args, **kwargs):
        self.config = kwargs
        (self.search_config, self.eval_config, self.run_config) = parse_method(*args, **kwargs)
        self.ingest_params()

    def ingest_params(self):
        for _dict in [self.search_config, self.eval_config, self.run_config]:
            for key, value in _dict.__dict__.items():
                self.__setattr__(key, value)

    @staticmethod
    def get_stdargs():
        config = DocUVerseConfig()
        config.read_args()
        return config
