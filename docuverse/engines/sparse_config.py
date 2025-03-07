from dataclasses import dataclass, field

@dataclass
class SparseConfig:
    doc_max_tokens: int = field(
        default=192,
        metadata={
            "help": "The maximum number of tokens to allow in the document sparse representation"
        }
    )

    query_max_tokens: int = field(
        default=50,
        metadata={
            "help": "The maximum number of tokens to allow in the query sparse representation"
        }
    )

    sparse_build_config: str = field(
        default="default",
        metadata={
            "help": "The build configuration to use when building the sparse data ingesting"
        }
    )

    sparse_search_config: str = field(
        default="SPLADE",
        metadata={
            "help": "The search configuration to use when building the sparse query search"
        }
    )

    runtime_query_encoding: str = field(
        default=True,
        metadata={
            "help": "Whether or not to run the query encoding on sparse models (default: True). If False,"
                    "the query will be comprised of the tokens with a weight of 1."
        }
    )