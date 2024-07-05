from dataclasses import field, dataclass
from typing import Optional


@dataclass
class SearchFilter:
    name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the search filter."
        }
    )

    query_field: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the query field to filter on."
        }
    )

    document_field: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the document field to filter on."
        }
    )

    type: str = field(
        default="should",
        metadata={
            "help": "The type of the search filter (can be 'should' or 'must')."
        }
    )
