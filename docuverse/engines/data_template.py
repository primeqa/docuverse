from dataclasses import dataclass, field
from typing import List, Tuple
from docuverse.utils import read_config_file

@dataclass
class DataTemplate:
    text_header: str = field(
        default="text",
        metadata={
            "help": "The header for the query text"
        }
    )
    id_header: str = field(
        default=None,
        metadata={
            "help": "The header name for the query ID"
        }
    )
    relevant_header: str = field(
        default=None,
        metadata={
            "help": "The header for the query header for relevant document ids."
        }
    )
    answers_header: str = field(
        default=None,
        metadata={
            "help": "The header for the query textual answers"
        }
    )

    keep_fields: str|None = field(
        default=None,
        metadata={
            "help": "The field names to keep in the index (comma-separated list)"
        }
    )

    extra_fields: List[str] | None = field(
        default=None,
        metadata={
            "help": "The field names to keep/add to the index"
        }
    )

    title_header: str|None = field(
        default="title",
        metadata={
            "help": "The title for the document."
        }
    )

    passage_header: str | None = field(
        default="passages",
        metadata={
            "help": "The header for passages in the document."
        }
    )

    passage_text_header: str | None = field(
        default=None,
        metadata={
            "help": "The header for the text within passage field in documents structure."
        }
    )

    passage_id_header: str | None = field(
        default=None,
        metadata={
            "help": "The header name for the passage ID within the document structure"
        }
    )

    truth_id: str|None = field(
        default="query-id",
        metadata={
            "help": "The truth id for the document (when the association id->doc_id is done separately."
        }
    )

    truth_label: str|None = field(
        default="corpus-id",
        metadata={
            "help": "The label for document containing the answer to the given question."
        }
    )

    def __post_init__(self):
        if self.keep_fields is not None:
            self.extra_fields = self.keep_fields.split("|")


def read_doc_query_format(filename: str) -> Tuple[DataTemplate, DataTemplate]:
    config = read_config_file(filename)
    if 'data_format' not in config or 'query_format' not in config:
        raise RuntimeError(f"The config file {filename} does not contain 'data_format' and 'query_format' fields.")
    return DataTemplate(**config['data_format']), DataTemplate(**config['query_format'])


default_query_template = DataTemplate(id_header='id|qid|_id',
                                      relevant_header='relevant',
                                      answers_header='answers',
                                      text_header='text|query|question',
                                      keep_fields=None)

default_data_template = DataTemplate(id_header='id',
                                     text_header="text|documents|document",
                                     keep_fields=None)

sap_data_template = DataTemplate(text_header = "document",
                                 passage_header = "text",
                                 id_header = "document_id",
                                 title_header = "title"
                                 )

beir_data_template = DataTemplate(text_header = "text",
                                  title_header="title",
                                  id_header = "_id")