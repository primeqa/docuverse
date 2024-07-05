import json
import csv
import re

from docuverse.engines import SearchData
from docuverse.engines.data_template import default_query_template
from docuverse.utils import get_param


class SearchQueries(SearchData):
    class Query:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __getitem__(self, key: str, default=None):
            return getattr(self, key, default)

        def __setattr__(self, key, value):
            setattr(self, key, value)

        def as_list(self):
            return self.__dict__

    def __init__(self, preprocessor, filenames, **data):
        super().__init__(filenames, **data)
        self.queries = preprocessor.get_queries()

    def __iter__(self):
        return iter(self.queries)

    def __getitem__(self, i: int):
        return self.queries[i]

    @staticmethod
    def read(query_file, template=default_query_template, **kwargs):
        return SearchQueries.read_question_data(in_files=query_file, template=template, **kwargs)

    @classmethod
    def read_question_data(cls, in_files, fields=None, lang="en",
                           remv_stopwords=False,
                           url=None,
                           query_template=default_query_template,
                           **kwargs):
        tpassages = []
        if isinstance(in_files, str):
            in_files = [in_files]
        elif not isinstance(in_files, list):
            raise RuntimeError(f"Invalid argument 'in_files' type: {type(in_files)}")

        for in_file in in_files:
            with open(in_file, "r", encoding="utf-8") as file_stream:
                delim = "," if ".csv" in in_file else "\t"
                csv_reader = \
                    csv.DictReader(file_stream, fieldnames=fields, delimiter=delim) \
                        if fields is not None \
                        else csv.DictReader(file_stream, delimiter=delim)
                # next(csv_reader)
                for it, row in enumerate(csv_reader):
                    question = get_param(row, query_template.text_header)
                    if url is not None:
                        question = (re.sub(url, lang, 'URL', question), remv_stopwords)
                    itm = {'text': question,
                           'id': get_param(row, query_template.id_header, str(it)),
                           'relevant': get_param(row, query_template.relevant_header).split(",")
                           }
                    for key in query_template.extra_fields:
                        if key in row:
                            itm[key] = get_param(row, key)
                    answers = get_param(row, query_template.answers_header, "")
                    if isinstance(answers, str):
                        if "::" in answers:
                            itm['answers'] = answers.split("::")
                        else:
                            itm['answers'] = [answers]
                    elif isinstance(answers, list):
                        itm['answers'] = answers
                    itm['passages'] = answers

                    tpassages.append(SearchQueries.Query(**itm))
        return tpassages
