import json
import csv
import re

from docuverse.engines import SearchData
from docuverse.engines.preprocessors import *
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
    def read(query_file, **kwargs):
        return SearchQueries.read_question_data(in_files=query_file, **kwargs)

    @classmethod
    def read_question_data(cls, in_files, fields=None, lang="en", remv_stopwords=False, url=None, **kwargs):
        tpassages = []
        if isinstance(in_files, str):
            in_files = [in_files]
        elif not isinstance(in_files, list):
            raise RuntimeError(f"Invalid argument 'in_files' type: {type(in_files)}")

        for in_file in in_files:
            with open(in_file, "r", encoding="utf-8") as file_stream:
                csv_reader = \
                    csv.DictReader(file_stream, fieldnames=fields, delimiter="\t") \
                        if fields is not None \
                        else csv.DictReader(file_stream, delimiter="\t")
                # next(csv_reader)
                for row in csv_reader:
                    question = get_param(row, "text|question")
                    if url is not None:
                        question = (re.sub(url, lang, 'URL', question), remv_stopwords)
                    itm = {'text': question, 'id': get_param(row, 'id|qid'),
                           'relevant': get_param(row, 'relevant|doc-id-list')}
                    # itm = {'text': (row["title"] + ' ' if 'title' in row else '') + row["text"],
                    #        'id': row['id']}
                    # if 'title' in row:
                    #     itm['title'] = row['title']
                    # itm['title'] = get_param(row, 'title|question')
                    # if 'relevant' in row:
                    #     itm['relevant'] = row['relevant'].split(",")
                    answers = get_param(row, 'answers')
                    if isinstance(answers, str):
                        if "::" in answers:
                            itm['answers'] = answers.split("::")
                        else:
                            itm['answers'] = [answers]
                    elif isinstance(answers, list):
                        itm['answers'] = answers
                    itm['passages'] = answers
                    # if 'answers' in row:
                    #     itm['answers'] = row['answers'].split("::")
                    #     itm['passages'] = itm['answers']
                    tpassages.append(SearchQueries.Query(**itm))
        return tpassages

