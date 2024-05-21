import json

from docuverse.engines import SearchData
from docuverse.engines.preprocessors import *


class SearchQueries(SearchData):
    def __init__(self, preprocessor, filenames, **data):
        super().__init__(filenames, **data)
        self.queries = preprocessor.get_queries()

    def __iter__(self):
        return iter(self.queries)

    def __getitem__(self, i: int):
        return self.queries[i]

    @staticmethod
    def read(query_file, **kwargs):
        return SearchData.read_question_data(in_files=query_file, **kwargs)
