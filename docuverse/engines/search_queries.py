import json
from docuverse.engines.preprocessors import *

class SearchQueries:
    def __init__(self, preprocessor):
        self.queries = preprocessor.get_queries()

    def __iter__(self):
        return iter(self.queries)

    def __get__(self, i: int):
        return self.queries[i]