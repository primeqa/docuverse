import json
from docuverse.engines.preprocessors.base_preprocessor import BasePreprocessor

class ListPreprocessor(BasePreprocessor):
    def __init__(self, queries):
        self.queries = queries