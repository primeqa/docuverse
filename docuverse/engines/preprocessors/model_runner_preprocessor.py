import json
from docuverse.engines.preprocessors.base_preprocessor import BasePreprocessor


class ModelRunnerPreprocessor(BasePreprocessor):
    def __init__(self, filepaths):
        super().__init__()
        self.queries = []

        for filepath in filepaths:
            self.queries.extend(self.read_query_file(filepath))

    # assuming model runner format for everything for now
    def read_query_file(self, filepath):
        queries = []
        with open(filepath, mode="r", encoding='utf-8') as fp:
            for line in fp:
                queries.append(json.loads(line)['input'][-1]['text'])
        return queries
