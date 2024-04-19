import json

class BasePreprocessor:
    def __init__(self):
        self.queries = []

    def get_queries(self):
        return self.queries