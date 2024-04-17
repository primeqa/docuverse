import json

class SearchQueries:
    def __init__(self, filepaths):
        self.queries = []
        
        for filepath in filepaths:
            self.queries.extend(self.read_query_file(filepath))

    def __iter__(self):
        return iter(self.queries)

    def __get__(self, i: int):
        return self.queries[i]

    # assuming model runner format for everything for now
    def read_query_file(self, filepath):
        queries = []
        with open(filepath, mode="r", encoding='utf-8') as fp:
            for line in fp:
                queries.append(json.loads(line)['input'][-1]['text'])
        return queries