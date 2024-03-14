
class SearchQueries:
    def __init__(self, filepath):
        self.queries = []
        pass

    def __iter__(self):
        return iter(self.queries)

    def __get__(self, i: int):
        return self.queries[i]

    def read_query_file(self, filepath):
        q = []
        return q