from docuverse import SearchResult, SearchQueries
from docuverse.engines.retrieval import RetrievalEngine


class FileEngine(RetrievalEngine):
    """
    A simple class for reading and returning a specific set of results. Useful if one just wants to
    rerank a model on top of a particular input set.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        filename = self.config.get("db_name").replace("file:", "")
        import json
        with open(filename, "r") as inp:
            output = json.load(inp)
        self.results = [SearchResult(SearchQueries.Query(**o['question']),
                            o['retrieved_passages']) for o in output]
        self.index = 0

    def search(self, query: SearchQueries.Query, **kwargs):
        out = self.results[self.index]
        self.index += 1
        return out
