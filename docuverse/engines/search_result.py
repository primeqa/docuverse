from typing import List, Dict
class SearchResult:
    """
    Represents the list of relevant text units (documents or paraphaphs) returned from a seqrch query.
    It's meant to be a soft wrapper around a list of dictionaries.
    """
    class SearchDatum:
        def __init__(self, data: Dict[str:str], **kwargs):
            self.__dict__.update(data)
        def __get__(self, key, default=None):
            if key in self.__dict__:
                return self.__dict__[key]
            else:
                return default

    def __init__(self, **kwargs):
        self.results = []
    def add(self, data: Dict[str:str], **kwargs):
        self.results.append(SearchResult.SearchDatum(data, **kwargs))

    def as_list(self):
        return self.results

    def __iter__(self):
        return iter(self.results)
