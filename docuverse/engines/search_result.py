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

    def __init__(self, data, **kwargs):
        self.results = []
        self.rouge_scorer = None

        self.read_data(data)

    def append(self, data: Dict[str:str], **kwargs):
        self.results.append(SearchResult.SearchDatum(data, **kwargs))

    def remove_duplicates(self, duplicate_removal: str = "none",
                          rouge_duplicate_threshold: float = -1.0):
        if duplicate_removal == "none" or self.results == []:
            return
        ret = SearchResult([])
        if duplicate_removal == "exact":
            seen = {self.results[0].get_text(): 1}
            ret = [self.results[0]]
            for r in self.results[1:]:
                text_ = r.get_text()
                if text_ not in seen:
                    seen[text_] = 1
                    ret.append(r)
        elif duplicate_removal == "rouge":
            from rouge_score.rouge_scorer import RougeScorer
            if self.rouge_scorer is None:
                self.rouge_scorer = RougeScorer(['rouge1', 'rougeL'],
                                                use_stemmer=True)

            for r in self.results[1:]:
                found = False
                text_ = r.get_text()
                for c in ret:
                    scr = self.rouge_scorer.score(c.get_text(), text_)
                    if scr['rougel'].fmeasure >= rouge_duplicate_threshold:
                        found = True
                        break
                if not found:
                    ret.append(r)
        return ret

    def read_data(self, data):
        if isinstance(data, list):
            d = data[0]
            if 'hits' in d and 'hits' in d['hits']:  # Looks like Elasticsearch results
                for d in data:
                    r = SearchResult.SearchDatum(d['_source'])
                    self.results.append(r)
            else:
                for d in data:
                    r = SearchResult.SearchDatum(d)
                    self.results.append(r)
        else: # Need to deal with other structures.
            return

    def as_list(self):
        return self.results

    def __iter__(self):
        return iter(self.results)
