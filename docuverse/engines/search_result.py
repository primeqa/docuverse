from typing import List, Dict, Union
from tqdm import tqdm


class SearchResult:
    """
    Represents the list of relevant text units (documents or paraphaphs) returned from a seqrch query.
    It's meant to be a soft wrapper around a list of dictionaries.
    """

    class SearchDatum:
        def __init__(self, data: Dict[str,str], **kwargs):
            self.__dict__.update(data)

        def __get__(self, key, default=None):
            if key in self.__dict__:
                return self.__dict__[key]
            else:
                return default
        
        def get_text(self):
            if '_text' in self.__dict__:
                return self._text
            elif '_source' in self.__dict__:
                return self._source['text']
            else:
                raise Exception("TODO: Pending implementation")
        
        def __str__(self) -> str:
            datum_str = ""
            for key, value in self.__dict__.items():
                datum_str = f"{datum_str}Key: {key}\tValue: {value}\n"
            return datum_str

    def __init__(self, data, **kwargs):
        self.results = []
        self.rouge_scorer = None

        self.read_data(data)

    def __len__(self):
        return len(self.results)

    def append(self, data: Union[Dict[str, str], SearchDatum], **kwargs):
        if isinstance(data, SearchResult.SearchDatum):
            self.results.append(data)
        else:
            self.results.append(SearchResult.SearchDatum(data, **kwargs))

    def remove_duplicates(self, duplicate_removal: str = "none",
                          rouge_duplicate_threshold: float = -1.0):
        if duplicate_removal is None or duplicate_removal == "none" or self.results == []:
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
                    if scr['rougeL'].fmeasure >= rouge_duplicate_threshold:
                        found = True
                        break
                if not found:
                    ret.append(r)
        return ret

    def __getitem__(self, i: int) -> SearchDatum:
        return self.results[i]

    def __iter__(self):
        return iter(self.results)

    def read_data(self, data):
        if isinstance(data, dict):
            if 'hits' in data and 'hits' in data['hits']:  # Looks like Elasticsearch results
                for d in tqdm(data['hits']['hits']):
                    # TODO - Do we need to store metadata or just the source text is ok?
                    # {'_index': 'jatin-testing', '_id': '1231', '_score': 0.3945668, '_ignored': ['text.keyword'], '_source': {'text': '\nRed Hat Enterprise Linux 4- all architectures Red Hat Enterprise Linux 5- all archit...Hat Enterprise Linux 6- all architectures '}}
                    # r = SearchResult.SearchDatum(d['_source'])
                    r = SearchResult.SearchDatum(d)
                    self.results.append(r)
            else:
                raise Exception("TODO: Pending implementation")
                # for d in data:
                #     r = SearchResult.SearchDatum(d)
                #     self.results.append(r)
        elif isinstance(data, list):
            if len(data) == 0:
                return []
            elif data[0].__class__ == SearchResult.SearchDatum:
                for item in data:
                    self.results.append(item)
            else: # Need to deal with other structures.
                raise Exception("TODO: Pending implementation")
        else: # Need to deal with other structures.
            raise Exception("TODO: Pending implementation")
            return

    def as_list(self):
        return self.results

    def __iter__(self):
        return iter(self.results)
