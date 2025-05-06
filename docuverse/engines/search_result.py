import json
from typing import List, Dict, Union

import pymilvus

from docuverse.engines.search_queries import SearchQueries
from docuverse.utils import get_param


class SearchResult:
    """
    Represents the list of relevant text units (documents or paraphaphs) returned from a seqrch query.
    It's meant to be a soft wrapper around a list of dictionaries.
    """

    class SearchDatum:
        def __init__(self, data: Dict[str,str], **kwargs):
            self.__dict__.update(data)
            self.__dict__.update(kwargs)
            for k, v in self.__dict__.items():
                # Milvus will convert deep json trees into strings,
                # so we're # undoing that here
                if isinstance(v, str):
                    try:
                        r = json.loads(v)
                        setattr(self, k, r)
                    except ValueError as e:
                        pass
        def __getitem__(self, key, default=None):
            if key in self.__dict__:
                return self.__dict__[key]
            else:
                return default
        
        def get_text(self):
            if '_text' in self.__dict__:
                return self._text
            elif 'text' in self.__dict__:
                return self.text
            elif '_source' in self.__dict__:
                return self._source['text']
            else:
                raise Exception("TODO: Pending implementation")
        
        def __str__(self) -> str:
            datum_str = ""
            for key, value in self.__dict__.items():
                datum_str = f"{datum_str}Key: {key}\tValue: {value}\n"
            return datum_str

        def as_dict(self):
            return self.__dict__

        def get(self, item, default=None):
            return getattr(self, item, default)

    def __init__(self, question, data, **kwargs):
        self.retrieved_passages = []
        self.rouge_scorer = None
        self.question = question
        self.read_data(data)

    def __len__(self):
        return len(self.retrieved_passages)

    def __add__(self, other):
        return self.append(other)

    def append(self, data: Union[Dict[str, str], SearchDatum], **kwargs):
        if isinstance(data, SearchResult.SearchDatum):
            self.retrieved_passages.append(data)
        else:
            self.retrieved_passages.append(SearchResult.SearchDatum(data, **kwargs))

    def remove_duplicates(self, duplicate_removal: str = "none",
                          rouge_duplicate_threshold: float = -1.0):
        if (duplicate_removal is None or
                duplicate_removal == "none" or
                self.retrieved_passages == []
        ):
            return
        # ret = SearchResult(question=self.question, data=[])
        keep_passages = []
        if duplicate_removal == "exact":
            seen = {self.retrieved_passages[0].get_text(): 1}
            ret = [self.retrieved_passages[0]]
            for r in self.retrieved_passages[1:]:
                text_ = r.get_text()
                if text_ not in seen:
                    seen[text_] = 1
                    keep_passages.append(r)
        elif duplicate_removal == "rouge":
            from rouge_score.rouge_scorer import RougeScorer
            if self.rouge_scorer is None:
                self.rouge_scorer = RougeScorer(['rouge1'],
                                                use_stemmer=True)

            for r in self.retrieved_passages[1:]:
                found = False
                text_ = r.get_text()
                for c in keep_passages:
                    scr = self.rouge_scorer.score(c.get_text(), text_)
                    if scr['rouge1'].fmeasure >= rouge_duplicate_threshold:
                        found = True
                        break
                if not found:
                    keep_passages.append(r)
                    # ret.append(r)
        elif duplicate_removal.startswith("key:"):
            key = duplicate_removal.replace("key:","")
            seen = set()
            for i, r in enumerate(self.retrieved_passages):
                try:
                    # print(f"it={i}")
                    val = get_param(r, key, None)
                    # print(f"val={val}")
                    if val and val not in seen:
                        seen.add(val)
                        keep_passages.append(r)
                except Exception as e:
                    print(f"Error on {r.metadata}: {e}")
        self.retrieved_passages = keep_passages

    def __getitem__(self, i: int) -> SearchDatum:
        return self.retrieved_passages[i]

    def __iter__(self):
        return iter(self.retrieved_passages)

    def top_k(self, k: int):
        return self.retrieved_passages[:k] if k>0 else self.retrieved_passages

    def read_data(self, data):
        if isinstance(data, dict):
            if 'hits' in data and 'hits' in data['hits']:  # Looks like Elasticsearch results
                for i, d in enumerate(data['hits']['hits']):
                    # TODO - Do we need to store metadata or just the source text is ok?
                    # {'_index': 'jatin-testing', '_id': '1231', '_score': 0.3945668, '_ignored': ['text.keyword'], '_source': {'text': '\nRed Hat Enterprise Linux 4- all architectures Red Hat Enterprise Linux 5- all archit...Hat Enterprise Linux 6- all architectures '}}
                    # r = SearchResult.SearchDatum(d['_source'])
                    r = SearchResult.SearchDatum(d, rank=i)
                    self.retrieved_passages.append(r)
            else:
                raise Exception("TODO: Pending implementation")
                # for d in data:
                #     r = SearchResult.SearchDatum(d)
                #     self.results.append(r)
        elif isinstance(data, list):
            if len(data) == 0:
                return []
            elif isinstance(data[0], dict) or isinstance(data[0], pymilvus.client.search_reasult.Hit):
                if 'entity' in data[0]:
                    data = sorted(data, key=lambda k: (k['distance'], k['entity']['id']), reverse=True)
                    for r in data:
                        self.retrieved_passages.append(SearchResult.SearchDatum(r['entity'], score=r['distance']))
                else:
                    for r in data:
                        self.retrieved_passages.append(SearchResult.SearchDatum(r))
            elif data[0].__class__ == SearchResult.SearchDatum:
                for item in data:
                    self.retrieved_passages.append(item)
            else: # Need to deal with other structures.
                raise Exception("TODO: Pending implementation")
        else: # Need to deal with other structures.
            raise Exception("TODO: Pending implementation")
            return

    def as_dict(self):
        return {"question": self.question.as_dict(), "retrieved_passages": [r.as_dict() for r in self.retrieved_passages]}

    def as_json(self, **kwargs):
        return json.dumps(self.as_dict(), **kwargs)