from datetime import datetime
from typing import Tuple, Dict

from docuverse.engines.search_corpus import SearchCorpus

class RetrievalEngine:
    """

    RetrievalEngine

    Class representing a retrieval engine for searching and ingesting data.

    Methods:
    - __init__(self, config_params, **kwargs)
        Initializes a RetrievalEngine object with the given configuration parameters.

    - search(self, query, **kwargs)
        Performs a search using the retrieval engine.

    - ingest(self, corpus: SearchCorpus, **kwargs)
        Ingests a corpus into the retrieval engine.

    - info(self)
        Retrieves information about the retrieval engine.

    - create_engine(retriever_config: dict) -> engine
        Creates a retriever object based on the given retrieval configuration.

    - create_query(text, **kwargs) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]
        Creates a query based on the given text.

    """
    def __init__(self, config_params, **kwargs):
        self.last_access = None
        self.args = kwargs
        # self.engine = self.create_engine(retriever_config=config_params)

    def search(self, query, **kwargs):
        pass
        # return self.engine.search(query)

    def ingest(self, corpus: SearchCorpus, **kwargs):
        pass
        # self.engine.ingest(corpus, **kwargs)

    def info(self):
        pass
        # return self.engine.info()

    def init_client(self):
        pass

    def has_index(self, index_name):
        return False

    @staticmethod
    def create_query(text, **kwargs) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        return None, None, None

    def reconnect_if_necessary(self):
        tm = datetime.now()

        if self.last_access is None or (tm-self.last_access).total_seconds() > 10*60: # more than 10 mins -> reconnect
            self.init_client()
        self.set_accessed()

    def set_accessed(self):
        self.last_access = datetime.now()
