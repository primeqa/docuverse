from datetime import datetime
from typing import Tuple, Dict, Union

from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_engine_config_params import SearchEngineConfig
from docuverse.utils import get_param


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

        if self.last_access is None or (
                tm - self.last_access).total_seconds() > 10 * 60:  # more than 10 mins -> reconnect
            self.init_client()
        self.set_accessed()

    def set_accessed(self):
        self.last_access = datetime.now()

    def load_model_config(self, config_params: Union[dict, SearchEngineConfig]):
        if isinstance(config_params, dict):
            config_params = SearchEngineConfig(config=config_params)

        PARAM_NAMES = ["index_name", "title_field", "text_field", "n_docs", "filters", "duplicate_removal",
                       "rouge_duplicate_threshold"]

        for param_name in PARAM_NAMES:
            setattr(self, param_name, get_param(config_params, param_name))

        self.config = config_params
