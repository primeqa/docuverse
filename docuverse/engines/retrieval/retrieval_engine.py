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
        self.config = None
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

    def delete_index(self, index_name, **kwargs):
        pass

    def check_index_rebuild(self, **kwargs) -> bool:
        """
        Checks if the user wants to recreate the index.

        :return: True if the user wants to recreate the index, False otherwise.
        """
        index_name = self.config.index_name
        import sys
        while True:
            r = input(
                f"Are you sure you want to recreate the index {index_name}? It might take a long time!!"
                f" Say 'yes', 'no', or 'skip':").strip()
            if r == 'no':
                print("OK - exiting. Run with '--actions r'")
                sys.exit(0)
            elif r == 'yes':
                return True
            elif r == 'skip':
                print("Skipping ingestion.")
                return False
            else:
                print(f"Please type 'yes' or 'no', not {r}!")
        # return True

    def create_update_index(self, do_update:bool=True, **kwargs) -> bool:
        """
        Create or update an index.

        Parameters:
        - do_update (bool): Specifies whether to perform an update operation (default=True).

        Returns:
        - bool: True if the operation is successful, False otherwise.

        """
        if self.has_index(index_name=self.config.index_name):
            if do_update:
                do_index = self.check_index_rebuild()
                if not do_index:
                    return False
                self.delete_index(index_name=self.config.index_name, **kwargs)
            else:
                print(f"This will overwrite your existent index {self.config.index_name} - use --actions 'u' instead.")
                return self.check_index_rebuild()
        else:
            if do_update:
                print("You are trying to update an index that does not exist "
                      "- will ignore your command and create the index.")
        if not self.has_index(index_name=self.config.index_name):
            self.create_index(self.config.index_name, **kwargs)
        return True

    def create_index(self, index_name: str, **kwargs):
        pass

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

        _param_names = ["index_name", "title_field", "text_field", "n_docs", "filters", "duplicate_removal",
                       "rouge_duplicate_threshold"]

        for param_name in _param_names:
            setattr(self, param_name, get_param(config_params, param_name))

        self.config = config_params






