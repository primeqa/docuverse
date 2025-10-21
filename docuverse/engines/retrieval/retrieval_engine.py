import inspect
from datetime import datetime
from typing import Tuple, Dict, Union


from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_engine_config_params import SearchEngineConfig, RetrievalArguments
from docuverse.utils import get_param, ask_for_confirmation


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
        self.ingestion_batch_size = get_param(config_params, "ingestion_batch_size", default=None)
        self.embeddings_name = get_param(config_params, "embeddings_name", None)
        self.last_access = None
        self.args = kwargs
        self.config = None
        self.client = None
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

    def check_client(self):
        if self.client is None:
            raise RuntimeError("MilvusEngine server is not defined/initialized.")

    def has_index(self, index_name):
        return False

    def delete_index(self, index_name, **kwargs):
        pass

    def check_index_rebuild(self, **kwargs) -> bool|str:
        """
        check_index_rebuild(**kwargs) -> bool

        Prompts the user to confirm if they want to recreate the index. This method will ask for confirmation using a prompt and wait for the user's response. Depending on the user's input, the method will return True, False, or exit the program.
        Parameters:
          **kwargs: Additional keyword arguments (not used in this function).
        Returns:
          bool: Returns True if the user confirms to recreate the index, False if the user chooses to skip the operation.
        """
        import sys
        while True:
            r = ask_for_confirmation(text=f"Are you sure you want to recreate the index {self.config.index_name}? It might take a long time!!",
                                     answers=['yes', 'no', 'skip', 'update'],
                                     default='skip')
            if r == 'no':
                print("OK - exiting. Run with '--actions r'")
                sys.exit(0)
            elif r == 'yes':
                return True
            elif r == 'skip':
                print("Skipping ingestion.")
                return False
            elif r == 'update':
                return "update"
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
                if do_index == False:
                    return False
                if do_index != "update": # skip if self.update==True
                    self.delete_index(index_name=self.config.index_name, **kwargs)
            else:
                print(f"This will overwrite your existent index {self.config.index_name} - use --actions 'u' instead.")
                result = self.check_index_rebuild()
                if result in ['update', False]:
                    return result
                else:
                    self.delete_index(index_name=self.config.index_name, **kwargs)
        else:
            if do_update:
                print(f"You are trying to update an index ({self.config.index_name}) that does not exist "
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
            # config_params = SearchEngineConfig(config=config_params)
            members = inspect.getmembers(RetrievalArguments)
            fields =[x.name for x in list(filter(lambda x: x[0] == '__dataclass_fields__', members))[0][1].values()]
            config_params = RetrievalArguments(**{k:v for k,v in config_params.items() if k in fields})

        _param_names = ["index_name", "title_field", "text_field", "n_docs", "filters", "duplicate_removal",
                       "rouge_duplicate_threshold"]

        for param_name in _param_names:
            setattr(self, param_name, get_param(config_params, param_name))

        self.config = config_params

    def encode_data(self, texts, batch_size, show_progress_bar=False):
        pass

    def _analyze_data(self, texts):
        pass

    def _create_data(self, **kwargs):
        pass





