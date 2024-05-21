from typing import Tuple, Dict

from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.retrieval import elastic
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
        self.args = kwargs
        self.engine = self.create_engine(retriever_config=config_params)

    def search(self, query, **kwargs):
        return self.engine.search(query)

    def ingest(self, corpus: SearchCorpus, **kwargs):
        self.engine.ingest(corpus)

    def info(self):
        return self.engine.info()

    @staticmethod
    def create_engine(retriever_config: dict):
        """
        Create a retriever object based on the given retrieval configuration.

        Parameters:
        retrieval_config (dict): A dictionary containing the retrieval configuration.

        Returns:
        engine: A retriever object.

        """
        name = retriever_config.get('db_engine')
        engine = None
        if name.startswith('elastic-') or name.startswith('es-'):
            if name in ['es-bm25', 'elastic-bm25']:
                engine = elastic.ElasticBM25Engine(retriever_config)
            elif name in ['es-dense', 'elastic-dense']:
                engine = elastic.ElasticDenseEngine(retriever_config)
            elif name in ['es-elser', 'elastic-elser']:
                engine = elastic.ElasticElserEngine(retriever_config)
            elif name in ['es-hybrid', "elastic-hybrid"]:
                engine = elastic.ElasticHybridEngine(retriever_config)
        elif name.startswith('primeqa'):
            pass
        elif name == 'chromadb':
            try:
                from docuverse.engines.retrieval.vectordb.chromadb import ChromaDBEngine
                engine = ChromaDBEngine(retriever_config)
            except ImportError as e:
                print("You need to install docuverse_chomadb package.")
                raise e
        elif name == 'milvus':
            try:
                from docuverse.engines.retrieval.milvus import MilvusEngine
                engine = MilvusEngine(retriever_config)
            except ImportError as e:
                print("You need to install docuverse_chomadb package.")
                raise e

        return engine

    @staticmethod
    def create_query(text, **kwargs) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        return None, None, None
