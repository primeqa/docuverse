from typing import Tuple, Dict

from docuverse import SearchEngine, SearchCorpus
from docuverse.utils import get_param

class RetrieverEngine:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.duplicate_removal = get_param(kwargs, 'duplicate_removal', 'none')
        self.rouge_duplicate_threshold = get_param(kwargs, 'rouge_duplicate_threshold', -1)

    def search(self, query, **kwargs):
        pass

    def ingest(self, corpus: SearchCorpus, max_document_size=-1, stride=-1, title_handling: str = "all",
               **kwargs):
        pass

    @staticmethod
    def create_engine(retriever_config:str):
        """
        Create a retriever object based on the given retrieval configuration.

        Parameters:
        retrieval_config (dict): A dictionary containing the retrieval configuration.

        Raises:
        RuntimeError: If the docuverse_elastic package is not installed.

        Returns:
        engine: A retriever object.

        """
        name = get_param(retriever_config, 'name')
        try:
            from docuverse.engines.retrieval import elastic
        except ImportError as e:
            print(f"You need to install the docuverse_elastic package!")
            raise e

        if name.startswith('elastic'):
            if name == 'elastic_bm25':
                engine = elastic.ElasticBM25Engine(retriever_config)
            elif name == 'elastic_dense':
                engine = elastic.ElasticDenseEngine(retriever_config)
            elif name == 'elastic_elser':
                engine = elastic.ElasticElserEngine(retriever_config)
            elif name == "elastic_hybrid":
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

    def create_query(self, text, **kwargs) -> Tuple[Dict[str:str], Dict[str:str], Dict[str:str]]:
        return None, None, None
