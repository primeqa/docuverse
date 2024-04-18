from typing import Tuple, Dict

from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.retrieval import elastic
from docuverse.utils import get_param

class RetrievalEngine:
    def __init__(self, config_params, **kwargs):
        self.args = kwargs
        self.engine = self.create_engine(retriever_config=config_params)

    def search(self, query, **kwargs):
        return self.engine.search(query)

    def ingest(self, corpus: SearchCorpus,**kwargs):
        self.engine.ingest(corpus)
    
    def info(self):
        return self.engine.info()
    
    def create_engine(self, retriever_config:dict):
        """
        Create a retriever object based on the given retrieval configuration.

        Parameters:
        retrieval_config (dict): A dictionary containing the retrieval configuration.

        Returns:
        engine: A retriever object.

        """
        name = get_param(retriever_config, 'name')
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
        
        return engine

    def create_query(self, text, **kwargs) -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
        return None, None, None
