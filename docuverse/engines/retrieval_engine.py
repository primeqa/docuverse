from importlib import import_module

from docuverse.engines import SearchCorpus
from docuverse.utils import get_param


class RetrievalEngine:
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
    def create_engine(retriever_config: str):
        """
        Create a retriever object based on the given retrieval configuration.
        Parameters:
        retrieval_config (dict): A dictionary containing the retrieval configuration.
        Raises:
        RuntimeError: If the docuverse_elastic package is not installed.
        Returns:
        engine: A retriever object.
        """
        engine_map = {
            'elastic_bm25': 'docuverse.engines.retrieval.elastic.ElasticBM25Engine',
            'elastic_dense': 'docuverse.engines.retrieval.elastic.ElasticDenseEngine',
            'elastic_elser': 'docuverse.engines.retrieval.elastic.ElasticElserEngine',
            'elastic_hybrid': 'docuverse.engines.retrieval.elastic.ElasticHybridEngine',
            'chromadb': 'docuverse.engines.retrieval.vectordb.chromadb.ChromaDBEngine',
            'milvus': 'docuverse.engines.retrieval.milvus.MilvusEngine',
        }

        name = get_param(retriever_config, 'name')
        engine_class_path = engine_map.get(name)

        if not engine_class_path:
            raise ValueError(f"Invalid retriever_config name: {name}")

        module_path, class_name = engine_class_path.rsplit('.', 1)

        try:
            module = import_module(module_path)
            engine_class = getattr(module, class_name)
            engine = engine_class(retriever_config)
        except ImportError as e:
            print(f"You need to install the appropriate package for {name}!")
            raise e

        return engine

        def create_query(self, text, **kwargs) -> tuple[Any, Any, Any]:
            return None, None, None
