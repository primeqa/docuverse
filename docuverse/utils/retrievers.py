import docuverse.engines.retrieval.elastic as elastic
from docuverse.engines.reranking.dense_reranker import DenseReranker
from docuverse.engines.search_engine_config_params import RerankerArguments


def create_retrieval_engine(retriever_config: dict):
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


def create_reranker_engine(reranker_config: dict|RerankerArguments):
    name = reranker_config.get('reranker_engine', 'dense')
    if name == 'dense':
        if reranker_config.reranker_model is None:
            return None
        return DenseReranker(reranker_config)
    else:
        raise RuntimeError("The only available reranking engine is 'dense'")
