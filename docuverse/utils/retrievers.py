import docuverse.engines.retrieval.elastic as elastic
from docuverse.engines.reranking.dense_reranker import DenseReranker
from docuverse.engines.reranking.splade_reranker import SpladeReranker
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
    elif name.startswith('milvus'):
        try:
            if name in ['milvus_dense', 'milvus', 'milvus-dense']:
                from docuverse.engines.retrieval.milvus.milvus_dense import MilvusDenseEngine
                engine = MilvusDenseEngine(retriever_config)
            elif name in ['milvus_sparse', "milvus-sparse"]:
                from docuverse.engines.retrieval.milvus.milvus_sparse import MilvusSparseEngine
                engine = MilvusSparseEngine(retriever_config)
            elif name in ["milvus_bm25", "milvus-bm25"]:
                from docuverse.engines.retrieval.milvus.milvus_bm25 import MilvusBM25Engine
                engine = MilvusBM25Engine(retriever_config)
            else:
                raise NotImplementedError(f"Unknown engine type: {name}")
        except ImportError as e:
            print("You need to install pymilvus package.")
            raise e
    elif name.startswith("file:"):
        from docuverse.engines.retrieval.file.file_engine import FileReaderEngine
        engine = FileReaderEngine(retriever_config)

    return engine

def create_reranker_engine(reranker_config: dict|RerankerArguments):
    name = reranker_config.get('reranker_engine', 'dense')
    if reranker_config.reranker_model is None or name == "none":
        return None
    if name == 'dense':
        return DenseReranker(reranker_config)
    elif name == "splade":
        return SpladeReranker(reranker_config)
    else:
        raise RuntimeError("The available reranking engine types are 'dense', 'splade', and 'none'.")
