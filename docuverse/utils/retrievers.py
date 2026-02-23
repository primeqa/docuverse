from typing import Any
from docuverse.engines.search_engine_config_params import RerankerArguments

def create_retrieval_engine(retriever_config: dict):
   """
   Create a retriever object based on the given retrieval configuration.
    Parameters:
    retrieval_config (dict): A dictionary containing the retrieval configuration.

    Returns:
    engine: A retriever object.

   """
   name: str | None = retriever_config.get('db_engine')
   engine = None
   print(f"Retrieval engine: {name}")
   if name.startswith('elastic-') or name.startswith('es-'):
       import docuverse.engines.retrieval.elastic as elastic
       if name in ['es-bm25', 'elastic-bm25']:
           engine = elastic.ElasticBM25Engine(retriever_config)
       elif name in ['es-dense', 'elastic-dense']:
           engine = elastic.ElasticDenseEngine(retriever_config)
       elif name in ['es-elser', 'elastic-elser']:
           engine = elastic.ElasticElserEngine(retriever_config)
   elif name.startswith('primeqa'):
       pass
   elif name == 'chromadb':
       try:
           from docuverse.engines.retrieval.chromadb.chromadb_engine import ChromaDBEngine
           engine = ChromaDBEngine(retriever_config)
       except ImportError as e:
           print("You need to install docuverse_chomadb package (run `pip install -r requirements-chromadb.txt`).")
           raise e
   elif name == 'faiss':
       try:
           from docuverse.engines.retrieval.faiss.faiss_engine import FAISSEngine
           engine = FAISSEngine(retriever_config)
       except ImportError as e:
           print("You need to install faiss package (run `pip install faiss-cpu` or `pip install faiss-gpu`).")
           raise e
   elif name.startswith('milvus'):
       import docuverse.engines.retrieval.milvus as milvus
       try:
           if name in ['milvus_dense', 'milvus', 'milvus-dense']:
               engine = milvus.MilvusDenseEngine(retriever_config)
           elif name in ['milvus_sparse', "milvus-sparse"]:
               engine = milvus.MilvusSparseEngine(retriever_config)
           elif name in ["milvus_bm25", "milvus-bm25"]:
               engine = milvus.MilvusBM25Engine(retriever_config)
           elif name in ['milvus_hybrid', "milvus-hybrid"]:
               engine = milvus.MilvusHybridEngine(retriever_config)
           elif name in ['milvus_splade ', 'milvus-splade']:
               engine = milvus.MilvusSpladeEngine(retriever_config)
           else:
               raise NotImplementedError(f"Unknown engine type: {name}")
       except ImportError as e:
           print("You need to install pymilvus package.")
           raise e
   elif name in ['lancedb', 'lance']:
       try:
           from docuverse.engines.retrieval.lancedb import LanceDBEngine
           engine = LanceDBEngine(retriever_config)
       except ImportError as e:
           print("You need to install lancedb package (run `pip install lancedb`).")
           raise e
   elif name.startswith("file:"):
       from docuverse.engines.retrieval.file.file_engine import FileEngine
       engine = FileEngine(retriever_config)

   return engine

def create_reranker_engine(reranker_config: dict|RerankerArguments):
    name = reranker_config.get('reranker_engine', 'dense')
    if reranker_config.reranker_model is None or name == "none":
        return None
    if name == 'dense':
        from docuverse.engines.reranking.dense_reranker import DenseReranker
        return DenseReranker(reranker_config)
    elif name == "splade":
        from docuverse.engines.reranking.splade_reranker import SpladeReranker
        return SpladeReranker(reranker_config)
    elif name == "cross-encoder":
        from docuverse.engines.reranking.cross_encoder_reranker import CrossEncoderReranker
        return CrossEncoderReranker(reranker_config)
    else:
        raise RuntimeError("The available reranking engine types are 'dense', 'splade', and 'none'.")
