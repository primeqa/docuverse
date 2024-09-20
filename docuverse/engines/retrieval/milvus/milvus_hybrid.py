from docuverse import SearchCorpus, SearchQueries, SearchResult
from docuverse.engines.retrieval.milvus.milvus import MilvusEngine
from docuverse.engines.retrieval.milvus.milvus_dense import MilvusDenseEngine
from docuverse.engines.retrieval.milvus.milvus_bm25 import MilvusBM25Engine
from docuverse.engines.retrieval.milvus.milvus_sparse import MilvusSparseEngine
from docuverse.utils.retrievers import create_retrieval_engine

try:
    from pymilvus import (
        FieldSchema, CollectionSchema, DataType, MilvusClient,
        Collection, AnnSearchRequest, RRFRanker,
)
except:
    print(f"You need to install pymilvus to be using Milvus functionality!")
    raise RuntimeError("fYou need to install pymilvus to be using Milvus functionality!")
from docuverse.engines.search_engine_config_params import SearchEngineConfig
from docuverse.utils import get_param
from docuverse.utils.embeddings.sparse_embedding_function import SparseEmbeddingFunction


class MilvusHybridEngine(MilvusEngine):
    """
    This type of engine will hold multiple engines, that will be called sequentially, as needed.
    """
    def __init__(self, config: SearchEngineConfig|dict, **kwargs) -> None:
        self.reranker = None
        self.index_params = None
        self.models = None
        self.configs = None
        self.shared_tokenizer = get_param(self, 'hybrid.shared_tokenizer',
                                          get_param(kwargs, 'shared_tokenizer', False))

        super().__init__(config, **kwargs)

    def init_model(self, kwargs):
        # self.model = SparseEmbeddingFunction(self.config.model_name, batch_size=self.config.bulk_batch)
        models_names = self.config.hybrid['models'].keys()
        servers = self.config.hybrid['servers']
        uniq_servers = list(set(servers))
        if len(uniq_servers) != 1:
            raise RuntimeError("In the MilvusHybridEngine, all milvus instances need "
                               f"to be on the same server, but I found [{uniq_servers}] servers.")
        uniq_names = list(set(models_names))
        if len(uniq_names) != len(models_names):
            raise RuntimeError("In the MilvusHybridEngine, all models need to have a different name"
                               f" but I found [{uniq_names}] names.")
        common_config = {k:v for k, v in self.config.hybrid.items() if k!='models'}
        for m in models_names:
            self.configs[m] = {**common_config, **self.config.hybrid['models'][m]}
        engine_types = {'milvus_dense|milvus-dense': MilvusDenseEngine,
                        'milvus_sparse|milvus-sparse': MilvusSparseEngine,
                        'milvus_bm25|milvus-bm25': MilvusBM25Engine}

        self.models = [get_param(self.configs[m], 'db_engine')(self.configs[m])
                       for m in models_names]
        self.model = None
        self.client = self.models[0].client
        self.reranker = RRFRanker()

    def ingest(self, corpus: SearchCorpus, update: bool = False):
        for m in self.models:
            m.ingest(corpus, update)

    def create_index(self, index_name: str=None, fields=None, fmt=None, **kwargs):
        for m in self.models:
            m.create_index(index_name=None, fields=fields, fmt=fmt, **kwargs)

    def prepare_index_params(self):
        raise NotImplementedError
        # self.index_params = [self.models.prepare_index_params() for m in self.models]
        # return self.index_params

    def create_fields(self):
        fields = super().create_fields()
        fields.append(
            FieldSchema(name="embeddings", dtype=DataType.SPARSE_FLOAT_VECTOR)
        )
        return fields

    def delete_index(self, index_name: str|None=None, fmt=None, **kwargs):
        for m in self.models:
            m.delete_index(index_name=None, fmt=fmt, **kwargs)

    def get_search_params(self):
        raise NotImplementedError

    def search(self, question: SearchQueries.Query, **kwargs):
        search_params = [m.get_search_params() for m in self.models]
        data = [m.encode_query(question) for m in self.models]
        requests = []
        for s, d, m in zip(search_params, data, self.models):
            s['data'] = d
            s['anns_field'] = 'embeddings'
            s['limit'] = m.config.top_k
            requests.append(AnnSearchRequest(**s))
        res = self.client.hybrid_search(requests, self.reranker, limit=self.config.top_k)
        result = SearchResult(question=question, data=res)
        result.remove_duplicates(self.config.duplicate_removal,
                                 self.config.rouge_duplicate_threshold)
        return result

    @classmethod
    def test(cls):
        collection_name = "hello_sparse"
        docs = [
            "Artificial intelligence was founded as an academic discipline in 1956.",
            "Alan Turing was the first person to conduct substantial research in AI.",
            "Born in Maida Vale, London, Turing was raised in southern England.",
            "The quick setup collection has two mandatory fields: the primary and vector fields. "
            "It also allows the insertion of undefined fields and their values in key-value pairs in a dynamic field."
        ]
        queries = ["Who is Alan Turing?", "What is quick setup?"]

        splade_enc, milvus_client = cls.setup(collection_name)
        cls.encode_and_index(collection_name, splade_enc, milvus_client, docs)
        cls.query(collection_name, splade_enc, milvus_client, queries)
        milvus_client.drop_collection(collection_name)

