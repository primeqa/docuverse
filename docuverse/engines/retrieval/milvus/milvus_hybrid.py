import os

from pandas.conftest import spmatrix
from pydantic.v1 import NoneBytes
from pymilvus.client.constants import ConsistencyLevel
from tqdm import tqdm
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings
from scipy.sparse._csr import csr_array
from docuverse import SearchCorpus, SearchQueries, SearchResult
from docuverse.engines.retrieval.milvus.milvus import MilvusEngine
from docuverse.engines.retrieval.milvus.milvus_dense import MilvusDenseEngine
from docuverse.engines.retrieval.milvus.milvus_bm25 import MilvusBM25Engine
from docuverse.engines.retrieval.milvus.milvus_sparse import MilvusSparseEngine
from docuverse.utils.retrievers import create_retrieval_engine

try:
    from pymilvus import (
        FieldSchema, CollectionSchema, DataType, MilvusClient,
        Collection, AnnSearchRequest, RRFRanker, connections, utility
)
except:
    print(f"You need to install pymilvus to be using Milvus functionality!")
    raise RuntimeError("fYou need to install pymilvus to be using Milvus functionality!")
from docuverse.engines.search_engine_config_params import SearchEngineConfig
from docuverse.utils import get_param
from docuverse.utils.embeddings.sparse_embedding_function import SparseEmbeddingFunction


class MilvusHybridEngine(MilvusEngine):
    """
        MilvusHybridEngine class extends the MilvusEngine to support hybrid search capabilities using multiple models.
        The code has to temporarily use the old interface, as the MilvusClient interface to hybrid search is not available.

        Attributes:
        reranker (Optional[RRFRanker]): Reranker object used to rerank search results.
        index_params (Optional[List[Dict]]): Index parameters for the models.
        models (Optional[List[MilvusEngine]]): List of MilvusEngine instances.
        configs (Optional[Dict]): Configuration details for each model.
        config (SearchEngineConfig|dict): Configuration for the search engine.
        shared_tokenizer (bool): Indicates if a shared tokenizer is used across models.

        Methods:
        __init__(config: SearchEngineConfig|dict, **kwargs) -> None:
            Initializes the MilvusHybridEngine with the given configuration and additional arguments.

        init_model(kwargs) -> None:
            Initializes the models as specified in the configuration, setting up their individual configurations and engines.

        init_client() -> None:
            Overrides the parent method to avoid re-initializing the client.

        ingest(corpus: SearchCorpus, update: bool = False) -> None:
            Ingests the given corpus into all models, with an option to update existing corpus.

        create_index(index_name: str = None, fields = None, fmt = None, **kwargs) -> None:
            Creates index for all models using the specified index name, fields, and format.

        prepare_index_params() -> None:
            Prepares index parameters for all models.

        create_fields() -> List[FieldSchema]:
            Extends the parent method to add additional fields specific to embedding vectors.

        delete_index(index_name: str = None, fmt = None, **kwargs) -> None:
            Deletes index for all models using the specified index name and format.

        get_search_params() -> None:
            Retrieves search parameters for all models.

        search(question: SearchQueries.Query, **kwargs) -> SearchResult:
            Conducts a hybrid search across models using the given search query and additional arguments.

        test() -> None:
            Class method to perform basic testing of the hybrid engine capabilities.
    """
    def __init__(self, config: SearchEngineConfig|dict, **kwargs) -> None:
        self.collection = None
        self.model_names = None
        self.embedding_names = []
        self.connection = None
        self.reranker = None
        self.index_params = None
        self.models = None
        self.configs = None
        self.config = config
        self.shared_tokenizer = get_param(config, 'hybrid.shared_tokenizer',
                                          get_param(kwargs, 'shared_tokenizer', False))

        super().__init__(config, **kwargs)

    def init_model(self, kwargs):
        self.model_names = list(self.config.hybrid['models'].keys())
        if self.config.hybrid_submodules:
            submodules = self.config.hybrid_submodules.split(",")
            self.model_names = [m for m in self.model_names if m in submodules]


        common_config = {k:v for k, v in self.config.hybrid.items() if k!='models'}
        self.configs = {}
        self.models = []
        engine_types = {'milvus_dense': MilvusDenseEngine,
                        'milvus-dense': MilvusDenseEngine,
                        'milvus_sparse': MilvusSparseEngine,
                        'milvus-sparse': MilvusSparseEngine,
                        'milvus_bm25': MilvusBM25Engine,
                        'milvus-bm25': MilvusBM25Engine}
        for m in self.model_names:
            self.configs[m] = {**{k:v for k,v in self.config.__dict__.items() if k!='hybrid'},
                               **common_config,
                               **self.config.hybrid['models'][m]}
            engine = get_param(engine_types, get_param(self.configs[m], 'db_engine'))
            self.models.append(engine(self.configs[m]))

            self.embedding_names.append(f"{m}_embeddings".replace("-", "_"))

        self.model = None
        self.client = None
        # self.connection = connections.connect(host=self.server.host, port=self.server.port)
        self.reranker = RRFRanker()

    def init_client(self): #override the parent functionality
        super().init_client()
        connections.connect(host=self.server.host, port=self.server.port,
                            secure=get_param(self.server, "secure", False),
                            server_pem_path=get_param(self.server, "server_pem_path", None))
        if self.has_index(self.config.index_name):
            self.collection = Collection(name=self.config.index_name)


    def _create_data(self, corpus, texts):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        data = []
        for m in self.models:
            data.append(m.encode_data(texts, self.ingest_batch_size))

        vals = []
        for i, item in enumerate(corpus):
            # if isinstance(passage_vectors[i], spmatrix) and passage_vectors[i].getnnz() == 0:
            #     continue
            dt = {key: item[key] for key in ['id', 'text', 'title']}
            for f in self.config.data_template.extra_fields:
                dt[f] = str(item[f])
            for j, (m, name) in enumerate(zip(self.models, self.model_names)):
                val = data[j][i]
                if self._check_zero_size_sparse_vector(val):
                    print(f"Skipping {texts[i]}")
                    dt = None
                    break
                dt[self.embedding_names[j]] = val
            if dt: # skip elements for which the sparse vector is empty..
                vals.append(dt)

        return vals
        #     if i%self.ingest_batch_size == 0 and vals:
        #         self.collection.insert(vals)
        #         vals = []
        # if len(vals) > 0:
        #     self.collection.insert(vals)

    @staticmethod
    def _check_zero_size_sparse_vector(val):
        return isinstance(val, csr_array) and val.getnnz() == 0

    def _insert_data(self, data):
        num_tries = 0
        for i in tqdm(range(0, len(data), 10*self.ingest_batch_size), desc="Ingesting documents"):
            num_tries = 0
            while num_tries < 10:
                try:
                    self.collection.insert(data[i:i+self.ingest_batch_size])
                    break # exit the while loop
                except Exception as e:
                    num_tries += 1
                    if num_tries > 5:
                        print(f"Ingestion stopped after {i} items.")
                        raise e
                    self.init_client()
        self.wait_for_ingestion(data)

    def create_fields(self, embeddings_name="embeddings", new_fields_only=False):
        fields = super().create_fields()

        for i, (m, name) in enumerate(zip(self.models, self.model_names)):
            self.embedding_names.append(f"{name}_embeddings".replace("-", "_"))
            fields.extend(m.create_fields(embeddings_name=self.embedding_names[i], new_fields_only=True))
        return fields

    def create_index(self, index_name: str=None, fields=None, fmt=None, **kwargs):
        if index_name is None:
            index_name = self.config.index_name

        schema = CollectionSchema(fields, index_name, consistency=ConsistencyLevel.Eventually)
        self.collection = Collection(name=index_name, schema=schema)
        for m, emb_name in zip(self.models, self.embedding_names):
            index_params = m.prepare_index_params()
            self.collection.create_index(emb_name, index_params)

    # def ingest(self, corpus: SearchCorpus|list[SearchCorpus], update: bool = False):
    #     texts = self._check_index_creation_and_get_text(corpus)
    #
    #     if texts is None:
    #         return False
    #     data = self._create_data(corpus, texts)
    #     self._insert_data(data)
    #     return True

    def prepare_index_params(self, embeddings_name="embeddings"):
        raise NotImplementedError

    def check_client(self):
        return True

    def delete_index(self, index_name: str|None=None, fmt=None, **kwargs):
        self.client.drop_collection(self.config.index_name)
        # for m in self.models:
        #     m.delete_index(index_name=None, fmt=fmt, **kwargs)

    def ingest(self, corpus: SearchCorpus, update: bool = False):
        data_ingested = super().ingest(corpus=corpus, update=update)
        if data_ingested:
            for m in self.models:
                if isinstance(m, MilvusBM25Engine):
                    m.save_idf_index()

    def get_search_params(self):
        raise NotImplementedError

    def search(self, question: SearchQueries.Query, **kwargs):
        search_params = [{"param": m.get_search_params()} for m in self.models]
        data = [m.encode_query(question) for m in self.models]
        requests = []
        self.collection.load()
        for s, d, m, name in zip(search_params, data, self.models, self.embedding_names):
            if self._check_zero_size_sparse_vector(d):
                if self.config.verbose:
                    print(f"Question {question.text} has a 0-size sparse representation")
                continue
            s['data'] = [d]
            s['anns_field'] = name
            s['limit'] = m.config.top_k
            requests.append(AnnSearchRequest(**s))

        if len(requests)==0:
            return SearchResult(question=question, data=[])
        res = self.collection.hybrid_search(requests, self.reranker, limit=self.config.top_k, output_fields=self.output_fields)
        res_as_dict = [hit.to_dict() for hit in res[0]]
        result = SearchResult(question=question, data=res_as_dict)
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

