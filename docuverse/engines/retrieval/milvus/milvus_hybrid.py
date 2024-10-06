import os

from pymilvus.client.constants import ConsistencyLevel
from tqdm import tqdm
from scipy.sparse._csr import csr_array
from docuverse import SearchCorpus, SearchQueries, SearchResult
from docuverse.engines.retrieval.milvus.milvus import MilvusEngine
from docuverse.engines.retrieval.milvus.milvus_dense import MilvusDenseEngine
from docuverse.engines.retrieval.milvus.milvus_bm25 import MilvusBM25Engine
from docuverse.engines.retrieval.milvus.milvus_sparse import MilvusSparseEngine

try:
    from pymilvus import (
        FieldSchema, CollectionSchema, DataType, MilvusClient,
        Collection, AnnSearchRequest, RRFRanker, connections, utility, WeightedRanker
)
except:
    print(f"You need to install pymilvus to be using Milvus functionality!")
    raise RuntimeError("fYou need to install pymilvus to be using Milvus functionality!")
from docuverse.engines.search_engine_config_params import DocUVerseConfig, RetrievalArguments
from docuverse.utils import get_param


class MilvusHybridEngine(MilvusEngine):
    """
    Class MilvusHybridEngine

    __init__(self, config, **kwargs)
        Initializes the MilvusHybridEngine.
        Parameters:
        - config (DocUVerseConfig|RetrievalArguments|dict): Configuration for the engine.
        - **kwargs: Additional arguments.

    init_model(self, kwargs)
        Initializes the models based on the provided configuration.
        Parameters:
        - kwargs: Additional arguments.

    init_client(self)
        Initializes the client, overriding the parent functionality.

    _create_data(self, corpus, texts)
        Creates data for ingestion.
        Parameters:
        - corpus: The corpus of documents.
        - texts: The texts to be encoded.

    _check_zero_size_sparse_vector(val)
        Checks if the sparse vector has zero size.
        Parameters:
        - val: The sparse vector to check.
        Returns:
        - bool: True if the vector has zero size, False otherwise.

    _insert_data(self, data)
        Inserts data into the collection with retries on failure.
        Parameters:
        - data: The data to insert.

    create_fields(self, embeddings_name, new_fields_only)
        Creates fields for the collection.
        Parameters:
        - embeddings_name: The name of the embeddings field.
        - new_fields_only: Whether to create only new fields.
        Returns:
        - list: The list of fields.

    create_index(self, index_name, fields, fmt, **kwargs)
        Creates an index for the collection.
        Parameters:
        - index_name: The name of the index.
        - fields: The fields of the schema.
        - fmt: The format of the schema.
        - **kwargs: Additional arguments.

    prepare_index_params(self, embeddings_name)
        Prepares index parameters.
        Parameters:
        - embeddings_name: The name of the embeddings field.
        Raises:
        - NotImplementedError: This method should be implemented by subclasses.

    check_client(self)
        Checks the client connection status.
        Returns:
        - bool: True if the client is connected, False otherwise.

    ingest(self, corpus, update)
        Ingests the provided corpus into the collection.
        Parameters:
        - corpus: The corpus to ingest.
        - update: Whether to update existing documents.
        Returns:
        - bool: True if data was ingested, False otherwise.

    get_search_params(self)
        Gets search parameters.
        Raises:
        - NotImplementedError: This method should be implemented by subclasses.

    search(self, question, **kwargs)
        Searches the collection based on the provided query.
        Parameters:
        - question: The query to search for.
        - **kwargs: Additional arguments.
        Returns:
        - SearchResult: The search results.

    test(cls)
        Class method to test the engine.
    """
    def __init__(self, config: DocUVerseConfig|RetrievalArguments|dict, **kwargs) -> None:
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
        combination_config = self.config.hybrid['combination']
        if combination_config == "rrf":
            self.reranker = RRFRanker()
        elif combination_config.find("weighted") >= 0:
            try:
                weights = [float(self.config.hybrid['models'][m]['weight']) for m in self.model_names]
                _sum = sum(weights)
                weights = [w/_sum for w in weights]
            except:
                print(f"Not all models have weights, yet you specidied a weighted combination! Exiting.")
                raise RuntimeError("Invalid number of weights.")
            self.reranker = WeightedRanker(*weights)

    def init_client(self): #override the parent functionality
        super().init_client()
        connections.connect(host=self.server.host, port=self.server.port,
                            secure=get_param(self.server, "secure", False),
                            server_pem_path=get_param(self.server, "server_pem_path", None))
        if self.has_index(self.config.index_name):
            self.collection = Collection(name=self.config.index_name)


    def _create_data(self, corpus, texts, show_progress_bar=True, tqdms=None):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        data = []
        if tqdms is None:
            tqdms = [None] * len(self.models)
        for m, tq in zip(self.models, tqdms):
            data.append(m.encode_data(texts, self.ingest_batch_size,
                                      show_progress_bar=show_progress_bar,
                                      tqdm_instance=tq)
                        )

        vals = []
        for i, item in enumerate(corpus):
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

    @staticmethod
    def _check_zero_size_sparse_vector(val):
        return isinstance(val, csr_array) and val.getnnz() == 0

    def _insert_data(self, data, show_progress_bar=True):
        tbatch_size = 10*self.ingest_batch_size
        for i in tqdm(range(0, len(data), tbatch_size),
                      desc="Ingesting documents", disable=not show_progress_bar):
            num_tries = 0
            while num_tries < 10:
                try:
                    self.collection.insert(data[i:i+tbatch_size])
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

    def prepare_index_params(self, embeddings_name="embeddings"):
        raise NotImplementedError

    def check_client(self):
        return True

    # def ingest(self, corpus: SearchCorpus, update: bool = False):


    def ingest(self, corpus: SearchCorpus, update: bool = False):
        # data_ingested = super().ingest(corpus=corpus, update=update)
        texts = self._check_index_creation_and_get_text(corpus, update)

        for m in self.models:
            m._analyze_data(texts)

        if texts is None:
            return False
        main_tqdm = tqdm(desc="Processing documents:", total=len(texts))
        tqdms = []
        for m, name in zip(self.models, self.model_names):
            tqdms.append(tqdm(desc=f"Encoding {name}", total=len(texts), leave=False))
        tbatch_size = 5*self.ingest_batch_size
        for bi in range(0, len(texts), tbatch_size):
            data = self._create_data(corpus[bi: bi+tbatch_size], texts[bi:bi+tbatch_size],
                                     show_progress_bar=False,
                                     tqdms=tqdms)
            # for tq in tqdms:
            #     tq.update(tbatch_size)
            self._insert_data(data, show_progress_bar=False)
            main_tqdm.update(tbatch_size)

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

