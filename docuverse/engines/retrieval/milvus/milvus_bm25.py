import os

from docuverse import SearchCorpus, SearchQueries
from docuverse.engines import SearchData
from docuverse.engines.retrieval.milvus.milvus import MilvusEngine

try:
    from pymilvus import (
        FieldSchema, CollectionSchema, DataType, MilvusClient,
        Collection,
    )
    from pymilvus import model as MilvusModel
    from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
    from milvus_model.sparse.bm25 import BM25EmbeddingFunction
except:
    print(f"You need to install pymilvus to be using Milvus functionality!")
    raise RuntimeError("fYou need to install pymilvus to be using Milvus functionality!")
from docuverse.engines.search_engine_config_params import SearchEngineConfig
from docuverse.utils import get_param
from tqdm import tqdm

from docuverse.utils.embeddings.sparse_embedding_function import SparseEmbeddingFunction

class MilvusBM25Engine(MilvusEngine):
    def __init__(self, config: SearchEngineConfig|dict, **kwargs) -> None:
        super().__init__(config, **kwargs)


    def init_model(self, kwargs):
        self.analyzer = build_default_analyzer(get_param(kwargs, 'language', "en"))
        self.bm25_ef = BM25EmbeddingFunction(self.analyzer)
        idf_file = self.config.milvus_idf_file
        if idf_file is not None and os.path.exists(idf_file):
            self.bm25_ef.load(idf_file)

    def prepare_index_params(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="_id",
            index_type="STL_SORT"
        )
        index_params.add_index(field_name="embeddings",
                               index_name="sparse_inverted_index",
                               index_type="SPARSE_INVERTED_INDEX",
                               metric_type="IP",
                               params={"drop_ratio_build": 0.2})
        return index_params

    def create_fields(self):
        fields = super().create_fields()
        fields.append(
            FieldSchema(name="embeddings", dtype=DataType.SPARSE_FLOAT_VECTOR)
        )
        return fields

    def analyze(self, text):
        res = self.bm25_ef.encode_documents([text])
        return list(res[0])

    def get_search_params(self):
        search_params = get_param(self.config, 'search_params',
                                  self.milvus_defaults['search_params']["BM25"])

        return search_params

    def encode_data(self, texts, batch_size, show_progress=True):
        print(f"Computing IDF - will might a while.")
        self.bm25_ef.fit(texts)
        # print("Computing embeddings for the input.")
        embeddings = []
        t=tqdm(desc="Encoding documents", total=len(texts), disable=not show_progress)
        for i in range(0, len(texts), batch_size):
            last = min(i + batch_size, len(texts))
            encs = self.bm25_ef.encode_documents(texts[i:last])
            embeddings.extend(list(encs))
            t.update(last - i)
        #embeddings = self.bm25_ef.encode_documents(texts)
        return embeddings

    def encode_query(self, question):
        return list(self.bm25_ef.encode_queries([question.text]))[0]

    def get_search_request(self, text):
        data = self.encode_data([text], batch_size=1, show_progress=False)
        search_params = self.get_search_params()

    def ingest(self, corpus: SearchCorpus, update: bool = False):
        index_created = super().ingest(corpus, update)
        if index_created:
            self.save_idf_index()

    def save_idf_index(self):
        idf_file = self.config.milvus_idf_file
        if idf_file is None:
            idf_file = os.path.join(self.config.project_name, f"{self.config.index_name}.idf")
        if not os.path.exists(os.path.dirname(idf_file)):
            os.makedirs(os.path.dirname(idf_file))
        self.bm25_ef.save(idf_file)

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
        config = {
            "index_name": collection_name,
            "server": "localhost",
            "milvus_idf_file": os.path.join("/tmp", collection_name+".idf"),
            "db_engine": "milvus-bm25",
            "num_preprocessor_threads": -1,
            "top_k": 2,
        }
        queries = ["Who is Alan Turing?", "What is quick setup?"]
        server = cls(config)
        input = [
            {"text": txt, 'id': f"sent{id}", "title": ""} for i, txt in enumerate(docs)
        ]
        data = SearchData.read_data(input, num_preprocessor_threads=-1)
        server.ingest(data)
        qs = [{"text": q, "id": i} for i, q in enumerate(queries)]
        qs = SearchQueries.read_question_data(qs)
        res = [server.search(q) for q in qs]
        import json
        for r in res:
            print(r.as_json(indent=2))
        # splade_enc, milvus_client = cls.setup(collection_name)
        # cls.encode_and_index(collection_name, splade_enc, milvus_client, docs)
        # cls.query(collection_name, splade_enc, milvus_client, queries)
        # milvus_client.drop_collection(collection_name)

