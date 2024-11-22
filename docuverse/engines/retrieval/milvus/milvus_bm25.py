import os

from docuverse import SearchCorpus, SearchQueries
from docuverse.engines import SearchData
from docuverse.engines.retrieval.milvus.milvus import MilvusEngine
from docuverse.utils.timer import timer

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
from docuverse.utils import get_param, ask_for_confirmation, convert_to_single_vectors
from tqdm import tqdm


class MilvusBM25Engine(MilvusEngine):
    def __init__(self, config: SearchEngineConfig|dict, **kwargs) -> None:
        super().__init__(config, **kwargs)


    def init_model(self, **kwargs):
        self.analyzer = build_default_analyzer(get_param(kwargs, 'language', "en"))
        self.bm25_ef = BM25EmbeddingFunction(self.analyzer)
        idf_file = self.config.milvus_idf_file
        if idf_file is not None and os.path.exists(idf_file):
            self.bm25_ef.load(idf_file)

    def prepare_index_params(self, embeddings_name="embeddings"):
        index_params = self.client.prepare_index_params()
        if self.server.type != "file":
            index_params.add_index(
                field_name="_id",
                index_type="STL_SORT"
            )
        index_params.add_index(field_name=embeddings_name,
                               index_name="sparse_inverted_index",
                               index_type="SPARSE_INVERTED_INDEX",
                               metric_type="IP",
                               params={"drop_ratio_build": 0.2})
        return index_params

    def create_fields(self, embeddings_name="embeddings", new_fields_only=False):
        fields = [] if new_fields_only else super().create_fields()
        self.embeddings_name = embeddings_name
        fields.append(
            FieldSchema(name=embeddings_name, dtype=DataType.SPARSE_FLOAT_VECTOR)
        )
        return fields

    def analyze(self, text):
        res = self.bm25_ef.encode_documents([text])
        return list(res[0])

    def get_search_params(self):
        search_params = get_param(self.config, 'search_params',
                                  self.milvus_defaults['search_params']["BM25"])

        return search_params

    def _analyze_data(self, texts):
        tm = timer()
        if os.path.exists(self.config.milvus_idf_file):
            ans=ask_for_confirmation(text=f"The file {self.config.milvus_idf_file} exists - recreate? (yes/No)",
                                 default="no")
            if ans == 'no':
                return
        print(f"Computing IDF - will might a while.", end="", flush=True)
        os.environ["TOKENIZERS_PARALLELISM"] = "false" # Prevent the system from complaining that we forked the tokenizer.
        self.bm25_ef.fit(texts)
        print(f" done in {tm.time_since_beginning()}.", flush=True)
        self.save_idf_index()

    def encode_data(self, texts, batch_size, show_progress_bar=True, tqdm_instance=None):
        # print("Computing embeddings for the input.")
        embeddings = []
        if tqdm_instance is None:
            t=tqdm(desc="Encoding documents (BM25)", total=len(texts), disable=not show_progress_bar)
        else:
            t=tqdm_instance
        for i in range(0, len(texts), batch_size):
            last = min(i + batch_size, len(texts))
            encs = self.bm25_ef.encode_documents(texts[i:last])
            # embeddings.extend([v for v in list(encs) if v.getnnz()>0])
            # embeddings.extend(list(encs))
            embeddings.extend(convert_to_single_vectors(encs))
            t.update(last-i)
        #embeddings = self.bm25_ef.encode_documents(texts)
        return embeddings

    def encode_query(self, question):
        return self.bm25_ef.encode_queries([question.text])[[0],:]

    def get_search_request(self, text):
        data = self.encode_data([text], batch_size=1, show_progress=False)
        search_params = self.get_search_params()

    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs):
        index_created = super().ingest(corpus, update, **kwargs)
        if index_created:
            self.save_idf_index()

    def save_idf_index(self):
        idf_file = self.config.milvus_idf_file
        if idf_file is None:
            idf_file = os.path.join(self.config.project_dir, f"{self.config.index_name}.idf")
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

