from typing import List

from docuverse.engines.retrieval.milvus.milvus import MilvusEngine

try:
    from pymilvus import (
        FieldSchema, CollectionSchema, DataType, MilvusClient,
        Collection,
    )
    from pymilvus import model as MilvusModel
except:
    print(f"You need to install pymilvus to be using Milvus functionality!")
    raise RuntimeError("fYou need to install pymilvus to be using Milvus functionality!")
from docuverse.engines.search_engine_config_params import SearchEngineConfig
from docuverse.utils import get_param
from docuverse.utils.embeddings.sparse_embedding_function import SparseEmbeddingFunction


class MilvusSparseEngine(MilvusEngine):
    def __init__(self, config: SearchEngineConfig|dict, **kwargs) -> None:
        super().__init__(config, **kwargs)
        # milvus_client = MilvusClient("http://euler.watson.ibm.com:19530")
        #
        # print("    all collections    ")
        # print(milvus_client.list_collections())
        # has_collection = milvus_client.has_collection(collection_name, timeout=5)
        # if has_collection:
        #     print(f"schema of collection {collection_name}")
        #     print(milvus_client.describe_collection(collection_name))
        # else:
        #     fields = [
        #         FieldSchema(name="pk", dtype=DataType.VARCHAR,
        #                     is_primary=True, auto_id=True, max_length=100),
        #         FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
        #         FieldSchema(name="embeddings", dtype=DataType.SPARSE_FLOAT_VECTOR),
        #     ]
        #     schema = CollectionSchema(
        #         fields, "demo for using sparse float vector with milvus client")
            # index_params = milvus_client.prepare_index_params()
            # index_params.add_index(field_name="embeddings", index_name="sparse_inverted_index",
            #                        index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
            #                        params={"drop_ratio_build": 0.2})
            # milvus_client.create_collection(collection_name, schema=schema,
            #                                 index_params=index_params, timeout=5, consistency_level="Strong")

    def init_model(self, kwargs):
        self.model = SparseEmbeddingFunction(self.config.model_name)
        # self.model = MilvusModel.sparse.SpladeEmbeddingFunction(
        #     model_name="/dccstor/jsmc-nmt-01/llm/expt/RESEARCHONLY/models/splade-a-hfhn_s0_20240807_LR_2e-5_l_1e-2_S_150000_D_False_P_1_BS_16/",
        #     device="cpu"  # or "cuda"
        # )
        # self.hidden_dim = len(self.model.encode('text', show_progress_bar=False))

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

    def get_search_params(self):
        search_params = get_param(self.config, 'search_params',
                                  self.milvus_defaults['search_params']["SPLADE"])

        # search_params = {
        #     "metric_type": "IP",
        #     "params": {
        #         "drop_ratio_search": 0.2,
        #     }
        # }

        return search_params

    # def encode_and_index(self,
    #                      collection_name: str,
    #                      splade_enc: MilvusModel.sparse.SpladeEmbeddingFunction,
    #                      milvus_client: MilvusClient,
    #                      docs: List[str]
    #                      ):
    #     docs_embeddings = splade_enc.encode_documents(docs)
    #     rows = [
    #         {
    #             "embeddings": docs_embeddings[i],
    #             "text": docs[i]
    #         } for i in range(len(docs))
    #     ]
    #     insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)
    #     print(insert_result)

    # def query(collection_name: str,
    #           splade_enc: MilvusClient.sparse.SpladeEmbeddingFunction,
    #           milvus_client: MilvusClient,
    #           queries: List[str]):
    #     queries_embeddings = splade_enc.encode_queries(queries)
    #     search_params = {
    #         "metric_type": "IP",
    #         "params": {
    #             "drop_ratio_search": 0.2,
    #         }
    #     }
    #
    #     # no need to specify anns_field for collections with only 1 vector field
    #     result = milvus_client.search(collection_name, queries_embeddings, limit=4, output_fields=[
    #         "pk", "text"], search_params=search_params)
    #     for ir, hits in enumerate(result):
    #         for ih, hit in enumerate(hits):
    #             print(f"hit: {ir} {ih} {hit}")
    #
    @classmethod
    def test(cls):
        collection_name = "hello_sparse"
        docs = [
            "Artificial intelligence was founded as an academic discipline in 1956.",
            "Alan Turing was the first person to conduct substantial research in AI.",
            "Born in Maida Vale, London, Turing was raised in southern England.",
            "The quick setup collection has two mandatory fields: the primary and vector fields. It also allows the insertion of undefined fields and their values in key-value pairs in a dynamic field."
        ]
        queries = ["Who is Alan Turing?", "What is quick setup?"]

        splade_enc, milvus_client = cls.setup(collection_name)
        cls.encode_and_index(collection_name, splade_enc, milvus_client, docs)
        cls.query(collection_name, splade_enc, milvus_client, queries)
        milvus_client.drop_collection(collection_name)

    # def ingest(self, corpus: SearchCorpus, update: bool = False):
    #     fmt = "\n=== {:30} ==="
    #     fields = [
    #         FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, description="ID", auto_id=True),
    #         FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=False, max_length=1000, auto_id=False),
    #         FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
    #         FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=10000),
    #         FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.hidden_dim)
    #     ]
    #     for f in self.config.data_template.extra_fields:
    #         fields.append(FieldSchema(name=f, dtype=DataType.VARCHAR, max_length=10000))
    #     # collection_loaded = utility.has_collection(self.config.index_name)
    #     collection_loaded = self.client.has_collection(self.config.index_name)
    #     if collection_loaded:
    #         print(fmt.format(f"Collection {self.config.index_name} exists, dropping"))
    #         self.client.drop_collection(self.config.index_name)
    #         # utility.drop_collection(self.config.index_name)
    #     schema = CollectionSchema(fields, self.config.index_name)
    #     print(fmt.format(f"Create collection `{self.config.index_name}`"))
    #
    #     index_params = self.client.prepare_index_params()
    #
    #     index_params.add_index(
    #         field_name="_id",
    #         index_type="STL_SORT"
    #     )
    #
    #     index_params.add_index(
    #         field_name="embeddings",
    #         index_type="IVF_FLAT",
    #         metric_type="IP",
    #         params={"nlist": 128}
    #     )
    #
    #     # self.index = Collection(self.config.index_name, schema, consistency_level="Strong", index_file_size=128)
    #     self.client.create_collection(self.config.index_name, schema=schema, index_params=index_params)
    #
    #     text_vectors = [row['text'] for row in corpus]
    #     batch_size = get_param(self.config, 'bulk_batch', 40)
    #     passage_vectors = self.model.encode(text_vectors,
    #                                         _batch_size=batch_size,
    #                                         show_progress_bar=True)
    #     # create embeddings
    #     batch_size = 1000
    #     data = []
    #     for i, item in enumerate(corpus):
    #         dt = {"id": item["id"], "embeddings": passage_vectors[i], "text": item['text'], 'title': item['title']}
    #         for f in self.config.data_template.extra_fields:
    #             dt[f] = str(item[f])
    #         data.append(dt)
    #
    #     for i in tqdm(range(0, len(data), batch_size), desc="Ingesting documents"):
    #         self.client.insert(collection_name=self.config.index_name, data=data[i:i + batch_size])
    #     self.client.create_index(collection_name=self.config.index_name, index_params=index_params)
