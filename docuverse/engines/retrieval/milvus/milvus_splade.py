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
from docuverse.utils.embeddings.splade3_embedding_function import SpladeEmbeddingFunction


class MilvusSpladeEngine(MilvusEngine):
    def __init__(self, config: SearchEngineConfig|dict, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.embeddings_name = "embeddings"

    def init_model(self, kwargs):
        self.model = SpladeEmbeddingFunction(self.config.model_name, batch_size=self.config.bulk_batch)

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
        fields.append(
            FieldSchema(name="embeddings", dtype=DataType.SPARSE_FLOAT_VECTOR)
        )
        return fields

    def get_search_params(self):
        search_params = get_param(self.config, 'search_params',
                                  self.milvus_defaults['search_params']["SPLADE"])

        return search_params

    def encode_query(self, question):
        query_vector = self.model.encode_query([question.text], show_progress_bar=False)
        return list(query_vector)[0]

    @classmethod
    def test(cls):
        collection_name = "hello_splade"
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

