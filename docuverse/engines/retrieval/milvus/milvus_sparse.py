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
        self.build_config = get_param(self.config, 'sparse_config')
        self.search_config = get_param(self.config, 'search_config')

    def init_model(self, **kwargs):
        self.model = SparseEmbeddingFunction(self.config.model_name, batch_size=self.config.bulk_batch,
                                             doc_max_tokens=get_param(self.config, 'sparse_config.doc_max_tokens', None),
                                             query_max_tokens=get_param(self.config, 'sparse_config.query_max_tokens', None),
                                             process_name="ingest_and_test::search"
                                             )

    def prepare_index_params(self, embeddings_name="embeddings"):
        index_params = self.client.prepare_index_params()
        if self.server.type != "file":
            index_params.add_index(
                field_name="_id",
                index_type="STL_SORT"
            )
        # index_build_vals = get_param(self.config, 'index_build_vals', None)
        build_configuration_name = get_param(self.config, 'sparse_config.sparse_build_config', 'SPLADE')
        build_config = get_param(self.config, 'index_params',
                                 self.milvus_defaults['index_params'][build_configuration_name])

        if isinstance(build_config, str):
            build_config = get_param(self.milvus_defaults['index_params'], build_config)
        index_params.add_index(field_name=embeddings_name,
                               **build_config
                               )
        return index_params

    def create_fields(self, embeddings_name="embeddings", new_fields_only=False):
        fields = [] if new_fields_only else super().create_fields()
        self.embeddings_name = embeddings_name
        fields.append(
            FieldSchema(name=embeddings_name, dtype=DataType.SPARSE_FLOAT_VECTOR)
        )
        return fields

    def get_search_params(self):
        search_config_name = get_param(self.config, 'sparse_config.sparse_search_config', 'SPLADE')
        search_params = get_param(self.config, 'search_params',
                                  self.milvus_defaults['search_params'][search_config_name])

        return search_params

    def encode_query(self, question):
        query_vector = self.model.encode_query(query=question.text,
                                               encode_question=get_param(self.config,
                                                                        'sparse_config.runtime_query_encoding',
                                                                         True)
                                               )
        return query_vector

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

