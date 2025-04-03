from docuverse.engines.retrieval.milvus.milvus import MilvusEngine
import numpy as np

try:
    from pymilvus import (
        FieldSchema, CollectionSchema, DataType,
        Collection,
    )
except:
    print(f"You need to install pymilvus to be using Milvus functionality!")
    raise RuntimeError("fYou need to install pymilvus to be using Milvus functionality!")
from docuverse.engines.search_engine_config_params import SearchEngineConfig
from docuverse.utils import get_param
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction


class MilvusDenseEngine(MilvusEngine):
    BF16=0
    FP16=1
    FP32=2
    STORAGE_MAP = {
        "bf16": (BF16, float, DataType.BFLOAT16_VECTOR),
        "fp16": (FP16, np.float16, DataType.FLOAT16_VECTOR),
        "fp32": (FP32, np.float32, DataType.FLOAT_VECTOR)
    }


    def __init__(self, config: SearchEngineConfig|dict, **kwargs) -> None:
        self.embeddings_name = None
        self.normalize_embs = False
        self.hidden_dim = 0
        super().__init__(config, **kwargs)
        self.storage_size = get_param([config, kwargs], 'storage_size', "fp16")
        self.storage_rep = self.STORAGE_MAP[self.storage_size]

    def init_model(self, **kwargs):
        self.model = DenseEmbeddingFunction(self.config.model_name)
        self.hidden_dim = len(self.model.encode(['text'], show_progress_bar=False)[0])
        self.normalize_embs = get_param(kwargs, 'normalize_embs', False)

    def prepare_index_params(self, embeddings_name="embeddings"):
        index_params = self.client.prepare_index_params()
        if self.server.type != "file":
            index_params.add_index(
                field_name="_id",
                index_type="STL_SORT"
            )

        _basic_index_params = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"nlist": 1024,
                    "M": 128,
                    "efConstruction": 128}
        }
        _default_index_params = get_param(self.milvus_defaults, "search_params.default_dense", _basic_index_params)
        _index_params = get_param(self.config, 'index_params', _default_index_params)
        if _index_params is None:
            _index_params = _basic_index_params
        import json
        print(f"Index params: {json.dumps(_index_params, indent=2)}")
        index_params.add_index(
            field_name=embeddings_name,
            **_index_params
            # index_type="HNSW",
            # metric_type="IP",
            # params={"nlist": 1024,
            #         "M": 128,
            #         "efConstruction": 128}
        )
        self.index_type = _index_params["index_type"]
        return index_params

    def create_fields(self, embeddings_name="embeddings", new_fields_only=False):
        fields = [] if new_fields_only else super().create_fields()
        self.embeddings_name = embeddings_name
        fields.append(
            FieldSchema(name=embeddings_name, dtype=self.storage_rep[2], dim=self.hidden_dim)
        )
        return fields

    def _create_collection(self, fields, fmt):
        self.collection = \
            self.client.create_collection(self.config.index_name,
                                          dimension=self.hidden_dim,
                                          schema=CollectionSchema(fields, self.config.index_name),
                                          index_params=self.prepare_index_params(embeddings_name=self.embeddings_name),
                                          vector_field_name=self.embeddings_name,
                                          consistency_level="Eventually"
                                          )

    def get_search_params(self):
        search_params = get_param(self.config, 'search_params',
                                  self.milvus_defaults['search_params']["HNSW"])
        return search_params

    def encode_data(self, texts, batch_size, **kwargs):
        passage_vectors = super().encode_data(texts=texts, batch_size=batch_size, **kwargs)
        if self.storage_size != "fp32":
            passage_vectors = [np.array(p, dtype=self.storage_rep[1]) for p in passage_vectors]
        return passage_vectors

    def encode_query(self, question):
        return  self.encode_data([question.text], batch_size=1, prompt_name=self.config.query_prompt_name)[0]


