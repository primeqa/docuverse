from docuverse.engines.retrieval.milvus.milvus import MilvusEngine

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
    def __init__(self, config: SearchEngineConfig|dict, **kwargs) -> None:
        self.normalize_embs = False
        self.hidden_dim = 0
        super().__init__(config, **kwargs)

    def init_model(self, kwargs):
        self.model = DenseEmbeddingFunction(self.config.model_name)
        self.hidden_dim = len(self.model.encode(['text'], show_progress_bar=False)[0])
        self.normalize_embs = get_param(kwargs, 'normalize_embs', False)

    def prepare_index_params(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="_id",
            index_type="STL_SORT"
        )
        index_params.add_index(
            field_name="embeddings",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128}
        )
        return index_params

    def create_fields(self, embedding_name="embeddings"):
        fields = super().create_fields()
        fields.append(
            FieldSchema(name=embedding_name, dtype=DataType.FLOAT_VECTOR, dim=self.hidden_dim)
        )
        return fields

    def get_search_params(self):
        search_params = get_param(self.config, 'search_params',
                                  self.milvus_defaults['search_params']["HNSW"])
        return search_params


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
