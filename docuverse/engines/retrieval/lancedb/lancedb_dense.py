"""Dense LanceDB engine — one dense vector column per instance."""
import numpy as np
import pyarrow as pa
from typing import Any, Dict, List

from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import get_param
from docuverse.utils.timer import timer
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction


class LanceDBDenseEngine(LanceDBEngine):
    """Dense vector retrieval over a single embeddings column."""

    DEFAULT_VECTOR_COLUMN = "vector"

    def __init__(self, config_params, **kwargs):
        self.model = None
        self.hidden_dim = None
        self.embeddings_name = get_param(config_params, "embeddings_name",
                                         self.DEFAULT_VECTOR_COLUMN)
        super().__init__(config_params, **kwargs)

    def init_model(self, **kwargs):
        self.model = DenseEmbeddingFunction(
            self.config.model_name,
            **self.config.__dict__,
        )
        self.hidden_dim = self.model.embedding_dim

    # ===== Hooks =====

    def extra_schema_fields(self):
        return [pa.field(
            self.embeddings_name,
            pa.list_(pa.float32(), list_size=self.hidden_dim),
        )]

    def encode_data(self, texts, tm=None, **kwargs):
        embs = self.model.encode(
            texts, show_progress_bar=False, _batch_size=len(texts), tm=tm
        )
        return [np.array(e, dtype=np.float32).tolist() for e in embs]

    def encode_query(self, question, tm=None):
        text = question.text if hasattr(question, "text") else question
        emb = self.model.encode(
            [text], show_progress_bar=False, prompt_name="query", tm=tm
        )[0]
        return np.array(emb, dtype=np.float32).tolist()

    def build_indexes(self):
        self._open_table()
        index_params = get_param(self.config, "index_params", None)
        if index_params is None:
            return
        index_type = index_params if isinstance(index_params, str) else \
            get_param(index_params, "index_type", None)
        if index_type is None:
            return
        kwargs = {"metric": self.metric, "vector_column_name": self.embeddings_name}
        if isinstance(index_params, dict):
            for k in ("num_partitions", "num_sub_vectors"):
                if k in index_params:
                    kwargs[k] = index_params[k]
        print(f"Creating {index_type} ANN index on {self.embeddings_name}...")
        self.table.create_index(index_type=index_type, **kwargs)
        print("ANN index created.")

    # ===== Search =====

    def _open_table(self):
        """Open table; auto-detect vector column for tables created by
        milvus_copy_to_lancedb.py (which preserves original Milvus field names).
        """
        if self.table is None:
            self.table = self.db.open_table(self.config.index_name)
            field_names = [f.name for f in self.table.schema]
            if self.embeddings_name not in field_names:
                for f in self.table.schema:
                    if pa.types.is_fixed_size_list(f.type) or pa.types.is_list(f.type):
                        self.embeddings_name = f.name
                        break

    def search(self, query: SearchQueries.Query, **kwargs):
        tm = timer("ingest_and_test::search::retrieve")
        self._open_table()
        tm.add_timing("open_table")
        qvec = self.encode_query(query, tm=tm)
        tm.add_timing("encode")

        output_cols = ["id", "text", "title", "_distance"] + self.extra_fields
        results = (
            self.table.search(qvec, vector_column_name=self.embeddings_name)
            .select(output_cols)
            .limit(int(self.config.top_k))
            .to_list()
        )
        tm.add_timing("lancedb_search")

        passages = []
        for r in results:
            dist = r.get("_distance", 0.0)
            if self.metric in ("dot", "cosine"):
                score = 1.0 - dist
            else:
                score = 1.0 / (1.0 + dist)
            p = {
                "id": r.get("id", ""),
                "text": r.get("text", ""),
                "title": r.get("title", ""),
                "score": float(score),
            }
            for f in self.extra_fields:
                if f in r:
                    p[f] = r[f]
            passages.append(p)
        res = SearchResult(query, passages)
        tm.add_timing("result_construction")
        return res

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info["model"] = self.config.model_name
        info["dimension"] = self.hidden_dim
        info["embeddings_name"] = self.embeddings_name
        return info
