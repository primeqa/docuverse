import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

from tqdm import tqdm

try:
    import lancedb
    import pyarrow as pa
except ImportError:
    raise RuntimeError(
        "lancedb and pyarrow are required for LanceDB support. "
        "Install with: pip install lancedb"
    )

from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import get_param, _trim_json
from docuverse.utils.timer import timer
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction


class LanceDBEngine(RetrievalEngine):
    """
    LanceDB retrieval engine for document retrieval.

    Uses LanceDB as the underlying vector store with pre-computed embeddings
    from a DenseEmbeddingFunction (SentenceTransformers). Data is persisted
    in Lance columnar format on disk.
    """

    VECTOR_COLUMN = "vector"

    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)

        self.model = None
        self.hidden_dim = None
        self.db = None
        self.table = None

        self.load_model_config(config_params)
        self.extra_fields = get_param(self.config.data_template, "extra_fields", [])
        self.persist_directory = get_param(self.config, "project_dir", "/tmp")

        # Metric: "dot" matches IP used by Milvus/FAISS with normalized embeddings
        self.metric = get_param(config_params, "metric", "dot")

        self.init_model(**kwargs)

        if self.config.ingestion_batch_size == 40:
            self.config.ingestion_batch_size = self.config.bulk_batch

        self.init_client()

    def init_model(self, **kwargs):
        self.model = DenseEmbeddingFunction(
            self.config.model_name,
            **self.config.__dict__,
        )
        self.hidden_dim = len(self.model.encode(["text"], show_progress_bar=False)[0])

    def init_client(self):
        # Allow direct path override via 'server' config param
        db_path = get_param(self.config, "server", None)
        if not db_path:
            db_path = os.path.join(self.persist_directory, "lancedb_data")
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)

    def check_client(self):
        pass  # Local database, no connection to check

    # ===== Schema =====

    def _build_schema(self):
        """Build a PyArrow schema for the LanceDB table."""
        fields = [
            pa.field("id", pa.utf8()),
            pa.field("text", pa.utf8()),
            pa.field("title", pa.utf8()),
        ]
        for f in self.extra_fields:
            fields.append(pa.field(f, pa.utf8()))
        fields.append(
            pa.field(self.VECTOR_COLUMN, pa.list_(pa.float32(), list_size=self.hidden_dim))
        )
        return pa.schema(fields)

    # ===== Index Management =====

    def has_index(self, index_name: str) -> bool:
        return index_name in self.db.table_names()

    def create_index(self, index_name: str = None, **kwargs):
        if index_name is None:
            index_name = self.config.index_name
        schema = self._build_schema()
        self.table = self.db.create_table(index_name, schema=schema, mode="overwrite")

    def delete_index(self, index_name: str = None, fmt=None, **kwargs):
        if index_name is None:
            index_name = self.config.index_name
        if fmt:
            print(fmt.format(f"Table {index_name} exists, dropping"))
        if index_name in self.db.table_names():
            self.db.drop_table(index_name)
        self.table = None

    def _open_table(self):
        """Open the table if not already open.

        Also auto-detects the vector column name from the table schema,
        so tables created by milvus_copy_to_lancedb.py (which preserves
        original Milvus field names like 'embeddings') work correctly.
        """
        if self.table is None:
            self.table = self.db.open_table(self.config.index_name)
            # Auto-detect vector column if the default doesn't exist
            schema = self.table.schema
            field_names = [f.name for f in schema]
            if self.VECTOR_COLUMN not in field_names:
                for f in schema:
                    if pa.types.is_fixed_size_list(f.type) or pa.types.is_list(f.type):
                        self.VECTOR_COLUMN = f.name
                        break

    # ===== Ingestion =====

    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs) -> bool:
        fmt = "\n=== {:30} ==="
        fields = None  # LanceDB manages its own schema
        still_create_index = self.create_update_index(fmt=fmt, update=update, fields=fields)
        if not still_create_index:
            return None

        self._open_table()

        tm = timer("LanceDB::ingest")
        corpus_size = len(corpus)
        batch_size = self.config.ingestion_batch_size

        tq = tqdm(desc="Creating data", total=corpus_size, leave=True)
        tq1 = tqdm(desc="  * Encoding data", total=corpus_size, leave=False)
        tq2 = tqdm(desc="  * Ingesting data", total=corpus_size, leave=False)

        for i in range(0, corpus_size, batch_size):
            last = min(i + batch_size, corpus_size)
            data = self._create_data(corpus[i:last], tq_instance=tq1)
            tm.add_timing("encoding_data")

            self._insert_data(data, tq_instance=tq2)
            tm.add_timing("data_insertion")

            tq.update(last - i)

        # Optionally create ANN index for large datasets
        self._maybe_create_ann_index(corpus_size)

        print(f"Ingested {corpus_size} documents into LanceDB table {self.config.index_name}")
        return True

    def _create_data(self, corpus, tq_instance=None, **kwargs):
        """Encode a batch of documents and prepare for insertion."""
        texts = []
        records = []

        for doc in corpus:
            text = _trim_json(
                get_param(doc, "text", ""),
                max_string_len=self.config.max_text_size,
            )
            if not text:
                continue

            record = {
                "id": get_param(doc, "id", ""),
                "text": text,
                "title": get_param(doc, "title", ""),
            }
            for f in self.extra_fields:
                val = get_param(doc, f, "")
                if isinstance(val, dict | list):
                    val = json.dumps(_trim_json(val))
                else:
                    val = str(val)
                record[f] = val

            texts.append(text)
            records.append(record)

        if not texts:
            return []

        embeddings = self.model.encode(texts, show_progress_bar=False,
                                       _batch_size=len(texts))

        for record, emb in zip(records, embeddings):
            record[self.VECTOR_COLUMN] = np.array(emb, dtype=np.float32).tolist()

        if tq_instance:
            tq_instance.update(len(records))

        return records

    def _insert_data(self, data, tq_instance=None, **kwargs):
        if not data:
            return
        self.table.add(data)
        if tq_instance:
            tq_instance.update(len(data))

    def _maybe_create_ann_index(self, num_rows):
        """Create an ANN index if the dataset is large enough and configured."""
        index_params = get_param(self.config, "index_params", None)
        if index_params is None:
            return

        index_type = index_params if isinstance(index_params, str) else \
            get_param(index_params, "index_type", None)
        if index_type is None:
            return

        print(f"Creating {index_type} ANN index...")
        kwargs = {"metric": self.metric, "vector_column_name": self.VECTOR_COLUMN}
        if isinstance(index_params, dict):
            for k in ("num_partitions", "num_sub_vectors"):
                if k in index_params:
                    kwargs[k] = index_params[k]
        self.table.create_index(index_type=index_type, **kwargs)
        print("ANN index created.")

    # ===== Search =====

    def search(self, query: SearchQueries.Query, **kwargs) -> SearchResult:
        tm = timer("ingest_and_test::search::retrieve")
        self._open_table()
        tm.add_timing("open_table")

        query_embedding = self.model.encode(
            [query.text], show_progress_bar=False, prompt_name="query"
        )[0]
        tm.add_timing("encode")

        output_cols = ["id", "text", "title", "_distance"] + self.extra_fields
        results = (
            self.table.search(np.array(query_embedding, dtype=np.float32).tolist(),
                              vector_column_name=self.VECTOR_COLUMN)
            .select(output_cols)
            .limit(int(self.config.top_k))
            .to_list()
        )
        tm.add_timing("lancedb_search")

        retrieved_passages = []
        for r in results:
            # Convert distance to score.
            # For "dot" metric: LanceDB returns _distance = 1 - dot(q,d),
            # so score = 1 - _distance = dot(q,d).
            # For "cosine": _distance = 1 - cosine_sim, so score = 1 - _distance.
            # For "L2": score = 1 / (1 + _distance).
            dist = r.get("_distance", 0.0)
            if self.metric in ("dot", "cosine"):
                score = 1.0 - dist
            else:  # L2
                score = 1.0 / (1.0 + dist)

            passage = {
                "id": r.get("id", ""),
                "text": r.get("text", ""),
                "title": r.get("title", ""),
                "score": float(score),
            }
            for f in self.extra_fields:
                if f in r:
                    passage[f] = r[f]
            retrieved_passages.append(passage)

        res = SearchResult(query, retrieved_passages)
        tm.add_timing("result_construction")
        return res

    # ===== Info =====

    def info(self) -> Dict[str, Any]:
        info = {
            "retriever_type": "lancedb",
            "model": self.config.model_name,
            "index_name": self.config.index_name,
            "dimension": self.hidden_dim,
            "metric": self.metric,
            "persist_directory": self.persist_directory,
        }
        if self.table is not None:
            info["row_count"] = self.table.count_rows()
        return info
