"""LanceDB base engine — shared client, schema plumbing, ingestion loop.

Subclasses contribute via four hooks:
  - extra_schema_fields()   columns this engine adds to the shared table
  - encode_data(texts)      values written to those columns at ingest time
  - encode_query(text)      what to send to LanceDB at search time
  - build_indexes()         post-ingest index creation (ANN, FTS, no-op)
  - search(query)           engine-specific retrieval
"""
import os
import json
from typing import Any, Dict, List, Optional

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
from docuverse.utils import get_param, _trim_json
from docuverse.utils.timer import timer


class LanceDBEngine(RetrievalEngine):
    """Base class for LanceDB-backed engines.

    Owns the lancedb client and the shared table. Subclasses add columns
    via ``extra_schema_fields`` and write values via ``encode_data``.
    """

    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)
        self.db = None
        self.table = None
        # load_model_config must come before any self.config access.
        self.load_model_config(config_params)
        self.extra_fields = get_param(self.config.data_template, "extra_fields", [])
        self.persist_directory = get_param(self.config, "project_dir", "/tmp")
        self.metric = get_param(config_params, "metric", "dot")

        self.init_model(**kwargs)

        if self.config.ingestion_batch_size == 40:
            self.config.ingestion_batch_size = self.config.bulk_batch

        self.init_client()

    # ===== Connection =====

    def init_model(self, **kwargs):
        """No-op default; dense/sparse subclasses override."""
        return

    def init_client(self):
        db_path = get_param(self.config, "server", None)
        if not db_path:
            db_path = os.path.join(self.persist_directory, "lancedb_data")
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)

    def check_client(self):
        pass

    # ===== Hooks for subclasses =====

    def extra_schema_fields(self) -> List[pa.Field]:
        """PyArrow fields this engine adds beyond id/text/title/extra_fields."""
        return []

    def encode_data(self, texts: List[str], tm=None, **kwargs):
        """Encode a batch of texts into this engine's column value(s).

        Returns a list of length ``len(texts)``; each item is whatever
        belongs in this engine's column for that row.
        """
        return [None] * len(texts)

    def encode_query(self, question, tm=None):
        """Encode a query into whatever ``search`` accepts."""
        return question.text if hasattr(question, "text") else question

    def build_indexes(self):
        """Create indexes on the shared table after ingestion. Default no-op."""
        return

    # ===== Schema =====

    def _base_schema_fields(self) -> List[pa.Field]:
        fields = [
            pa.field("id", pa.utf8()),
            pa.field("text", pa.utf8()),
            pa.field("title", pa.utf8()),
        ]
        for f in self.extra_fields:
            fields.append(pa.field(f, pa.utf8()))
        return fields

    def _build_schema(self) -> pa.Schema:
        return pa.schema(self._base_schema_fields() + self.extra_schema_fields())

    # ===== Index management =====

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
        if self.table is None:
            self.table = self.db.open_table(self.config.index_name)

    # ===== Ingestion =====

    def _base_record(self, doc) -> Optional[dict]:
        text = _trim_json(
            get_param(doc, "text", ""),
            max_string_len=self.config.max_text_size,
        )
        if not text:
            return None
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
        return record

    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs) -> bool:
        fmt = "\n=== {:30} ==="
        still_create_index = self.create_update_index(fmt=fmt, update=update, fields=None)
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
            data = self._create_data(corpus[i:last], tq_instance=tq1, tm=tm)
            tm.add_timing("encoding_data")
            self._insert_data(data, tq_instance=tq2)
            tm.add_timing("data_insertion")
            tq.update(last - i)

        self.build_indexes()
        print(f"Ingested {corpus_size} documents into LanceDB table {self.config.index_name}")
        return True

    def _create_data(self, corpus, tq_instance=None, tm=None, **kwargs):
        records = []
        texts = []
        for doc in corpus:
            base = self._base_record(doc)
            if base is None:
                continue
            records.append(base)
            texts.append(base["text"])
        if not records:
            return []
        encoded = self.encode_data(texts, tm=tm)
        col_names = [f.name for f in self.extra_schema_fields()]
        for record, enc in zip(records, encoded):
            self._assign_encoded(record, enc, col_names)
        if tq_instance:
            tq_instance.update(len(records))
        return records

    def _assign_encoded(self, record: dict, enc: Any, col_names: List[str]):
        """Place encoded data into the right column(s).

        Default behaviour: if this engine declares one column, ``enc`` IS the
        column value. Subclasses with multiple or differently-named columns
        override this.
        """
        if not col_names:
            return
        if len(col_names) == 1:
            record[col_names[0]] = enc
            return
        raise NotImplementedError(
            f"{type(self).__name__} declares {len(col_names)} columns "
            f"but does not override _assign_encoded()."
        )

    def _insert_data(self, data, tq_instance=None, **kwargs):
        if not data:
            return
        self.table.add(data)
        if tq_instance:
            tq_instance.update(len(data))

    # ===== Info =====

    def info(self) -> Dict[str, Any]:
        info = {
            "retriever_type": type(self).__name__.lower(),
            "index_name": self.config.index_name,
            "metric": self.metric,
            "persist_directory": self.persist_directory,
        }
        if self.table is not None:
            info["row_count"] = self.table.count_rows()
        return info
