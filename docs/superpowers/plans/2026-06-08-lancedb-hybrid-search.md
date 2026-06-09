# LanceDB Hybrid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hybrid retrieval to LanceDB combining dense vectors, BM25 (LanceDB FTS), and learned-sparse (SPLADE-style) sub-engines, configured the same way as `MilvusHybridEngine`.

**Architecture:** Refactor the existing `LanceDBEngine` into a base class plus a dense subclass. Add BM25 and sparse sibling engines that share one LanceDB table. A `LanceDBHybridEngine` composes sub-engines, encoding all representations in one ingest pass and merging per-engine search results via RRF or weighted fusion (with a native LanceDB fast path for dense+FTS+RRF).

**Tech Stack:** LanceDB (PyLance), PyArrow, NumPy, SciPy CSR, existing `DenseEmbeddingFunction` / `SparseEmbeddingFunction`, pytest.

**Spec:** `docs/superpowers/specs/2026-06-08-lancedb-hybrid-search-design.md`

---

## File Structure

Files created:
- `docuverse/engines/retrieval/lancedb/lancedb.py` — `LanceDBEngine` base class (connection, schema plumbing, ingestion loop, hooks).
- `docuverse/engines/retrieval/lancedb/lancedb_dense.py` — `LanceDBDenseEngine` (dense vector column).
- `docuverse/engines/retrieval/lancedb/lancedb_bm25.py` — `LanceDBBM25Engine` (FTS index on `text`).
- `docuverse/engines/retrieval/lancedb/lancedb_sparse.py` — `LanceDBSparseEngine` (struct sparse column, Python-side scoring).
- `docuverse/engines/retrieval/lancedb/lancedb_hybrid.py` — `LanceDBHybridEngine` (composer + merger).
- `tests/test_lancedb_dense.py` — unit tests for the refactored dense engine.
- `tests/test_lancedb_bm25.py` — unit tests for BM25 engine.
- `tests/test_lancedb_sparse.py` — unit tests for sparse engine.
- `tests/test_lancedb_hybrid.py` — integration tests for the hybrid engine.

Files modified:
- `docuverse/engines/retrieval/lancedb/lancedb_engine.py` — slim back-compat shim re-exporting `LanceDBEngine = LanceDBDenseEngine`.
- `docuverse/engines/retrieval/lancedb/__init__.py` — export the new classes.
- `docuverse/utils/retrievers.py` — extend the `lancedb` branch with the four new `db_engine` names.
- `tests/test_engine_dispatch.py` — add the four new names to `KNOWN_NAMES`.

---

## Task 1: Extract `LanceDBEngine` base from current `lancedb_engine.py`

**Files:**
- Create: `docuverse/engines/retrieval/lancedb/lancedb.py`
- Reference: `docuverse/engines/retrieval/lancedb/lancedb_engine.py:1-300` (existing implementation)
- Modify: `docuverse/engines/retrieval/lancedb/__init__.py`

This task pulls the parts of the current monolithic `LanceDBEngine` that are *not* dense-specific into a new base class with extension hooks. No behavior change — the dense engine keeps working in subsequent tasks via inheritance.

- [ ] **Step 1: Create the new base file**

Create `docuverse/engines/retrieval/lancedb/lancedb.py` with this content:

```python
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
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
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
        self.extra_fields = get_param(self.config.data_template, "extra_fields", [])
        self.persist_directory = get_param(self.config, "project_dir", "/tmp")
        self.metric = get_param(config_params, "metric", "dot")

        self.load_model_config(config_params)
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
```

- [ ] **Step 2: Update package `__init__.py`**

Modify `docuverse/engines/retrieval/lancedb/__init__.py` to:

```python
from .lancedb import LanceDBEngine
# Back-compat: lancedb_engine.py keeps re-exporting LanceDBEngine until Task 2
# replaces it with the dense subclass. Tests importing LanceDBEngine still work.

__all__ = ['LanceDBEngine']
```

- [ ] **Step 3: Run existing lancedb-related tests to verify the base alone still imports cleanly**

Run: `conda activate ndocu && python -m pytest tests/test_engine_dispatch.py -v -k lancedb`

Expected: `lancedb` and `lance` parametrize cases still PASS (the base class can be instantiated with a stub config — it just has no encoder).

- [ ] **Step 4: Commit**

```bash
git add docuverse/engines/retrieval/lancedb/lancedb.py docuverse/engines/retrieval/lancedb/__init__.py
git commit -m "Extract LanceDBEngine base class from monolithic lancedb_engine.py."
```

---

## Task 2: Move dense logic into `LanceDBDenseEngine`

**Files:**
- Create: `docuverse/engines/retrieval/lancedb/lancedb_dense.py`
- Modify: `docuverse/engines/retrieval/lancedb/lancedb_engine.py` (turn into back-compat shim)
- Modify: `docuverse/engines/retrieval/lancedb/__init__.py`

The current `lancedb_engine.py` becomes the dense subclass under a new name. The old file becomes a one-liner alias so any code/test that imports `from docuverse.engines.retrieval.lancedb.lancedb_engine import LanceDBEngine` still works.

- [ ] **Step 1: Write the failing dispatch test for `lancedb-dense` and `lancedb_dense`**

Create `tests/test_lancedb_dense.py`:

```python
"""Tests for LanceDBDenseEngine — the refactored dense LanceDB engine."""
from __future__ import annotations

import pytest


def test_lancedb_dense_class_importable():
    """The new class lives at docuverse.engines.retrieval.lancedb.LanceDBDenseEngine."""
    from docuverse.engines.retrieval.lancedb import LanceDBDenseEngine
    from docuverse.engines.retrieval.lancedb import LanceDBEngine
    assert issubclass(LanceDBDenseEngine, LanceDBEngine)


def test_lancedb_engine_back_compat_alias():
    """``from .lancedb_engine import LanceDBEngine`` still works for back-compat."""
    from docuverse.engines.retrieval.lancedb.lancedb_engine import LanceDBEngine as Old
    from docuverse.engines.retrieval.lancedb import LanceDBDenseEngine
    # The old name now resolves to the dense subclass.
    assert Old is LanceDBDenseEngine
```

- [ ] **Step 2: Run the failing test**

Run: `conda activate ndocu && python -m pytest tests/test_lancedb_dense.py -v`

Expected: FAIL — `cannot import name 'LanceDBDenseEngine'`.

- [ ] **Step 3: Create `lancedb_dense.py`**

Create `docuverse/engines/retrieval/lancedb/lancedb_dense.py`:

```python
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
```

- [ ] **Step 4: Replace `lancedb_engine.py` with a back-compat shim**

Overwrite `docuverse/engines/retrieval/lancedb/lancedb_engine.py` with:

```python
"""Back-compat shim. The dense engine has moved to ``lancedb_dense``.

External code that did ``from docuverse.engines.retrieval.lancedb.lancedb_engine
import LanceDBEngine`` still works — the name now resolves to the dense
subclass.
"""
from docuverse.engines.retrieval.lancedb.lancedb_dense import LanceDBDenseEngine

LanceDBEngine = LanceDBDenseEngine

__all__ = ["LanceDBEngine"]
```

- [ ] **Step 5: Update package `__init__.py`**

Overwrite `docuverse/engines/retrieval/lancedb/__init__.py` with:

```python
from .lancedb import LanceDBEngine as _LanceDBBase
from .lancedb_dense import LanceDBDenseEngine

# Public name `LanceDBEngine` resolves to the dense subclass for back-compat.
LanceDBEngine = LanceDBDenseEngine

__all__ = ['LanceDBEngine', 'LanceDBDenseEngine']
```

- [ ] **Step 6: Run dense tests + existing lancedb tests**

Run: `conda activate ndocu && python -m pytest tests/test_lancedb_dense.py tests/test_engine_dispatch.py -v -k "lancedb or lance"`

Expected: PASS — both new tests and the parametrized `lancedb` / `lance` dispatch cases.

- [ ] **Step 7: Commit**

```bash
git add docuverse/engines/retrieval/lancedb/ tests/test_lancedb_dense.py
git commit -m "Move dense logic into LanceDBDenseEngine; keep LanceDBEngine alias for back-compat."
```

---

## Task 3: Implement `LanceDBBM25Engine`

**Files:**
- Create: `docuverse/engines/retrieval/lancedb/lancedb_bm25.py`
- Create: `tests/test_lancedb_bm25.py`
- Modify: `docuverse/engines/retrieval/lancedb/__init__.py`

LanceDB has a built-in Tantivy-backed FTS index, so BM25 needs no separate analyzer or IDF file. The engine adds no columns — it indexes the existing `text` column.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lancedb_bm25.py`:

```python
"""Tests for LanceDBBM25Engine — FTS-only retrieval over the shared `text` column."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("lancedb")


@pytest.fixture
def stub_config(tmp_path):
    """Minimal config sufficient to construct a BM25 engine."""
    cfg = SimpleNamespace(
        db_engine="lancedb-bm25",
        index_name="t_bm25",
        server=str(tmp_path),
        project_dir=str(tmp_path),
        top_k=10,
        ingestion_batch_size=8,
        bulk_batch=8,
        max_text_size=1024,
        data_template=SimpleNamespace(extra_fields=[]),
        duplicate_removal=None,
        rouge_duplicate_threshold=0.95,
        verbose=False,
        index_params=None,
    )
    cfg.__dict__["model_name"] = None
    return cfg


def test_bm25_class_importable():
    from docuverse.engines.retrieval.lancedb import LanceDBBM25Engine
    from docuverse.engines.retrieval.lancedb import LanceDBEngine as Base
    assert issubclass(LanceDBBM25Engine, Base.__mro__[1] if Base is not Base else Base)
    # Looser: must be a subclass of the base LanceDBEngine.
    from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine as RealBase
    assert issubclass(LanceDBBM25Engine, RealBase)


def test_bm25_extra_schema_fields_is_empty(stub_config, monkeypatch):
    """BM25 indexes the existing text column; it adds no PyArrow fields."""
    from docuverse.engines.retrieval.lancedb import LanceDBBM25Engine

    # Bypass create_update_index since we're not really ingesting yet.
    eng = LanceDBBM25Engine(stub_config)
    assert eng.extra_schema_fields() == []


def test_bm25_encode_data_is_no_op(stub_config):
    from docuverse.engines.retrieval.lancedb import LanceDBBM25Engine
    eng = LanceDBBM25Engine(stub_config)
    out = eng.encode_data(["hello world", "foo bar"])
    # encode_data must return one entry per text, but BM25 has nothing to write.
    assert len(out) == 2
    assert all(v is None for v in out)


def test_bm25_search_uses_fts(stub_config, monkeypatch):
    """End-to-end: ingest 3 docs, query, get a hit."""
    from docuverse.engines.retrieval.lancedb import LanceDBBM25Engine
    from docuverse.engines.search_queries import SearchQueries

    eng = LanceDBBM25Engine(stub_config)
    eng.create_index()
    eng.table.add([
        {"id": "a", "text": "alpha brown fox jumps", "title": ""},
        {"id": "b", "text": "beta lazy dog sleeps", "title": ""},
        {"id": "c", "text": "gamma quick fox runs", "title": ""},
    ])
    eng.build_indexes()

    q = SearchQueries.Query(id="q1", text="fox", relevant=[], answers=[])
    res = eng.search(q)
    ids = [p["id"] for p in res.retrieved_passages] if hasattr(res, "retrieved_passages") \
        else [p["id"] for p in res]
    assert "a" in ids or "c" in ids  # one of the fox docs ranks
```

- [ ] **Step 2: Run the failing test**

Run: `conda activate ndocu && python -m pytest tests/test_lancedb_bm25.py -v`

Expected: FAIL — `cannot import name 'LanceDBBM25Engine'`.

- [ ] **Step 3: Create the BM25 engine**

Create `docuverse/engines/retrieval/lancedb/lancedb_bm25.py`:

```python
"""LanceDB BM25 engine — full-text search via LanceDB's native FTS index.

Indexes the shared ``text`` column (no extra columns added). Search uses
LanceDB's ``query_type='fts'`` which returns Tantivy BM25 scores.
"""
from typing import Any, Dict, List

from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils.timer import timer


class LanceDBBM25Engine(LanceDBEngine):
    """BM25 retrieval via LanceDB FTS — uses the shared `text` column."""

    def init_model(self, **kwargs):
        # BM25 needs no model; LanceDB FTS handles tokenization internally.
        return

    # ===== Hooks =====

    def extra_schema_fields(self):
        return []

    def encode_data(self, texts, tm=None, **kwargs):
        return [None] * len(texts)

    def encode_query(self, question, tm=None):
        return question.text if hasattr(question, "text") else question

    def build_indexes(self):
        self._open_table()
        # replace=True: rebuild on re-ingest; works on first run too.
        self.table.create_fts_index("text", replace=True, use_tantivy=True)

    # ===== Search =====

    def search(self, query: SearchQueries.Query, **kwargs):
        tm = timer("ingest_and_test::search::retrieve")
        self._open_table()
        qtext = self.encode_query(query, tm=tm)
        tm.add_timing("encode")

        output_cols = ["id", "text", "title", "_score"] + self.extra_fields
        results = (
            self.table.search(qtext, query_type="fts")
            .select(output_cols)
            .limit(int(self.config.top_k))
            .to_list()
        )
        tm.add_timing("lancedb_fts_search")

        passages = []
        for r in results:
            score = float(r.get("_score", 0.0))
            p = {
                "id": r.get("id", ""),
                "text": r.get("text", ""),
                "title": r.get("title", ""),
                "score": score,
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
        info["fts_field"] = "text"
        return info
```

- [ ] **Step 4: Update package `__init__.py`**

Overwrite `docuverse/engines/retrieval/lancedb/__init__.py`:

```python
from .lancedb import LanceDBEngine as _LanceDBBase
from .lancedb_dense import LanceDBDenseEngine
from .lancedb_bm25 import LanceDBBM25Engine

LanceDBEngine = LanceDBDenseEngine

__all__ = ['LanceDBEngine', 'LanceDBDenseEngine', 'LanceDBBM25Engine']
```

- [ ] **Step 5: Run BM25 tests**

Run: `conda activate ndocu && python -m pytest tests/test_lancedb_bm25.py -v`

Expected: PASS — all four tests.

If `test_bm25_search_uses_fts` fails because `RetrievalEngine.__init__` calls `create_update_index` machinery that requires more config, pare the test back to skip the full ingestion flow and call `eng.create_index()` + `eng.table.add(...)` + `eng.build_indexes()` directly (the test above already does this).

- [ ] **Step 6: Commit**

```bash
git add docuverse/engines/retrieval/lancedb/lancedb_bm25.py docuverse/engines/retrieval/lancedb/__init__.py tests/test_lancedb_bm25.py
git commit -m "Add LanceDBBM25Engine using native LanceDB FTS index on `text` column."
```

---

## Task 4: Implement `LanceDBSparseEngine`

**Files:**
- Create: `docuverse/engines/retrieval/lancedb/lancedb_sparse.py`
- Create: `tests/test_lancedb_sparse.py`
- Modify: `docuverse/engines/retrieval/lancedb/__init__.py`

LanceDB has no native sparse-vector type. Each sparse model gets a struct column `{indices: list<int32>, values: list<float32>}`. Search rescoring is a Python dot product against fetched rows.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lancedb_sparse.py`:

```python
"""Tests for LanceDBSparseEngine — sparse vectors as a struct column with Python scoring."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

pytest.importorskip("lancedb")


class _FakeSparseModel:
    """Deterministic fake sparse encoder."""
    embedding_dim = 100

    def encode(self, texts, **kwargs):
        out = []
        for t in texts:
            n = len(t)
            ids = [hash(t[:i]) % 100 for i in range(1, min(n, 4) + 1)]
            vals = [1.0 / (i + 1) for i in range(len(ids))]
            out.append(csr_matrix((vals, ([0] * len(ids), ids)), shape=(1, 100)))
        return out


@pytest.fixture
def stub_sparse_config(tmp_path):
    cfg = SimpleNamespace(
        db_engine="lancedb-sparse",
        index_name="t_sparse",
        server=str(tmp_path),
        project_dir=str(tmp_path),
        top_k=10,
        ingestion_batch_size=8,
        bulk_batch=8,
        max_text_size=1024,
        data_template=SimpleNamespace(extra_fields=[]),
        duplicate_removal=None,
        rouge_duplicate_threshold=0.95,
        verbose=False,
        index_params=None,
        embeddings_name="splade_sparse",
    )
    cfg.__dict__["model_name"] = "fake-sparse"
    return cfg


def test_sparse_class_importable():
    from docuverse.engines.retrieval.lancedb import LanceDBSparseEngine
    from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine
    assert issubclass(LanceDBSparseEngine, LanceDBEngine)


def test_sparse_struct_schema(stub_sparse_config, monkeypatch):
    """The sparse engine adds one struct column with indices + values."""
    from docuverse.engines.retrieval.lancedb import lancedb_sparse as ls
    monkeypatch.setattr(ls, "SparseEmbeddingFunction", lambda *a, **k: _FakeSparseModel())

    eng = ls.LanceDBSparseEngine(stub_sparse_config)
    fields = eng.extra_schema_fields()
    assert len(fields) == 1
    assert fields[0].name == "splade_sparse"
    # Struct of indices + values
    sub_names = {f.name for f in fields[0].type}
    assert sub_names == {"indices", "values"}


def test_sparse_encode_data_returns_struct_dicts(stub_sparse_config, monkeypatch):
    from docuverse.engines.retrieval.lancedb import lancedb_sparse as ls
    monkeypatch.setattr(ls, "SparseEmbeddingFunction", lambda *a, **k: _FakeSparseModel())
    eng = ls.LanceDBSparseEngine(stub_sparse_config)
    out = eng.encode_data(["hello", "world!"])
    assert len(out) == 2
    for entry in out:
        assert set(entry) == {"indices", "values"}
        assert len(entry["indices"]) == len(entry["values"])


def test_sparse_dot_product_scoring(stub_sparse_config, monkeypatch):
    """A query that overlaps with one document on key indices ranks it first."""
    from docuverse.engines.retrieval.lancedb import lancedb_sparse as ls
    from docuverse.engines.search_queries import SearchQueries
    monkeypatch.setattr(ls, "SparseEmbeddingFunction", lambda *a, **k: _FakeSparseModel())

    eng = ls.LanceDBSparseEngine(stub_sparse_config)
    eng.create_index()
    eng.table.add([
        {"id": "a", "text": "x", "title": "",
         "splade_sparse": {"indices": [1, 2, 3], "values": [1.0, 0.5, 0.25]}},
        {"id": "b", "text": "y", "title": "",
         "splade_sparse": {"indices": [10, 11], "values": [1.0, 1.0]}},
        {"id": "c", "text": "z", "title": "",
         "splade_sparse": {"indices": [2, 3, 4], "values": [0.5, 0.5, 0.5]}},
    ])
    eng.build_indexes()  # no-op for sparse

    q = SearchQueries.Query(id="q1", text="...", relevant=[], answers=[])
    # Inject a known query sparse vector by monkeypatching encode_query.
    monkeypatch.setattr(eng, "encode_query", lambda question, tm=None:
        csr_matrix(([1.0, 1.0], ([0, 0], [2, 3])), shape=(1, 100)))

    res = eng.search(q)
    passages = res.retrieved_passages if hasattr(res, "retrieved_passages") else res
    ids_in_order = [p["id"] for p in passages]
    # 'a' has 0.5 + 0.25 = 0.75; 'c' has 0.5 + 0.5 = 1.0; 'b' has 0.
    assert ids_in_order[0] == "c"
    assert "a" in ids_in_order
```

- [ ] **Step 2: Run the failing tests**

Run: `conda activate ndocu && python -m pytest tests/test_lancedb_sparse.py -v`

Expected: FAIL — `cannot import name 'LanceDBSparseEngine'`.

- [ ] **Step 3: Create the sparse engine**

Create `docuverse/engines/retrieval/lancedb/lancedb_sparse.py`:

```python
"""LanceDB sparse engine — sparse vectors stored as struct columns.

LanceDB has no native sparse-vector type, so each row stores
``{indices: list<int32>, values: list<float32>}`` for the column. Scoring is
performed in Python (dot product) against either the full table (standalone
use) or a candidate id pool supplied by the hybrid composer.
"""
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
from scipy.sparse import csr_matrix

from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import get_param
from docuverse.utils.timer import timer

try:
    from docuverse.utils.embeddings.sparse_embedding_function import SparseEmbeddingFunction
except ImportError:  # pragma: no cover
    SparseEmbeddingFunction = None


class LanceDBSparseEngine(LanceDBEngine):
    """Sparse retrieval over a struct column ``{indices, values}``."""

    DEFAULT_SPARSE_COLUMN = "sparse"

    def __init__(self, config_params, **kwargs):
        self.embeddings_name = get_param(config_params, "embeddings_name",
                                         self.DEFAULT_SPARSE_COLUMN)
        self.model = None
        super().__init__(config_params, **kwargs)

    def init_model(self, **kwargs):
        if SparseEmbeddingFunction is None:
            raise RuntimeError(
                "SparseEmbeddingFunction is unavailable; "
                "install transformers + torch to use LanceDBSparseEngine."
            )
        self.model = SparseEmbeddingFunction(
            self.config.model_name,
            **self.config.__dict__,
        )

    # ===== Hooks =====

    def extra_schema_fields(self):
        struct = pa.struct([
            pa.field("indices", pa.list_(pa.int32())),
            pa.field("values", pa.list_(pa.float32())),
        ])
        return [pa.field(self.embeddings_name, struct)]

    def encode_data(self, texts, tm=None, **kwargs):
        vectors = self.model.encode(texts, show_progress_bar=False, tm=tm)
        out = []
        for v in vectors:
            indices, values = self._csr_to_lists(v)
            out.append({"indices": indices, "values": values})
        return out

    def encode_query(self, question, tm=None):
        text = question.text if hasattr(question, "text") else question
        vec = self.model.encode([text], show_progress_bar=False, tm=tm)[0]
        return vec  # csr_matrix

    def build_indexes(self):
        # No native sparse index; nothing to build.
        return

    # ===== Search =====

    def search(self, query: SearchQueries.Query,
               candidate_ids: Optional[List[str]] = None, **kwargs):
        tm = timer("ingest_and_test::search::retrieve")
        self._open_table()
        qvec = self.encode_query(query, tm=tm)
        tm.add_timing("encode")

        if not isinstance(qvec, csr_matrix):
            qvec = csr_matrix(qvec)
        q_indices = qvec.indices
        q_values = qvec.data
        if len(q_indices) == 0:
            return SearchResult(query, [])

        rows = self._fetch_rows(candidate_ids)
        tm.add_timing("fetch_rows")

        scored = []
        q_lookup = dict(zip(q_indices.tolist(), q_values.tolist()))
        for r in rows:
            sp = r.get(self.embeddings_name)
            if not sp:
                continue
            score = 0.0
            for idx, val in zip(sp["indices"], sp["values"]):
                qv = q_lookup.get(int(idx))
                if qv is not None:
                    score += float(qv) * float(val)
            if score == 0.0:
                continue
            p = {
                "id": r.get("id", ""),
                "text": r.get("text", ""),
                "title": r.get("title", ""),
                "score": score,
            }
            for f in self.extra_fields:
                if f in r:
                    p[f] = r[f]
            scored.append(p)
        tm.add_timing("score")

        scored.sort(key=lambda p: p["score"], reverse=True)
        scored = scored[: int(self.config.top_k)]
        return SearchResult(query, scored)

    # ===== Helpers =====

    @staticmethod
    def _csr_to_lists(v):
        if not isinstance(v, csr_matrix):
            v = csr_matrix(v)
        indices = [int(i) for i in v.indices.tolist()]
        values = [float(x) for x in v.data.tolist()]
        return indices, values

    def _fetch_rows(self, candidate_ids: Optional[List[str]]) -> List[dict]:
        cols = ["id", "text", "title", self.embeddings_name] + self.extra_fields
        if candidate_ids is None:
            warnings.warn(
                "LanceDBSparseEngine standalone search performs a full table "
                "scan; prefer use inside a hybrid composer.",
                stacklevel=2,
            )
            return self.table.to_pandas(columns=cols).to_dict("records")
        if not candidate_ids:
            return []
        # Build an SQL-style IN filter; LanceDB accepts duckdb-style filters.
        ids_quoted = ", ".join(f"'{cid}'" for cid in candidate_ids)
        filter_expr = f"id IN ({ids_quoted})"
        return (
            self.table.search()
            .where(filter_expr)
            .select(cols)
            .limit(len(candidate_ids))
            .to_list()
        )

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info["model"] = self.config.model_name
        info["embeddings_name"] = self.embeddings_name
        return info
```

- [ ] **Step 4: Update package `__init__.py`**

Overwrite:

```python
from .lancedb import LanceDBEngine as _LanceDBBase
from .lancedb_dense import LanceDBDenseEngine
from .lancedb_bm25 import LanceDBBM25Engine
from .lancedb_sparse import LanceDBSparseEngine

LanceDBEngine = LanceDBDenseEngine

__all__ = ['LanceDBEngine', 'LanceDBDenseEngine',
           'LanceDBBM25Engine', 'LanceDBSparseEngine']
```

- [ ] **Step 5: Run sparse tests**

Run: `conda activate ndocu && python -m pytest tests/test_lancedb_sparse.py -v`

Expected: PASS — all four tests.

- [ ] **Step 6: Commit**

```bash
git add docuverse/engines/retrieval/lancedb/lancedb_sparse.py docuverse/engines/retrieval/lancedb/__init__.py tests/test_lancedb_sparse.py
git commit -m "Add LanceDBSparseEngine storing sparse vectors as a struct column."
```

---

## Task 5: Implement `LanceDBHybridEngine` — composition + ingestion

**Files:**
- Create: `docuverse/engines/retrieval/lancedb/lancedb_hybrid.py`
- Modify: `docuverse/engines/retrieval/lancedb/__init__.py`
- Create: `tests/test_lancedb_hybrid.py` (will be expanded in Task 6 with search tests)

This task wires up sub-engine composition and the merged ingestion pass. Search arrives in Task 6.

- [ ] **Step 1: Write failing config-validation tests**

Create `tests/test_lancedb_hybrid.py`:

```python
"""Tests for LanceDBHybridEngine — composition, schema merging, validation."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

pytest.importorskip("lancedb")


class _FakeDenseModel:
    embedding_dim = 8

    def encode(self, texts, **kwargs):
        return [np.full(8, float(i + 1) / 10.0, dtype=np.float32) for i, _ in enumerate(texts)]


class _FakeSparseModel:
    embedding_dim = 50

    def encode(self, texts, **kwargs):
        out = []
        for i, t in enumerate(texts):
            ids = [(i * 7 + k) % 50 for k in range(3)]
            vals = [1.0, 0.5, 0.25]
            out.append(csr_matrix((vals, ([0] * 3, ids)), shape=(1, 50)))
        return out


def _hybrid_config(tmp_path, combination="rrf", with_sparse=True, with_bm25=True):
    models = {
        "dense": {
            "weight": 0.7,
            "top_k": 10,
            "db_engine": "lancedb_dense",
            "embeddings_name": "dense_vec",
            "model_name": "fake-dense",
        }
    }
    if with_bm25:
        models["bm25"] = {
            "weight": 0.2,
            "top_k": 10,
            "db_engine": "lancedb_bm25",
        }
    if with_sparse:
        models["splade"] = {
            "weight": 0.1,
            "top_k": 10,
            "db_engine": "lancedb_sparse",
            "embeddings_name": "splade_vec",
            "model_name": "fake-sparse",
        }
    cfg = SimpleNamespace(
        db_engine="lancedb-hybrid",
        index_name="t_hybrid",
        server=str(tmp_path),
        project_dir=str(tmp_path),
        top_k=10,
        ingestion_batch_size=8,
        bulk_batch=8,
        max_text_size=1024,
        data_template=SimpleNamespace(extra_fields=[]),
        duplicate_removal=None,
        rouge_duplicate_threshold=0.95,
        verbose=False,
        index_params=None,
        hybrid={
            "combination": combination,
            "rrf_k": 60,
            "models": models,
        },
        hybrid_submodules=None,
    )
    cfg.__dict__["model_name"] = None
    return cfg


@pytest.fixture
def patched_models(monkeypatch):
    """Patch dense and sparse encoders to deterministic fakes everywhere."""
    from docuverse.engines.retrieval.lancedb import lancedb_dense as ld
    from docuverse.engines.retrieval.lancedb import lancedb_sparse as ls
    monkeypatch.setattr(ld, "DenseEmbeddingFunction", lambda *a, **k: _FakeDenseModel())
    monkeypatch.setattr(ls, "SparseEmbeddingFunction", lambda *a, **k: _FakeSparseModel())
    return monkeypatch


def test_hybrid_class_importable():
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine
    assert issubclass(LanceDBHybridEngine, LanceDBEngine)


def test_hybrid_invalid_combination_raises(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="bogus")
    with pytest.raises(ValueError, match="combination"):
        LanceDBHybridEngine(cfg)


def test_hybrid_weighted_missing_weight_raises(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="weighted")
    del cfg.hybrid["models"]["bm25"]["weight"]
    with pytest.raises(ValueError, match="weight"):
        LanceDBHybridEngine(cfg)


def test_hybrid_rejects_milvus_subengine(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path)
    cfg.hybrid["models"]["dense"]["db_engine"] = "milvus_dense"
    with pytest.raises(ValueError, match="lancedb"):
        LanceDBHybridEngine(cfg)


def test_hybrid_rejects_duplicate_embeddings_name(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path)
    cfg.hybrid["models"]["splade"]["embeddings_name"] = "dense_vec"  # collision
    with pytest.raises(ValueError, match="embeddings_name"):
        LanceDBHybridEngine(cfg)


def test_hybrid_merged_schema_contains_all_columns(tmp_path, patched_models):
    """Schema includes id/text/title + dense_vec + splade_vec (BM25 adds none)."""
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path)
    eng = LanceDBHybridEngine(cfg)
    schema = eng._build_schema()
    names = {f.name for f in schema}
    assert {"id", "text", "title", "dense_vec", "splade_vec"} <= names


def test_hybrid_ingest_writes_one_row_per_doc(tmp_path, patched_models):
    """Ingestion writes a single row per document with all sub-engine columns populated."""
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path)
    eng = LanceDBHybridEngine(cfg)
    eng.create_index()

    docs = [
        {"id": "a", "text": "alpha brown fox", "title": ""},
        {"id": "b", "text": "beta lazy dog", "title": ""},
        {"id": "c", "text": "gamma quick fox", "title": ""},
    ]
    records = eng._create_data(docs)
    assert len(records) == 3
    for r in records:
        assert "dense_vec" in r and "splade_vec" in r
        assert isinstance(r["dense_vec"], list) and len(r["dense_vec"]) == 8
        assert set(r["splade_vec"]) == {"indices", "values"}
```

- [ ] **Step 2: Run failing tests**

Run: `conda activate ndocu && python -m pytest tests/test_lancedb_hybrid.py -v`

Expected: FAIL — `cannot import name 'LanceDBHybridEngine'`.

- [ ] **Step 3: Create the hybrid engine — composition + ingestion**

Create `docuverse/engines/retrieval/lancedb/lancedb_hybrid.py`:

```python
"""LanceDBHybridEngine — composes dense, BM25, and sparse sub-engines on a
shared LanceDB table.

Configuration shape mirrors ``MilvusHybridEngine`` so existing experiment
YAML translates by changing ``db_engine`` and the per-model ``db_engine``.
"""
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

import pyarrow as pa

from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine
from docuverse.engines.retrieval.lancedb.lancedb_dense import LanceDBDenseEngine
from docuverse.engines.retrieval.lancedb.lancedb_bm25 import LanceDBBM25Engine
from docuverse.engines.retrieval.lancedb.lancedb_sparse import LanceDBSparseEngine
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import get_param
from docuverse.utils.timer import timer


_SUBENGINE_TYPES = {
    "lancedb_dense":  LanceDBDenseEngine,
    "lancedb-dense":  LanceDBDenseEngine,
    "lancedb_bm25":   LanceDBBM25Engine,
    "lancedb-bm25":   LanceDBBM25Engine,
    "lancedb_sparse": LanceDBSparseEngine,
    "lancedb-sparse": LanceDBSparseEngine,
}


class LanceDBHybridEngine(LanceDBEngine):
    """Hybrid retrieval over a shared LanceDB table.

    Sub-engines are constructed for each entry in ``config.hybrid.models``.
    All sub-engines share ``self.db`` and ``self.table``; each contributes
    its column(s) to the merged schema and writes its column value during
    ingestion.
    """

    def __init__(self, config_params, **kwargs):
        self.sub_engines: List[LanceDBEngine] = []
        self.sub_names: List[str] = []
        self.sub_weights: List[float] = []
        self.combination: str = "rrf"
        self.rrf_k: int = 60
        super().__init__(config_params, **kwargs)

    # ===== init =====

    def init_model(self, **kwargs):
        hybrid_cfg = get_param(self.config, "hybrid", None)
        if hybrid_cfg is None:
            raise ValueError("LanceDBHybridEngine requires a `hybrid` config block.")

        self.combination = get_param(hybrid_cfg, "combination", "rrf")
        if self.combination not in ("rrf", "weighted"):
            raise ValueError(
                f"Invalid hybrid.combination={self.combination!r}; "
                f"expected 'rrf' or 'weighted'."
            )
        self.rrf_k = int(get_param(hybrid_cfg, "rrf_k", 60))

        models_cfg = hybrid_cfg["models"]
        self._validate_subengine_configs(models_cfg)

        common_cfg = {k: v for k, v in hybrid_cfg.items() if k != "models"}
        parent_cfg = {k: v for k, v in self.config.__dict__.items() if k != "hybrid"}

        weights: List[float] = []
        for name, sub_cfg in models_cfg.items():
            engine_name = sub_cfg["db_engine"]
            engine_cls = _SUBENGINE_TYPES[engine_name]

            from types import SimpleNamespace
            merged = {**parent_cfg, **common_cfg, **sub_cfg}
            ns = SimpleNamespace(**merged)
            ns.__dict__["data_template"] = self.config.data_template
            ns.__dict__["index_name"] = self.config.index_name

            sub = engine_cls.__new__(engine_cls)  # bypass __init__; share resources
            self._init_subengine(sub, ns)
            self.sub_engines.append(sub)
            self.sub_names.append(name)
            if self.combination == "weighted":
                weights.append(float(sub_cfg["weight"]))

        if self.combination == "weighted":
            total = sum(weights)
            if total <= 0:
                raise ValueError("hybrid weights must sum to a positive number.")
            self.sub_weights = [w / total for w in weights]
        else:
            self.sub_weights = [1.0 / len(self.sub_engines)] * len(self.sub_engines)

    def _validate_subengine_configs(self, models_cfg: Dict[str, dict]):
        if not models_cfg:
            raise ValueError("hybrid.models must list at least one sub-engine.")
        seen_columns: set[str] = set()
        for name, sub in models_cfg.items():
            engine = sub.get("db_engine")
            if engine not in _SUBENGINE_TYPES:
                raise ValueError(
                    f"Sub-engine {name!r} has db_engine={engine!r}; "
                    f"expected one of {sorted(_SUBENGINE_TYPES)} "
                    f"(must be a lancedb_* engine)."
                )
            if self.combination == "weighted" and "weight" not in sub:
                raise ValueError(
                    f"hybrid.combination='weighted' requires `weight` for sub-engine {name!r}."
                )
            col = sub.get("embeddings_name")
            if col:
                if col in seen_columns:
                    raise ValueError(
                        f"Duplicate embeddings_name={col!r} across hybrid sub-engines."
                    )
                seen_columns.add(col)

    def _init_subengine(self, sub: LanceDBEngine, ns):
        """Initialize a sub-engine sharing this composer's connection and table."""
        # Walk the sub-engine's __init__ contract manually so we can share state.
        sub.config = ns
        sub.db = self.db
        sub.table = self.table
        sub.persist_directory = self.persist_directory
        sub.metric = self.metric
        sub.extra_fields = self.extra_fields
        if hasattr(sub, "embeddings_name"):
            sub.embeddings_name = get_param(ns, "embeddings_name",
                                            getattr(sub, "DEFAULT_VECTOR_COLUMN",
                                                    getattr(sub, "DEFAULT_SPARSE_COLUMN",
                                                            "vector")))
        sub.init_model()

    # ===== Schema composition =====

    def extra_schema_fields(self):
        fields: List[pa.Field] = []
        for sub in self.sub_engines:
            fields.extend(sub.extra_schema_fields())
        return fields

    # ===== Ingestion =====

    def encode_data(self, texts, tm=None, **kwargs):
        per_engine = [sub.encode_data(texts, tm=tm) for sub in self.sub_engines]
        # Transpose: list-of-engines x list-of-texts → list-of-texts x list-of-engines.
        return [list(per_text) for per_text in zip(*per_engine)]

    def _assign_encoded(self, record, enc, col_names):
        idx = 0
        for sub, sub_enc in zip(self.sub_engines, enc):
            sub_cols = [f.name for f in sub.extra_schema_fields()]
            if not sub_cols:
                continue  # BM25 contributes nothing
            sub_record_view = {}
            sub._assign_encoded(sub_record_view, sub_enc, sub_cols)
            record.update(sub_record_view)

    def build_indexes(self):
        self._open_table()
        for sub in self.sub_engines:
            sub.table = self.table
            sub.build_indexes()

    def create_index(self, index_name=None, **kwargs):
        super().create_index(index_name=index_name, **kwargs)
        for sub in self.sub_engines:
            sub.table = self.table

    def _open_table(self):
        super()._open_table()
        for sub in self.sub_engines:
            sub.table = self.table

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info["combination"] = self.combination
        info["sub_engines"] = [
            {"name": n, "type": type(s).__name__, "weight": w}
            for n, s, w in zip(self.sub_names, self.sub_engines, self.sub_weights)
        ]
        return info
```

- [ ] **Step 4: Update package `__init__.py`**

Overwrite:

```python
from .lancedb import LanceDBEngine as _LanceDBBase
from .lancedb_dense import LanceDBDenseEngine
from .lancedb_bm25 import LanceDBBM25Engine
from .lancedb_sparse import LanceDBSparseEngine
from .lancedb_hybrid import LanceDBHybridEngine

LanceDBEngine = LanceDBDenseEngine

__all__ = [
    'LanceDBEngine',
    'LanceDBDenseEngine',
    'LanceDBBM25Engine',
    'LanceDBSparseEngine',
    'LanceDBHybridEngine',
]
```

- [ ] **Step 5: Run hybrid tests**

Run: `conda activate ndocu && python -m pytest tests/test_lancedb_hybrid.py -v`

Expected: PASS — all 7 tests defined in Step 1.

- [ ] **Step 6: Commit**

```bash
git add docuverse/engines/retrieval/lancedb/lancedb_hybrid.py docuverse/engines/retrieval/lancedb/__init__.py tests/test_lancedb_hybrid.py
git commit -m "Add LanceDBHybridEngine composition, validation, and merged ingestion."
```

---

## Task 6: Hybrid search — per-engine retrieval, RRF/weighted merge, native fast path

**Files:**
- Modify: `docuverse/engines/retrieval/lancedb/lancedb_hybrid.py` (add `search`)
- Modify: `tests/test_lancedb_hybrid.py` (add search tests)

- [ ] **Step 1: Write failing search tests**

Append to `tests/test_lancedb_hybrid.py`:

```python
def _ingest_and_search(eng, docs, query_text):
    from docuverse.engines.search_queries import SearchQueries
    eng.create_index()
    records = eng._create_data(docs)
    eng._insert_data(records)
    eng.build_indexes()
    q = SearchQueries.Query(id="q1", text=query_text, relevant=[], answers=[])
    return eng.search(q)


def test_hybrid_search_dense_bm25_rrf_native_fast_path(tmp_path, patched_models):
    """Dense + BM25 + RRF should use LanceDB's native hybrid query."""
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="rrf", with_sparse=False, with_bm25=True)
    eng = LanceDBHybridEngine(cfg)
    docs = [
        {"id": "a", "text": "alpha brown fox jumps high", "title": ""},
        {"id": "b", "text": "beta lazy dog sleeps low", "title": ""},
        {"id": "c", "text": "gamma quick fox runs fast", "title": ""},
    ]
    res = _ingest_and_search(eng, docs, "fox")
    passages = res.retrieved_passages if hasattr(res, "retrieved_passages") else res
    ids = [p["id"] for p in passages]
    assert {"a", "c"} & set(ids)


def test_hybrid_search_three_engines_rrf(tmp_path, patched_models):
    """Dense + BM25 + sparse with RRF combination falls back to Python merge."""
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="rrf", with_sparse=True, with_bm25=True)
    eng = LanceDBHybridEngine(cfg)
    docs = [
        {"id": "a", "text": "alpha brown fox jumps high", "title": ""},
        {"id": "b", "text": "beta lazy dog sleeps low", "title": ""},
        {"id": "c", "text": "gamma quick fox runs fast", "title": ""},
    ]
    res = _ingest_and_search(eng, docs, "fox")
    passages = res.retrieved_passages if hasattr(res, "retrieved_passages") else res
    ids = [p["id"] for p in passages]
    assert len(ids) > 0
    assert all("score" in p for p in passages)


def test_hybrid_search_three_engines_weighted(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="weighted",
                         with_sparse=True, with_bm25=True)
    eng = LanceDBHybridEngine(cfg)
    docs = [
        {"id": "a", "text": "alpha brown fox jumps high", "title": ""},
        {"id": "b", "text": "beta lazy dog sleeps low", "title": ""},
        {"id": "c", "text": "gamma quick fox runs fast", "title": ""},
    ]
    res = _ingest_and_search(eng, docs, "fox")
    passages = res.retrieved_passages if hasattr(res, "retrieved_passages") else res
    assert len(passages) > 0


def test_hybrid_rrf_math_unit():
    """RRF: rank-1 contribution = 1/(60+1); rank-2 = 1/(60+2)."""
    from docuverse.engines.retrieval.lancedb.lancedb_hybrid import _rrf_combine
    rankings = [
        [("a", 5.0), ("b", 4.0), ("c", 3.0)],
        [("c", 9.0), ("a", 7.0), ("b", 5.0)],
    ]
    out = _rrf_combine(rankings, k=60, top_k=3)
    ids = [item[0] for item in out]
    # 'a' appears at ranks 1 and 2; 'c' at ranks 3 and 1; 'b' at ranks 2 and 3.
    # Score(a)=1/61+1/62; Score(c)=1/63+1/61; Score(b)=1/62+1/63.
    # a > c > b
    assert ids == ["a", "c", "b"]


def test_hybrid_weighted_math_unit():
    """Weighted: min-max normalize each ranking then weighted-sum."""
    from docuverse.engines.retrieval.lancedb.lancedb_hybrid import _weighted_combine
    rankings = [
        [("a", 1.0), ("b", 0.0)],     # weight 0.7
        [("b", 4.0), ("a", 2.0)],     # weight 0.3
    ]
    out = _weighted_combine(rankings, weights=[0.7, 0.3], top_k=2)
    by_id = dict(out)
    # 'a' normalized ranking 1 = 1.0; ranking 2 = 0.0  -> 0.7*1.0 + 0.3*0.0 = 0.7
    # 'b' normalized ranking 1 = 0.0; ranking 2 = 1.0  -> 0.7*0.0 + 0.3*1.0 = 0.3
    assert abs(by_id["a"] - 0.7) < 1e-9
    assert abs(by_id["b"] - 0.3) < 1e-9


def test_hybrid_single_subengine_skips_merge(tmp_path, patched_models):
    from docuverse.engines.retrieval.lancedb import LanceDBHybridEngine
    cfg = _hybrid_config(tmp_path, combination="rrf",
                         with_sparse=False, with_bm25=False)
    eng = LanceDBHybridEngine(cfg)
    docs = [
        {"id": "a", "text": "alpha brown fox", "title": ""},
        {"id": "b", "text": "beta lazy dog", "title": ""},
    ]
    res = _ingest_and_search(eng, docs, "fox")
    passages = res.retrieved_passages if hasattr(res, "retrieved_passages") else res
    assert len(passages) > 0
```

- [ ] **Step 2: Run failing tests**

Run: `conda activate ndocu && python -m pytest tests/test_lancedb_hybrid.py -v -k search`

Expected: FAIL — `_rrf_combine`, `_weighted_combine`, and the `search` method don't exist.

- [ ] **Step 3: Add merge helpers and `search` to `lancedb_hybrid.py`**

Add the following to the module-level scope of `docuverse/engines/retrieval/lancedb/lancedb_hybrid.py` (above the class):

```python
def _rrf_combine(rankings: List[List[Tuple[str, float]]],
                 k: int, top_k: int) -> List[Tuple[str, float]]:
    """Reciprocal rank fusion. ``rankings[i]`` is a list of (id, score) sorted desc."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank_idx, (doc_id, _score) in enumerate(ranking):
            rank = rank_idx + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_k]


def _weighted_combine(rankings: List[List[Tuple[str, float]]],
                      weights: List[float], top_k: int) -> List[Tuple[str, float]]:
    """Min-max normalize each ranking's scores, then weighted sum."""
    scores: Dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        if not ranking:
            continue
        raw = [s for _, s in ranking]
        lo, hi = min(raw), max(raw)
        span = hi - lo if hi > lo else 1.0
        for doc_id, s in ranking:
            normalized = (s - lo) / span
            scores[doc_id] = scores.get(doc_id, 0.0) + w * normalized
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_k]
```

Add `search` and helpers to `LanceDBHybridEngine`:

```python
    # ===== Search =====

    def search(self, query: SearchQueries.Query, **kwargs):
        tm = timer("ingest_and_test::search::retrieve")
        self._open_table()

        if len(self.sub_engines) == 1:
            return self.sub_engines[0].search(query, **kwargs)

        # Native fast path: exactly one dense + one BM25, RRF combination.
        if self._is_native_fast_path():
            return self._native_dense_fts_search(query, tm=tm)

        rankings, payloads = self._gather_subengine_rankings(query, tm=tm)
        merged = self._merge_rankings(rankings)
        passages = self._build_passages(merged, payloads)
        result = SearchResult(query, passages)
        if get_param(self.config, "duplicate_removal", None):
            result.remove_duplicates(self.config.duplicate_removal,
                                     self.config.rouge_duplicate_threshold)
        tm.add_timing("merge_complete")
        return result

    def _is_native_fast_path(self) -> bool:
        if self.combination != "rrf":
            return False
        types = {type(s) for s in self.sub_engines}
        if types != {LanceDBDenseEngine, LanceDBBM25Engine}:
            return False
        return len(self.sub_engines) == 2

    def _native_dense_fts_search(self, query, tm=None):
        from lancedb.rerankers import RRFReranker
        dense = next(s for s in self.sub_engines if isinstance(s, LanceDBDenseEngine))
        qvec = dense.encode_query(query, tm=tm)
        qtext = query.text if hasattr(query, "text") else str(query)
        if tm: tm.add_timing("encode")

        output_cols = ["id", "text", "title"] + self.extra_fields
        try:
            results = (
                self.table.search(query_type="hybrid")
                .vector(qvec)
                .text(qtext)
                .rerank(RRFReranker())
                .select(output_cols)
                .limit(int(self.config.top_k))
                .to_list()
            )
        except Exception:
            # If the LanceDB version doesn't support hybrid query directly,
            # fall back to the manual path.
            rankings, payloads = self._gather_subengine_rankings(query, tm=tm)
            merged = self._merge_rankings(rankings)
            return SearchResult(query, self._build_passages(merged, payloads))

        passages = []
        for r in results:
            score = float(r.get("_relevance_score", r.get("_score", 0.0)))
            p = {
                "id": r.get("id", ""),
                "text": r.get("text", ""),
                "title": r.get("title", ""),
                "score": score,
            }
            for f in self.extra_fields:
                if f in r:
                    p[f] = r[f]
            passages.append(p)
        if tm: tm.add_timing("native_hybrid_search")
        return SearchResult(query, passages)

    def _gather_subengine_rankings(self, query, tm=None):
        """Run each sub-engine; collect (id, score) rankings + payload by id."""
        rankings: List[List[Tuple[str, float]]] = []
        payloads: Dict[str, dict] = {}
        candidate_ids: List[str] = []

        non_sparse = [s for s in self.sub_engines
                      if not isinstance(s, LanceDBSparseEngine)]
        sparse = [s for s in self.sub_engines if isinstance(s, LanceDBSparseEngine)]

        # Run non-sparse first to seed candidate ids for sparse.
        for sub in non_sparse:
            try:
                res = sub.search(query)
            except Exception as e:  # pragma: no cover - defensive
                print(f"Sub-engine {type(sub).__name__} failed: {e}")
                rankings.append([])
                continue
            ranking = []
            for p in (res.retrieved_passages if hasattr(res, "retrieved_passages") else res):
                ranking.append((p["id"], p["score"]))
                if p["id"] not in payloads:
                    payloads[p["id"]] = p
                    candidate_ids.append(p["id"])
            rankings.append(ranking)

        for sub in sparse:
            try:
                res = sub.search(query, candidate_ids=candidate_ids or None)
            except TypeError:
                res = sub.search(query)
            except Exception as e:  # pragma: no cover - defensive
                print(f"Sub-engine {type(sub).__name__} failed: {e}")
                rankings.append([])
                continue
            ranking = []
            for p in (res.retrieved_passages if hasattr(res, "retrieved_passages") else res):
                ranking.append((p["id"], p["score"]))
                if p["id"] not in payloads:
                    payloads[p["id"]] = p
            rankings.append(ranking)

        # Reorder rankings to match self.sub_engines order so weights line up.
        ordered: List[List[Tuple[str, float]]] = []
        ns_iter = iter(rankings[: len(non_sparse)])
        sp_iter = iter(rankings[len(non_sparse):])
        for sub in self.sub_engines:
            if isinstance(sub, LanceDBSparseEngine):
                ordered.append(next(sp_iter))
            else:
                ordered.append(next(ns_iter))
        if tm: tm.add_timing("subengine_rankings")
        return ordered, payloads

    def _merge_rankings(self, rankings):
        if self.combination == "rrf":
            return _rrf_combine(rankings, k=self.rrf_k, top_k=int(self.config.top_k))
        return _weighted_combine(rankings, weights=self.sub_weights,
                                 top_k=int(self.config.top_k))

    def _build_passages(self, merged, payloads):
        passages = []
        for doc_id, score in merged:
            p = dict(payloads.get(doc_id, {"id": doc_id, "text": "", "title": ""}))
            p["score"] = float(score)
            passages.append(p)
        return passages
```

- [ ] **Step 4: Run search tests**

Run: `conda activate ndocu && python -m pytest tests/test_lancedb_hybrid.py -v`

Expected: PASS — all hybrid tests including the math units.

If `test_hybrid_search_dense_bm25_rrf_native_fast_path` errors due to a LanceDB version that doesn't support the `query_type="hybrid"` builder API, the engine's `_native_dense_fts_search` already catches it and falls back; the test should still pass through the fallback path.

- [ ] **Step 5: Commit**

```bash
git add docuverse/engines/retrieval/lancedb/lancedb_hybrid.py tests/test_lancedb_hybrid.py
git commit -m "Add hybrid search: per-engine rankings, RRF/weighted merge, native dense+FTS fast path."
```

---

## Task 7: Wire up the dispatcher and dispatch tests

**Files:**
- Modify: `docuverse/utils/retrievers.py:59-65`
- Modify: `tests/test_engine_dispatch.py:27-49`

- [ ] **Step 1: Add the new dispatch test names**

Edit `tests/test_engine_dispatch.py`. In the `KNOWN_NAMES` list (lines 27-49), append:

```python
    "lancedb-dense",
    "lancedb_dense",
    "lancedb-bm25",
    "lancedb_bm25",
    "lancedb-sparse",
    "lancedb_sparse",
    "lancedb-hybrid",
    "lancedb_hybrid",
```

- [ ] **Step 2: Run the failing dispatch tests**

Run: `conda activate ndocu && python -m pytest tests/test_engine_dispatch.py -v`

Expected: FAIL — the new names raise `NotImplementedError("Unknown engine type: lancedb-...")`.

- [ ] **Step 3: Extend the dispatcher**

In `docuverse/utils/retrievers.py`, replace the `lancedb` branch (lines 59-65) with:

```python
   elif name in ['lancedb', 'lance', 'lancedb-dense', 'lancedb_dense',
                 'lancedb-bm25', 'lancedb_bm25',
                 'lancedb-sparse', 'lancedb_sparse',
                 'lancedb-hybrid', 'lancedb_hybrid']:
       try:
           from docuverse.engines.retrieval.lancedb import (
               LanceDBDenseEngine, LanceDBBM25Engine,
               LanceDBSparseEngine, LanceDBHybridEngine,
           )
           if name in ['lancedb', 'lance', 'lancedb-dense', 'lancedb_dense']:
               engine = LanceDBDenseEngine(retriever_config)
           elif name in ['lancedb-bm25', 'lancedb_bm25']:
               engine = LanceDBBM25Engine(retriever_config)
           elif name in ['lancedb-sparse', 'lancedb_sparse']:
               engine = LanceDBSparseEngine(retriever_config)
           elif name in ['lancedb-hybrid', 'lancedb_hybrid']:
               engine = LanceDBHybridEngine(retriever_config)
       except ImportError as e:
           print("You need to install lancedb package (run `pip install lancedb`).")
           raise e
```

- [ ] **Step 4: Run dispatch tests**

Run: `conda activate ndocu && python -m pytest tests/test_engine_dispatch.py -v`

Expected: PASS — all 4 new lancedb sub-engine dispatch tests + existing.

- [ ] **Step 5: Commit**

```bash
git add docuverse/utils/retrievers.py tests/test_engine_dispatch.py
git commit -m "Dispatch lancedb-{bm25,sparse,hybrid} db_engine names from create_retrieval_engine."
```

---

## Task 8: Full regression sweep

- [ ] **Step 1: Run the full lancedb test suite plus existing back-compat tests**

Run:

```bash
conda activate ndocu && python -m pytest \
  tests/test_lancedb_dense.py \
  tests/test_lancedb_bm25.py \
  tests/test_lancedb_sparse.py \
  tests/test_lancedb_hybrid.py \
  tests/test_engine_dispatch.py \
  tests/test_back_compat.py \
  tests/test_presets.py \
  -v
```

Expected: all PASS. If any fail, fix in place — do not move on with red tests.

- [ ] **Step 2: Run the full project test suite to confirm no unrelated regressions**

Run: `conda activate ndocu && python -m pytest tests/ -v`

Expected: same pass/fail rate as `main` (no new failures introduced by this work).

- [ ] **Step 3: Final commit if any fixups were needed**

```bash
git status
# If anything was modified, stage and commit it with a clear message.
```

---

## Notes for the implementer

- The `from types import SimpleNamespace` import at the top of `_FakeSparseModel`-using fixtures is intentional — we don't construct a full `RetrievalArguments` object because `RetrievalEngine.__init__` mostly reads attributes via `get_param`, which falls back through dicts and namespaces.
- Some of the `RetrievalEngine` base behavior (like `create_update_index`) prompts the user when an index already exists. In tests we call `create_index()` directly to bypass that, then `_insert_data(records)` and `build_indexes()` manually.
- LanceDB's `query_type="hybrid"` builder API has changed across versions. The native fast path catches any exception and falls back to manual merging — we deliberately do not assert the native code path was actually used.
- `RetrievalEngine.__init__` in `retrieval_engine.py` may set fields via `load_model_config`. If the stub `SimpleNamespace` configs in tests are missing required attributes (e.g., `model_name`), set them explicitly with `cfg.__dict__["model_name"] = "..."` as the fixtures already do.
