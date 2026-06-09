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
        self.model = None
        super().__init__(config_params, **kwargs)
        # RetrievalEngine.__init__ sets embeddings_name via get_param, which
        # silently returns None for SimpleNamespace/object inputs (no .get()).
        # Read the attribute directly and fall back to our default.
        raw = (
            config_params.get("embeddings_name")
            if isinstance(config_params, dict)
            else getattr(config_params, "embeddings_name", None)
        )
        self.embeddings_name = raw if raw is not None else self.DEFAULT_SPARSE_COLUMN

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
            arrow_table = self.table.to_arrow()
            rows = arrow_table.to_pylist()
        else:
            if not candidate_ids:
                return []
            arrow_table = self.table.to_arrow()
            id_set = set(candidate_ids)
            rows = [r for r in arrow_table.to_pylist() if r.get("id") in id_set]
        present = [c for c in cols if not rows or c in rows[0]]
        return [{c: r.get(c) for c in present} for r in rows]

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info["model"] = self.config.model_name
        info["embeddings_name"] = self.embeddings_name
        return info
