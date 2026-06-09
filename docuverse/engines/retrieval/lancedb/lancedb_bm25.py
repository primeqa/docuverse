"""LanceDB BM25 engine — full-text search via LanceDB's native FTS index.

Indexes the shared ``text`` column (no extra columns added). Search uses
LanceDB's ``query_type='fts'`` which returns BM25 scores.

Note: ``use_tantivy=True`` is omitted here; the ``tantivy`` Python package
is not a required dependency.  LanceDB's built-in FTS implementation is used
instead (``use_tantivy=False``, the default).
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
        # use_tantivy omitted (defaults to False) — tantivy package not required.
        self.table.create_fts_index("text", replace=True)

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
