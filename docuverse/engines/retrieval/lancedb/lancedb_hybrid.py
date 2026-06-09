"""LanceDBHybridEngine — composes dense, BM25, and sparse sub-engines on a
shared LanceDB table.

Configuration shape mirrors ``MilvusHybridEngine`` so existing experiment
YAML translates by changing ``db_engine`` and the per-model ``db_engine``.
"""
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

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
        # NOTE: self.config is a SimpleNamespace here; use getattr, not get_param.
        hybrid_cfg = getattr(self.config, "hybrid", None)
        if hybrid_cfg is None:
            raise ValueError("LanceDBHybridEngine requires a `hybrid` config block.")

        # hybrid_cfg is a plain dict from the config fixture, so get_param works.
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
        seen_columns: set = set()
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

    def _init_subengine(self, sub: LanceDBEngine, ns: SimpleNamespace):
        """Initialize a sub-engine sharing this composer's connection and table."""
        # Walk the sub-engine's __init__ contract manually so we can share state.
        sub.config = ns
        sub.db = self.db
        sub.table = self.table
        sub.persist_directory = self.persist_directory
        sub.metric = self.metric
        sub.extra_fields = self.extra_fields
        # Set embeddings_name BEFORE init_model so the sub-engine's schema hooks work.
        # NOTE: ns is a SimpleNamespace, so getattr is required (get_param returns None).
        if hasattr(sub, "embeddings_name") or isinstance(sub, (LanceDBDenseEngine, LanceDBSparseEngine)):
            default_col = getattr(sub.__class__, "DEFAULT_VECTOR_COLUMN",
                                  getattr(sub.__class__, "DEFAULT_SPARSE_COLUMN", "vector"))
            sub.embeddings_name = getattr(ns, "embeddings_name", None) or default_col
        sub.init_model()

    # ===== Schema composition =====

    def extra_schema_fields(self) -> List[pa.Field]:
        fields: List[pa.Field] = []
        for sub in self.sub_engines:
            fields.extend(sub.extra_schema_fields())
        return fields

    # ===== Ingestion =====

    def encode_data(self, texts, tm=None, **kwargs):
        per_engine = [sub.encode_data(texts, tm=tm) for sub in self.sub_engines]
        # Transpose: list-of-engines x list-of-texts -> list-of-texts x list-of-engines.
        return [list(per_text) for per_text in zip(*per_engine)]

    def _assign_encoded(self, record: dict, enc, col_names: List[str]):
        # enc is a list of per-sub-engine encoded values for one document.
        for sub, sub_enc in zip(self.sub_engines, enc):
            sub_cols = [f.name for f in sub.extra_schema_fields()]
            if not sub_cols:
                continue  # BM25 contributes nothing
            sub_record_view: dict = {}
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
        if getattr(self.config, "duplicate_removal", None):
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
        if tm:
            tm.add_timing("encode")

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
        if tm:
            tm.add_timing("native_hybrid_search")
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
        if tm:
            tm.add_timing("subengine_rankings")
        return ordered, payloads

    def _merge_rankings(self, rankings):
        if self.combination == "rrf":
            return _rrf_combine(rankings, k=self.rrf_k, top_k=int(self.config.top_k))
        return _weighted_combine(rankings, weights=self.sub_weights,
                                 top_k=int(self.config.top_k))

    def _build_passages(self, merged, payloads):
        passages = []
        for doc_id, score in merged:
            raw = payloads.get(doc_id, {"id": doc_id, "text": "", "title": ""})
            # payloads may contain SearchDatum objects (from sub-engine results)
            # or plain dicts; normalise to plain dict.
            if hasattr(raw, "__dict__"):
                p = dict(raw.__dict__)
            elif isinstance(raw, dict):
                p = dict(raw)
            else:
                p = {"id": doc_id, "text": "", "title": ""}
            p["score"] = float(score)
            passages.append(p)
        return passages

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info["combination"] = self.combination
        info["sub_engines"] = [
            {"name": n, "type": type(s).__name__, "weight": w}
            for n, s, w in zip(self.sub_names, self.sub_engines, self.sub_weights)
        ]
        return info
