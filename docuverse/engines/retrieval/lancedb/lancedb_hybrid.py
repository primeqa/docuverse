"""LanceDBHybridEngine — composes dense, BM25, and sparse sub-engines on a
shared LanceDB table.

Configuration shape mirrors ``MilvusHybridEngine`` so existing experiment
YAML translates by changing ``db_engine`` and the per-model ``db_engine``.
"""
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pyarrow as pa

from docuverse.engines.retrieval.lancedb.lancedb import LanceDBEngine
from docuverse.engines.retrieval.lancedb.lancedb_dense import LanceDBDenseEngine
from docuverse.engines.retrieval.lancedb.lancedb_bm25 import LanceDBBM25Engine
from docuverse.engines.retrieval.lancedb.lancedb_sparse import LanceDBSparseEngine
from docuverse.engines.search_corpus import SearchCorpus
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

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info["combination"] = self.combination
        info["sub_engines"] = [
            {"name": n, "type": type(s).__name__, "weight": w}
            for n, s, w in zip(self.sub_names, self.sub_engines, self.sub_weights)
        ]
        return info
