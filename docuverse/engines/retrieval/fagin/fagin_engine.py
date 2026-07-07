"""FaginThresholdEngine — DocUVerse engine running Fagin's Threshold
Algorithm (Fagin, Lotem, Naor, PODS'01) over per-dimension sorted lists.

Mirrors SimdqEngine: in-process, file-backed (no client/server split). At
ingest time we accumulate corpus-encoded fp32 vectors across all batches,
then build (NumPy argsort + gather) and save the index in one shot. At
search time the index is mmap-loaded once on the first query and each
query runs the native TA kernel — exact top-k inner product by default;
fagin_epsilon / fagin_max_depth trade exactness for speed.

Persist layout:
    <persist_directory>/fagin_data/<index_name>/
        meta.json
        Y.npy
        order.bin
        vals.bin
        engine_metadata.json      # id_map + per-doc metadata sidecar
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from docuverse.engines.retrieval.fagin.fagin_index import FaginIndex
from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import _trim_json, get_param
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction
from docuverse.utils.timer import timer

_STAT_KEYS = ("depth", "sorted_accesses", "random_accesses", "rounds", "exhausted")


class FaginThresholdEngine(RetrievalEngine):
    """In-process exact top-k engine using Fagin's Threshold Algorithm.

    Reads fagin_* fields from RetrievalArguments. Index lives at
    ``<persist_directory>/fagin_data/<index_name>/``.
    """

    SUBDIR = "fagin_data"

    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)
        self.model: Optional[DenseEmbeddingFunction] = None
        self.hidden_dim: Optional[int] = None
        self.index: Optional[FaginIndex] = None
        self.id_map: List[str] = []
        self.metadata_store: Dict[str, dict] = {}
        # Optional precomputed query embeddings, keyed by id(query); see
        # precompute_query_embeddings().
        self._query_emb_cache: Dict[int, np.ndarray] = {}
        # Aggregated TA access counters across all queries (thread-safe;
        # search_all runs searches from a thread pool). Exposed via info().
        self._stats_lock = threading.Lock()
        self.ta_stats: Dict[str, int] = {"queries": 0,
                                         **{k: 0 for k in _STAT_KEYS}}

        self.load_model_config(config_params)
        self.text_header = "text"
        self.title_header = "title"
        self.id_header = "id"
        self.extra_fields = get_param(self.config.data_template, "extra_fields", [])
        self.persist_directory = get_param(self.config, "project_dir", "/tmp")

        # Defer the (GPU) model load until first use — same rationale as
        # SimdqEngine (fork-friendly preprocessing).
        self._init_model_kwargs = kwargs
        self.init_client()

    # ===== Init =====

    def init_model(self, **kwargs):
        if self.model is not None:
            return
        self.model = DenseEmbeddingFunction(
            self.config.model_name,
            **self.config.__dict__,
        )
        self.hidden_dim = self.model.embedding_dim

    def _ensure_model(self):
        if self.model is None:
            self.init_model(**self._init_model_kwargs)

    def init_client(self):
        os.makedirs(os.path.join(self.persist_directory, self.SUBDIR), exist_ok=True)

    def check_client(self):
        pass

    # ===== Paths / has-index =====

    def _index_dir(self, index_name: Optional[str] = None) -> str:
        if index_name is None:
            index_name = self.config.index_name
        return os.path.join(self.persist_directory, self.SUBDIR, index_name)

    def _metadata_path(self, index_name: Optional[str] = None) -> str:
        return os.path.join(self._index_dir(index_name), "engine_metadata.json")

    def has_index(self, index_name: str) -> bool:
        return os.path.exists(os.path.join(self._index_dir(index_name), "meta.json"))

    def create_index(self, index_name: Optional[str] = None, **kwargs):
        # No-op: index files materialize at ingest end. We just clear any
        # existing partial directory.
        d = self._index_dir(index_name)
        if os.path.exists(d):
            shutil.rmtree(d)

    def delete_index(self, index_name: str, **kwargs):
        d = self._index_dir(index_name)
        if os.path.exists(d):
            shutil.rmtree(d)
            logging.info(f"Deleted fagin index: {index_name}")
        self.index = None
        self.id_map = []
        self.metadata_store = {}

    # ===== Ingest =====

    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs) -> bool:
        self.check_client()
        self._ensure_model()
        fmt = "\n=== {:30} ==="
        still_create_index = self.create_update_index(fmt=fmt, do_update=update)
        if not still_create_index:
            return None

        tm = timer("fagin::ingest")
        corpus_size = len(corpus)
        batch = self.ingestion_batch_size or self.config.bulk_batch or 256

        all_vecs: List[np.ndarray] = []
        all_ids: List[str] = []
        self.metadata_store = {}

        tq = tqdm(desc="fagin ingest", total=corpus_size, leave=True)
        for i in range(0, corpus_size, batch):
            chunk = corpus[i:min(i + batch, corpus_size)]
            texts: List[str] = []
            ids:   List[str] = []
            for j, doc in enumerate(chunk):
                t = _trim_json(get_param(doc, self.text_header, ""),
                               max_string_len=self.config.max_text_size)
                if not t:
                    continue
                doc_id = get_param(doc, self.id_header, f"doc_{i+j}")
                ids.append(doc_id)
                texts.append(t)
                meta = {
                    "text": t,
                    "title": get_param(doc, self.title_header, ""),
                }
                for f in self.extra_fields:
                    meta[f] = str(get_param(doc, f, ""))
                self.metadata_store[doc_id] = meta
            if not texts:
                tq.update(len(chunk))
                continue
            embs = self.model.encode(texts, show_progress_bar=False,
                                     _batch_size=len(texts), tm=tm)
            all_vecs.append(np.asarray(embs, dtype=np.float32))
            all_ids.extend(ids)
            tm.add_timing("encoding")
            tq.update(len(chunk))
        tq.close()

        if not all_vecs:
            raise RuntimeError("fagin ingest: no documents survived text filtering")
        Y = np.vstack(all_vecs)
        if Y.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f"fagin ingest: encoder produced dim {Y.shape[1]} but "
                f"hidden_dim={self.hidden_dim}"
            )

        self.index = FaginIndex.build(vectors=Y, encoder_id=self.config.model_name)
        out_dir = self._index_dir()
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        self.index.save(out_dir)
        # Side-car: id_map + metadata for rehydration.
        with open(self._metadata_path(), "w") as f:
            json.dump({"id_map": all_ids, "metadata": self.metadata_store}, f)
        self.id_map = all_ids
        tm.add_timing("build_save")
        logging.info(f"Ingested {corpus_size} docs into fagin index "
                     f"{self.config.index_name}")
        return True

    # ===== Search =====

    def _ensure_loaded(self):
        if self.index is None:
            self.index = FaginIndex.load(self._index_dir())
            with open(self._metadata_path()) as f:
                side = json.load(f)
            self.id_map = side["id_map"]
            self.metadata_store = side["metadata"]

    def precompute_query_embeddings(self, queries) -> None:
        """Batch-encode all query texts in one GPU forward pass (see
        SimdqEngine.precompute_query_embeddings for the rationale)."""
        self._ensure_loaded()
        self._ensure_model()
        items = list(queries)
        texts = [(q.text if hasattr(q, "text") else q) for q in items]
        if not texts:
            return
        tm = timer("fagin::precompute_query_embeddings")
        embs = self.model.encode(texts, show_progress_bar=False,
                                 prompt_name="query", tm=tm)
        embs = np.asarray(embs, dtype=np.float32)
        self._query_emb_cache = {id(q): embs[i] for i, q in enumerate(items)}
        tm.add_timing("encode_batch")

    def search_all(self, queries, num_threads: int = 1) -> List[SearchResult]:
        """Batch-encode all queries on the GPU, then scan in parallel on the
        CPU (thread pool; the native TA scan releases the GIL). Same two-phase
        structure as SimdqEngine.search_all."""
        self._ensure_loaded()
        items = list(queries)
        if not items:
            return []
        if num_threads is not None and num_threads < 0:
            num_threads = os.cpu_count() or 1

        # Phase 1: one batched GPU encode for all queries.
        self.precompute_query_embeddings(items)

        # Phase 2: parallel CPU scan.
        if num_threads is None or num_threads <= 1:
            return [self.search(q, scan_threads=1)
                    for q in tqdm(items, desc="fagin search", leave=True)]

        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: List[Optional[SearchResult]] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futs = {ex.submit(self.search, q, scan_threads=1): i
                    for i, q in enumerate(items)}
            with tqdm(total=len(items), desc="fagin search", leave=True) as tk:
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()
                    tk.update(1)
        return results

    def _accumulate_stats(self, stats: dict) -> None:
        with self._stats_lock:
            self.ta_stats["queries"] += 1
            for k in _STAT_KEYS:
                self.ta_stats[k] += int(stats.get(k, 0))

    def search(self, query: SearchQueries.Query, scan_threads: Optional[int] = None,
               **kwargs) -> SearchResult:
        tm = timer("fagin::search")
        self._ensure_loaded()
        tm.add_timing("load")

        cached = self._query_emb_cache.get(id(query))
        if cached is not None:
            q = np.asarray(cached, dtype=np.float32)
        else:
            self._ensure_model()
            text = query.text if hasattr(query, "text") else query
            emb = self.model.encode([text], show_progress_bar=False,
                                    prompt_name="query", tm=tm)[0]
            q = np.asarray(emb, dtype=np.float32)
        tm.add_timing("encode")

        K = int(self.config.top_k)
        idxs, scores, stats = self.index.search(
            q, K=K,
            batch=self.config.fagin_batch_rows,
            epsilon=self.config.fagin_epsilon,
            max_depth=self.config.fagin_max_depth,
            num_threads=(self.config.fagin_num_threads
                         if scan_threads is None else scan_threads),
        )
        self._accumulate_stats(stats)
        tm.add_timing("scan")

        passages = []
        for k_idx, score in zip(idxs, scores):
            if int(k_idx) < 0:
                continue
            doc_id = self.id_map[int(k_idx)]
            meta = self.metadata_store.get(doc_id, {})
            passages.append({
                "id": doc_id,
                "text": meta.get("text", ""),
                "title": meta.get("title", ""),
                "score": float(score),
                **{f: meta.get(f, "") for f in self.extra_fields},
            })
        tm.add_timing("result")
        return SearchResult(query, passages)

    # ===== Info =====

    def info(self) -> Dict[str, Any]:
        out = {
            "retriever_type": "FaginThresholdEngine",
            "index_name": self.config.index_name,
            "model": self.config.model_name,
            "dimension": self.hidden_dim,
            "batch_rows": self.config.fagin_batch_rows,
            "epsilon": self.config.fagin_epsilon,
            "max_depth": self.config.fagin_max_depth,
            "ta_stats": dict(self.ta_stats),
        }
        try:
            with open(os.path.join(self._index_dir(), "meta.json")) as f:
                out["index_meta"] = json.load(f)
        except FileNotFoundError:
            pass
        return out
