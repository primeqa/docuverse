"""SimdqEngine — DocUVerse SearchEngine backed by an in-process simdq index.

Mirrors FAISSEngine: in-process, file-backed (no client/server split). At
ingest time we accumulate corpus-encoded vectors across all batches into a
single fp32 numpy buffer (peak RAM = 4 * N * D bytes), then build + save
the index in one shot. At search time the index is mmap-loaded once on
the first query.

Persist layout:
    <persist_directory>/simdq_data/<index_name>/
        meta.json
        W.npy
        codes.bin
        scales.bin            # asymmetric only
        floats.bin            # only if simdq_store_floats=True
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import _trim_json, get_param
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction
from docuverse.utils.timer import timer


class SimdqEngine(RetrievalEngine):
    """In-process SIMD-quantized retrieval engine.

    Reads simdq_* fields from RetrievalArguments. Index lives at
    ``<persist_directory>/simdq_data/<index_name>/``.
    """

    SUBDIR = "simdq_data"

    def __init__(self, config_params, **kwargs):
        super().__init__(config_params, **kwargs)
        self.model: Optional[DenseEmbeddingFunction] = None
        self.hidden_dim: Optional[int] = None
        self.index: Optional[SimdqIndex] = None
        self.id_map: List[str] = []
        self.metadata_store: Dict[str, dict] = {}

        self.load_model_config(config_params)
        self.text_header = "text"
        self.title_header = "title"
        self.id_header = "id"
        self.extra_fields = get_param(self.config.data_template, "extra_fields", [])
        self.persist_directory = get_param(self.config, "project_dir", "/tmp")

        self.init_model(**kwargs)
        self.init_client()

    # ===== Init =====

    def init_model(self, **kwargs):
        self.model = DenseEmbeddingFunction(
            self.config.model_name,
            **self.config.__dict__,
        )
        self.hidden_dim = self.model.embedding_dim

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
            logging.info(f"Deleted simdq index: {index_name}")
        self.index = None
        self.id_map = []
        self.metadata_store = {}

    # ===== Ingest =====

    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs) -> bool:
        self.check_client()
        fmt = "\n=== {:30} ==="
        still_create_index = self.create_update_index(fmt=fmt, do_update=update)
        if not still_create_index:
            return None

        tm = timer("simdq::ingest")
        corpus_size = len(corpus)
        batch = self.ingestion_batch_size or self.config.bulk_batch or 256

        all_vecs: List[np.ndarray] = []
        all_ids: List[str] = []
        self.metadata_store = {}

        tq = tqdm(desc="simdq ingest", total=corpus_size, leave=True)
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
            raise RuntimeError("simdq ingest: no documents survived text filtering")
        Y = np.vstack(all_vecs)
        if Y.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f"simdq ingest: encoder produced dim {Y.shape[1]} but hidden_dim={self.hidden_dim}"
            )

        self.index = SimdqIndex.build(
            vectors=Y,
            family=self.config.simdq_family,
            b=self.config.simdq_b if self.config.simdq_family == "asymmetric" else None,
            d=self.config.simdq_d,
            projection=self.config.simdq_projection,
            projection_seed=self.config.simdq_projection_seed,
            store_floats=self.config.simdq_store_floats,
            encoder_id=self.config.model_name,
            standardize=self.config.simdq_standardize,
            itq_iters=self.config.simdq_itq_iters,
        )
        out_dir = self._index_dir()
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        self.index.save(out_dir)
        # Side-car: id_map + metadata for rehydration.
        with open(self._metadata_path(), "w") as f:
            json.dump({"id_map": all_ids, "metadata": self.metadata_store}, f)
        self.id_map = all_ids
        tm.add_timing("build_save")
        logging.info(f"Ingested {corpus_size} docs into simdq index {self.config.index_name}")
        return True

    # ===== Search =====

    def _ensure_loaded(self):
        if self.index is None:
            self.index = SimdqIndex.load(self._index_dir())
            with open(self._metadata_path()) as f:
                side = json.load(f)
            self.id_map = side["id_map"]
            self.metadata_store = side["metadata"]

    def search(self, query: SearchQueries.Query, **kwargs) -> SearchResult:
        tm = timer("simdq::search")
        self._ensure_loaded()
        tm.add_timing("load")

        text = query.text if hasattr(query, "text") else query
        emb = self.model.encode([text], show_progress_bar=False,
                                prompt_name="query", tm=tm)[0]
        q = np.asarray(emb, dtype=np.float32)
        tm.add_timing("encode")

        K = int(self.config.top_k)
        K_prime = K * self.config.simdq_rescore_alpha
        K_prime = max(K, min(K_prime, 256))         # spec: K_prime <= 256

        idxs, scores = self.index.search(
            q, K=K, K_prime=K_prime,
            num_threads=self.config.simdq_num_threads,
        )
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
            "retriever_type": "SimdqEngine",
            "index_name": self.config.index_name,
            "model": self.config.model_name,
            "dimension": self.hidden_dim,
            "family": self.config.simdq_family,
            "b": self.config.simdq_b,
            "d": self.config.simdq_d,
            "projection": self.config.simdq_projection,
            "standardize": self.config.simdq_standardize,
        }
        try:
            with open(os.path.join(self._index_dir(), "meta.json")) as f:
                out["index_meta"] = json.load(f)
        except FileNotFoundError:
            pass
        return out
