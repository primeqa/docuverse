"""
Data loading, embedding caching, and neighbor mining for Matryoshka training.
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path for imports
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class EmbeddingDataset:
    """Manages embedding loading, caching, and neighbor mining.

    Supports two modes:
    1. Load pre-computed embeddings from pickle files
    2. Compute embeddings from JSONL text files using a SentenceTransformer model

    For unsupervised training, mines top-k nearest neighbors from full-dim embeddings.
    For supervised training, loads query-corpus relevance pairs.
    """

    def __init__(self, config):
        """
        Args:
            config: MatryoshkaTrainingConfig
        """
        self.config = config
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.corpus_ids: Optional[List[str]] = None
        self.query_embeddings: Optional[np.ndarray] = None
        self.query_ids: Optional[List[str]] = None
        self.neighbor_indices: Optional[np.ndarray] = None
        self.relevance: Optional[Dict] = None
        self.embedding_dim: int = 0

        # Train/val split indices
        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None

    def load_corpus_embeddings(self) -> np.ndarray:
        """Load or compute corpus embeddings.

        Returns:
            (N, d) numpy array of corpus embeddings
        """
        if self.corpus_embeddings is not None:
            return self.corpus_embeddings

        if self.config.embeddings_cache and os.path.exists(self.config.embeddings_cache):
            print(f"Loading cached embeddings from {self.config.embeddings_cache}")
            self.corpus_embeddings, self.corpus_ids = self._load_pickle(
                self.config.embeddings_cache
            )
        elif self.config.corpus_file:
            print(f"Computing embeddings from {self.config.corpus_file}")
            self.corpus_embeddings, self.corpus_ids = self._compute_embeddings(
                self.config.corpus_file,
                self.config.text_field,
                self.config.id_field,
            )
            # Cache for next time
            if self.config.embeddings_cache:
                self._save_pickle(
                    self.config.embeddings_cache,
                    self.corpus_embeddings,
                    self.corpus_ids,
                )
        else:
            raise ValueError(
                "Either embeddings_cache or corpus_file must be provided"
            )

        self.embedding_dim = self.corpus_embeddings.shape[1]
        self._create_train_val_split()
        return self.corpus_embeddings

    def load_query_embeddings(self) -> np.ndarray:
        """Load or compute query embeddings (supervised mode).

        Returns:
            (M, d) numpy array of query embeddings
        """
        if self.query_embeddings is not None:
            return self.query_embeddings

        if self.config.query_embeddings_cache and os.path.exists(
            self.config.query_embeddings_cache
        ):
            print(
                f"Loading cached query embeddings from {self.config.query_embeddings_cache}"
            )
            self.query_embeddings, self.query_ids = self._load_pickle(
                self.config.query_embeddings_cache
            )
        elif self.config.query_file:
            print(f"Computing query embeddings from {self.config.query_file}")
            self.query_embeddings, self.query_ids = self._compute_embeddings(
                self.config.query_file,
                self.config.query_text_field,
                self.config.query_id_field,
                prompt_name="query",
            )
            if self.config.query_embeddings_cache:
                self._save_pickle(
                    self.config.query_embeddings_cache,
                    self.query_embeddings,
                    self.query_ids,
                )
        else:
            raise ValueError(
                "Either query_embeddings_cache or query_file must be provided "
                "for supervised training"
            )

        return self.query_embeddings

    def load_relevance(self) -> Dict[str, Dict[str, float]]:
        """Load query-corpus relevance pairs.

        Expects a JSONL file with fields: query_id, corpus_id, relevance (optional).

        Returns:
            Dict mapping query_id -> {corpus_id: relevance_score}
        """
        if self.relevance is not None:
            return self.relevance

        if not self.config.relevance_file:
            raise ValueError("relevance_file must be provided for supervised training")

        self.relevance = {}
        with open(self.config.relevance_file, "r") as f:
            for line in f:
                rec = json.loads(line.strip())
                qid = str(rec.get("query_id", rec.get("qid", "")))
                cid = str(rec.get("corpus_id", rec.get("docid", rec.get("pid", ""))))
                score = float(rec.get("relevance", rec.get("score", 1.0)))
                if qid not in self.relevance:
                    self.relevance[qid] = {}
                self.relevance[qid][cid] = score

        print(
            f"Loaded {sum(len(v) for v in self.relevance.values())} relevance pairs "
            f"for {len(self.relevance)} queries"
        )
        return self.relevance

    def _neighbor_cache_path(self, k: int) -> Optional[str]:
        """Derive the neighbor cache path from embeddings_cache + '.top{k}'."""
        if self.config.embeddings_cache:
            return f"{self.config.embeddings_cache}.top{k}"
        return None

    def mine_topk_neighbors(self, k: Optional[int] = None) -> np.ndarray:
        """Mine top-k nearest neighbors in full-dimensional embedding space.

        Uses brute-force cosine similarity for small corpora, FAISS for large ones.
        Results are cached to ``{embeddings_cache}.top{k}`` so subsequent runs
        skip the expensive mining step.

        Args:
            k: number of neighbors (defaults to config.topk_neighbors)

        Returns:
            (N, k) numpy array of neighbor indices
        """
        if self.neighbor_indices is not None:
            return self.neighbor_indices

        if k is None:
            k = self.config.topk_neighbors

        # Try loading from cache
        cache_path = self._neighbor_cache_path(k)
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached top-{k} neighbors from {cache_path}")
            with open(cache_path, "rb") as f:
                self.neighbor_indices = pickle.load(f)
            print(f"  Loaded neighbor indices: shape={self.neighbor_indices.shape}")
            return self.neighbor_indices

        embeddings = self.load_corpus_embeddings()
        N = embeddings.shape[0]

        print(f"Mining top-{k} neighbors for {N} embeddings...")

        if N <= 100000:
            # Brute force: compute all pairwise similarities
            self.neighbor_indices = self._mine_neighbors_brute_force(embeddings, k)
        else:
            # Use FAISS for large corpora
            self.neighbor_indices = self._mine_neighbors_faiss(embeddings, k)

        # Save to cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.neighbor_indices, f)
            print(f"  Cached top-{k} neighbors to {cache_path}")

        return self.neighbor_indices

    def sample_corpus_batch(
        self, batch_size: int, split: str = "train"
    ) -> np.ndarray:
        """Sample a random batch of corpus embedding indices.

        Args:
            batch_size: number of embeddings to sample
            split: "train" or "val"

        Returns:
            (batch_size,) numpy array of indices
        """
        indices = self.train_indices if split == "train" else self.val_indices
        if indices is None:
            indices = np.arange(len(self.corpus_embeddings))
        return np.random.choice(indices, size=min(batch_size, len(indices)), replace=False)

    def sample_supervised_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of (query, positive, negative) triplets.

        Returns:
            Tuple of (query_indices, pos_corpus_indices, neg_corpus_indices)
        """
        relevance = self.load_relevance()
        query_ids = self.query_ids
        corpus_ids = self.corpus_ids

        # Build ID -> index maps
        qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
        cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

        # Collect valid triplets
        q_indices = []
        pos_indices = []
        neg_indices = []

        query_keys = list(relevance.keys())
        np.random.shuffle(query_keys)

        for qid in query_keys:
            if len(q_indices) >= batch_size:
                break
            if qid not in qid_to_idx:
                continue

            pos_cids = [
                cid for cid, score in relevance[qid].items()
                if score > 0 and cid in cid_to_idx
            ]
            if not pos_cids:
                continue

            qi = qid_to_idx[qid]
            pi = cid_to_idx[np.random.choice(pos_cids)]
            # Random negative
            ni = np.random.randint(len(corpus_ids))
            while str(corpus_ids[ni]) in relevance.get(qid, {}):
                ni = np.random.randint(len(corpus_ids))

            q_indices.append(qi)
            pos_indices.append(pi)
            neg_indices.append(ni)

        return (
            np.array(q_indices),
            np.array(pos_indices),
            np.array(neg_indices),
        )

    # ---- Internal helpers ----

    def _create_train_val_split(self):
        """Create train/val split of corpus indices."""
        N = len(self.corpus_embeddings)
        indices = np.random.permutation(N)
        val_size = max(1, int(N * self.config.val_fraction))
        self.val_indices = indices[:val_size]
        self.train_indices = indices[val_size:]

    def _load_pickle(self, path: str) -> Tuple[np.ndarray, List[str]]:
        """Load embeddings from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        embeddings = data["embeddings"]
        ids = data.get("ids", [str(i) for i in range(len(embeddings))])
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        print(f"  Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
        return embeddings, ids

    def _save_pickle(self, path: str, embeddings: np.ndarray, ids: List[str]):
        """Save embeddings to pickle file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "ids": ids}, f)
        print(f"  Saved {len(ids)} embeddings to {path}")

    def _compute_embeddings(
        self,
        jsonl_path: str,
        text_field: str,
        id_field: str,
        prompt_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute embeddings from a JSONL file using SentenceTransformer."""
        from sentence_transformers import SentenceTransformer
        from docuverse.utils import detect_device

        # Load texts
        texts, ids = [], []
        with open(jsonl_path, "r") as f:
            for line in f:
                rec = json.loads(line.strip())
                texts.append(str(rec[text_field]))
                ids.append(str(rec[id_field]))

        print(f"  Loaded {len(texts)} texts from {jsonl_path}")

        # Load model
        device = self.config.device
        if device == "auto":
            device = detect_device()
        model = SentenceTransformer(self.config.model_name, device=device, trust_remote_code=True)

        # Encode
        embeddings = model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            prompt_name=prompt_name,
        )

        return embeddings, ids

    def _mine_neighbors_brute_force(
        self, embeddings: np.ndarray, k: int
    ) -> np.ndarray:
        """Brute-force top-k neighbor mining using cosine similarity."""
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = embeddings / norms

        N = normed.shape[0]
        neighbors = np.zeros((N, k), dtype=np.int64)

        batch_size = min(1000, N)
        for i in tqdm(range(0, N, batch_size), desc="Mining neighbors"):
            end = min(i + batch_size, N)
            sims = normed[i:end] @ normed.T  # (batch, N)
            # Zero out self-similarity
            for j in range(end - i):
                sims[j, i + j] = -1.0
            # Top-k
            top_indices = np.argpartition(-sims, k, axis=1)[:, :k]
            # Sort the top-k by similarity
            for j in range(end - i):
                top_sims = sims[j, top_indices[j]]
                sorted_order = np.argsort(-top_sims)
                neighbors[i + j] = top_indices[j][sorted_order]

        print(f"  Mined {k} neighbors for {N} embeddings")
        return neighbors

    def _mine_neighbors_faiss(
        self, embeddings: np.ndarray, k: int
    ) -> np.ndarray:
        """FAISS-based top-k neighbor mining using HNSW for large corpora.

        Uses HNSW (approximate) for N > 500K, flat index (exact) for smaller.
        HNSW build is O(N log N) and search is O(log N) per query, making it
        practical for millions of vectors.
        """
        try:
            import faiss
        except ImportError:
            print("FAISS not available, falling back to brute force")
            return self._mine_neighbors_brute_force(embeddings, k)

        N, d = embeddings.shape
        embs = embeddings.astype(np.float32).copy()

        # Normalize for cosine similarity
        faiss.normalize_L2(embs)

        if N > 500_000:
            # HNSW — approximate but fast for large corpora
            M = 32          # connections per node (higher = better recall, more RAM)
            ef_construction = 200  # build-time beam width
            ef_search = max(128, k * 8)  # search-time beam width

            print(f"  Building HNSW index (M={M}, ef_construction={ef_construction})...")
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            index.add(embs)
            print(f"  HNSW index built. Searching with ef_search={ef_search}...")
        else:
            # Exact flat index for moderate sizes
            index = faiss.IndexFlatIP(d)
            index.add(embs)

        # Search (k+1 to exclude self)
        _, indices = index.search(embs, k + 1)

        # Remove self from results
        neighbors = np.zeros((N, k), dtype=np.int64)
        for i in range(N):
            mask = indices[i] != i
            nn = indices[i][mask][:k]
            neighbors[i, : len(nn)] = nn

        index_type = "HNSW" if N > 500_000 else "Flat"
        print(f"  Mined {k} neighbors for {N} embeddings (FAISS {index_type})")
        return neighbors
