"""
Permutation-based Matryoshka trainer.

Implements the post-hoc permutation approach from the whitepaper:
  "Post-Hoc Embedding Permutation for Matryoshka-Style Embeddings"

Learns a permutation of embedding dimensions such that prefix truncation
of the reordered vector preserves retrieval quality. Key advantage: the
full-dimensional inner product is preserved (backward-compatible).

Approaches:
- Score-vector + NeuralSort (d parameters, recommended)
- Full Sinkhorn (d^2 parameters, optional)
- Gumbel-Sinkhorn (Sinkhorn + Gumbel noise for exploration)
- Variance-sort (zero parameters, baseline)
"""

import math
import os
from typing import Optional

import numpy as np
import torch

from .base_trainer import BaseMatryoshkaTrainer, TrainingBatch
from .config import MatryoshkaTrainingConfig
from .losses import (
    infonce_loss,
    neighborhood_preservation_loss,
    topk_similarity_loss,
    pairwise_similarity_loss,
)
from .models import (
    ScoreVectorPermutation,
    SinkhornPermutation,
    VarianceSortPermutation,
)


class PermutationTrainer(BaseMatryoshkaTrainer):
    """Trainer for permutation-based Matryoshka dimensionality reduction.

    Supports three parameterizations:
    1. "permutation-score": Score-vector + NeuralSort (d params, recommended)
    2. "permutation-sinkhorn": Full d x d Sinkhorn matrix (d^2 params)
    3. "variance-sort": Zero-parameter baseline

    The key property is that permutation preserves full-dim cosine similarity,
    enabling backward-compatible deployment with existing vector indexes.
    """

    def __init__(self, config: MatryoshkaTrainingConfig):
        super().__init__(config)

    def build_model(self):
        """Build the permutation model based on config.method."""
        method = self.config.method
        d = self.config.embedding_dim

        if method == "permutation-score":
            return ScoreVectorPermutation(d, tau=self.config.tau_start)
        elif method == "permutation-sinkhorn":
            return SinkhornPermutation(
                d,
                tau=self.config.tau_start,
                n_iters=self.config.sinkhorn_iters,
                use_gumbel=self.config.use_gumbel_noise,
            )
        elif method == "variance-sort":
            return VarianceSortPermutation(d)
        else:
            raise ValueError(f"Unknown permutation method: {method}")

    def compute_loss(self, batch: TrainingBatch) -> torch.Tensor:
        """Compute the permutation training loss.

        For the score-vector and Sinkhorn approaches, the loss encourages
        the learned permutation to place discriminative dimensions first.

        Unsupervised: neighborhood preservation loss (or topk/pairwise similarity)
        Supervised: InfoNCE at each prefix size
        """
        dim_weights = self.config.get_dim_weights()
        matryoshka_dims = self.config.matryoshka_dims

        # Update temperature based on training progress
        self._update_temperature()

        if self.config.supervised and hasattr(batch, "query_embs") and batch.query_embs is not None:
            return self._supervised_loss(batch, matryoshka_dims, dim_weights)
        else:
            return self._unsupervised_loss(batch, matryoshka_dims, dim_weights)

    def _supervised_loss(
        self,
        batch: TrainingBatch,
        matryoshka_dims: list,
        dim_weights: list,
    ) -> torch.Tensor:
        """Supervised loss: InfoNCE at each prefix size.

        For each prefix size m, apply soft permutation and compute InfoNCE
        using the truncated prefix.
        """
        model = self.model

        # Soft-permute query, positive, and negative embeddings
        q_perm = model.soft_permute(batch.query_embs)
        pos_perm = model.soft_permute(batch.pos_embs)
        neg_perm = model.soft_permute(batch.neg_embs)

        return infonce_loss(
            q_perm, pos_perm, neg_perm,
            matryoshka_dims,
            temperature=0.05,
            dim_weights=dim_weights,
        )

    def _unsupervised_loss(
        self,
        batch: TrainingBatch,
        matryoshka_dims: list,
        dim_weights: list,
    ) -> torch.Tensor:
        """Unsupervised loss: preserve neighborhood structure.

        Uses the full-dim top-k neighbors as supervision signal.
        The loss encourages the permuted prefix to maintain the same
        similarity relationships as the full-dim space.
        """
        model = self.model
        permuted = model.soft_permute(batch.corpus_embs)

        # Soft-permute the pre-gathered neighbor embeddings
        orig_nn_embs = batch.neighbor_embs  # (B, k, d)
        B, k, d = orig_nn_embs.shape
        nn_flat = orig_nn_embs.reshape(B * k, d)
        permuted_nn = model.soft_permute(nn_flat).reshape(B, k, d)

        # Use top-k similarity preservation (analogous to Adaptor's L_topk)
        loss = topk_similarity_loss(
            batch.corpus_embs, permuted, batch.neighbor_idx,
            matryoshka_dims, dim_weights,
            original_neighbor_embs=orig_nn_embs,
            adapted_neighbor_embs=permuted_nn,
        )

        # Also add pairwise similarity preservation
        loss = loss + self.config.alpha * pairwise_similarity_loss(
            batch.corpus_embs, permuted,
            matryoshka_dims, dim_weights,
        )

        return loss

    def _update_temperature(self):
        """Cosine annealing of temperature from tau_start to tau_end."""
        if not hasattr(self.model, "set_temperature"):
            return

        progress = self.global_step / max(self.config.max_iterations, 1)
        tau = self.config.tau_end + 0.5 * (self.config.tau_start - self.config.tau_end) * (
            1 + math.cos(math.pi * progress)
        )
        self.model.set_temperature(tau)

    def apply_to_embeddings(
        self, embeddings: np.ndarray, dim: Optional[int] = None
    ) -> np.ndarray:
        """Apply the learned permutation to embeddings.

        At inference, uses the hard permutation (index selection, O(d)).

        Args:
            embeddings: (N, d) numpy array
            dim: optional prefix size

        Returns:
            (N, d) or (N, dim) permuted embeddings
        """
        if isinstance(self.model, VarianceSortPermutation):
            return self.model.apply(
                torch.tensor(embeddings, dtype=torch.float32), dim=dim
            ).numpy()

        self.model.eval()
        with torch.no_grad():
            perm = self.model.hard_permutation()
            result = embeddings[:, perm.cpu().numpy()]
            if dim is not None:
                result = result[:, :dim]
            return result

    def get_dimension_importance(self) -> np.ndarray:
        """Get the learned dimension importance ranking.

        Returns:
            (d,) array where result[i] is the importance rank of original dimension i.
            Lower rank = more important (placed earlier in permutation).
        """
        if isinstance(self.model, VarianceSortPermutation):
            perm = self.model.hard_permutation()
        elif isinstance(self.model, ScoreVectorPermutation):
            perm = self.model.hard_permutation()
        elif isinstance(self.model, SinkhornPermutation):
            perm = self.model.hard_permutation()
        else:
            raise RuntimeError(f"Unknown model type: {type(self.model)}")

        # perm[i] = original dim index at position i
        # We want: rank[original_dim] = position
        rank = np.zeros(len(perm), dtype=np.int64)
        perm_np = perm.cpu().numpy() if isinstance(perm, torch.Tensor) else perm.numpy()
        for position, orig_dim in enumerate(perm_np):
            rank[orig_dim] = position
        return rank

    def verify_inner_product_preservation(
        self, embeddings: np.ndarray, n_pairs: int = 1000
    ) -> float:
        """Verify that full-dim cosine similarity is preserved after permutation.

        Returns the maximum absolute difference in cosine similarity
        between original and permuted embeddings (should be ~0).
        """
        N = embeddings.shape[0]
        permuted = self.apply_to_embeddings(embeddings)

        max_diff = 0.0
        for _ in range(n_pairs):
            i, j = np.random.randint(0, N, size=2)
            orig_sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
            )
            perm_sim = np.dot(permuted[i], permuted[j]) / (
                np.linalg.norm(permuted[i]) * np.linalg.norm(permuted[j]) + 1e-8
            )
            max_diff = max(max_diff, abs(orig_sim - perm_sim))

        return max_diff

    def train(self):
        """Train the permutation model."""
        # Load embeddings first so we know the embedding_dim
        self.dataset.load_corpus_embeddings()
        if self.config.supervised:
            self.dataset.load_query_embeddings()
            self.dataset.load_relevance()
        self._init_model_and_optimizer()

        if isinstance(self.model, VarianceSortPermutation):
            # No training needed - just compute permutation from corpus variance
            print("Computing variance-sort permutation (no training)...")
            corpus_tensor = torch.tensor(
                self.dataset.corpus_embeddings, dtype=torch.float32
            )
            self.model.fit(corpus_tensor)
            self.dataset.mine_topk_neighbors()
            self._evaluate_all_dims(corpus_tensor)

            # Verify inner product preservation
            max_diff = self.verify_inner_product_preservation(
                self.dataset.corpus_embeddings[:1000]
            )
            print(f"\nInner product preservation check: max diff = {max_diff:.2e}")

            self.save(os.path.join(self.config.output_dir, "best_model.pt"))
        else:
            # Model is already built; call the parent train which will skip
            # re-loading embeddings and re-building the model
            super().train()

            # After training, verify inner product preservation
            if self.dataset.corpus_embeddings is not None:
                max_diff = self.verify_inner_product_preservation(
                    self.dataset.corpus_embeddings[:1000]
                )
                print(f"\nInner product preservation check: max diff = {max_diff:.2e}")
                if max_diff > 1e-5:
                    print(
                        "  WARNING: Inner product not fully preserved. "
                        "This is expected for soft permutation; use hard_permutation() for inference."
                    )
