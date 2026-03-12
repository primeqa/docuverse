"""
Matryoshka-Adaptor trainer.

Implements the training method from:
  "Matryoshka-Adaptor: Unsupervised and Supervised Tuning for Smaller
   Embedding Dimensions" (Yoon et al., EMNLP 2024)

The adapter is a shallow MLP with skip connection:
    adapted = original + f(original)

Training objective (unsupervised, Eq. 4):
    min_f  L_topk(f) + alpha * L_pair(f) + beta * L_rec(f)

Training objective (supervised, Eq. 6):
    min_f  L_topk(f) + alpha * L_pair(f) + beta * L_rec(f) + gamma * L_rank(f)

Two-stage training: first unsupervised (Eq. 4), then supervised (Eq. 6).
"""

import os
from typing import Optional

import numpy as np
import torch

from .base_trainer import BaseMatryoshkaTrainer, TrainingBatch
from .config import MatryoshkaTrainingConfig
from .losses import (
    pairwise_similarity_loss,
    topk_similarity_loss,
    reconstruction_loss,
    ranking_loss_triplet,
)
from .models import AdaptorMLP


class MatryoshkaAdaptorTrainer(BaseMatryoshkaTrainer):
    """Trainer for the Matryoshka-Adaptor (Yoon et al., EMNLP 2024).

    Learns a shallow MLP f: R^d -> R^d with skip connection such that
    adapted = original + f(original) has improved Matryoshka properties.
    """

    def __init__(self, config: MatryoshkaTrainingConfig):
        super().__init__(config)

    def build_model(self) -> AdaptorMLP:
        """Build the adapter MLP."""
        return AdaptorMLP(
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
        )

    def compute_loss(self, batch: TrainingBatch) -> torch.Tensor:
        """Compute the Matryoshka-Adaptor loss.

        Unsupervised (always):
            L_topk + alpha * L_pair + beta * L_rec

        Supervised (if query/pos/neg are in batch):
            + gamma * L_rank
        """
        dim_weights = self.config.get_dim_weights()
        matryoshka_dims = self.config.matryoshka_dims

        # Adapter forward: compute residual and adapted embeddings
        residual = self.model(batch.corpus_embs)  # f(ce)
        adapted = batch.corpus_embs + residual  # Skip connection: ce + f(ce)

        # Pre-gathered neighbor embeddings from the full corpus
        orig_nn_embs = batch.neighbor_embs  # (B, k, d)
        # Adapt the neighbor embeddings through the same adapter
        B, k, d = orig_nn_embs.shape
        nn_flat = orig_nn_embs.reshape(B * k, d)
        nn_residual = self.model(nn_flat)
        adapted_nn_embs = (nn_flat + nn_residual).reshape(B, k, d)

        # L_topk: preserve top-k neighbor similarities
        loss = topk_similarity_loss(
            batch.corpus_embs, adapted, batch.neighbor_idx,
            matryoshka_dims, dim_weights,
            original_neighbor_embs=orig_nn_embs,
            adapted_neighbor_embs=adapted_nn_embs,
        )

        # L_pair: preserve pairwise similarities
        loss = loss + self.config.alpha * pairwise_similarity_loss(
            batch.corpus_embs, adapted, matryoshka_dims, dim_weights,
        )

        # L_rec: reconstruction regularizer
        loss = loss + self.config.beta * reconstruction_loss(
            batch.corpus_embs, residual,
        )

        # L_rank: supervised ranking loss (if available)
        if (
            self.config.supervised
            and hasattr(batch, "query_embs")
            and batch.query_embs is not None
        ):
            q_residual = self.model(batch.query_embs)
            adapted_q = batch.query_embs + q_residual
            adapted_pos = batch.pos_embs + self.model(batch.pos_embs)
            adapted_neg = batch.neg_embs + self.model(batch.neg_embs)

            loss = loss + self.config.gamma * ranking_loss_triplet(
                adapted_q, adapted_pos, adapted_neg,
                matryoshka_dims, dim_weights,
            )

        return loss

    def apply_to_embeddings(
        self, embeddings: np.ndarray, dim: Optional[int] = None
    ) -> np.ndarray:
        """Apply the trained adapter to embeddings.

        Args:
            embeddings: (N, d) numpy array
            dim: optional prefix size

        Returns:
            (N, d) or (N, dim) adapted embeddings
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            adapted = self.model.adapt(x)  # x + f(x)
            if dim is not None:
                adapted = adapted[:, :dim]
            return adapted.cpu().numpy()

    def train(self):
        """Two-stage training as described in Section 4.2.

        Stage 1: Unsupervised training with L_topk + alpha*L_pair + beta*L_rec
        Stage 2: Supervised training with all four losses (if supervised mode)

        If a ``best_model.pt`` checkpoint already exists in the output
        directory, Stage 1 is skipped and the checkpoint is loaded instead.
        Stage 2 (supervised fine-tuning) always runs when supervised=True.
        """
        if self.config.supervised:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")

            if os.path.exists(best_path):
                # Load existing unsupervised checkpoint, skip Stage 1
                print("\n--- Stage 1: Unsupervised (loading existing checkpoint) ---")
                print(f"  Found existing checkpoint: {best_path}")

                # Ensure embeddings and model are ready
                if self.dataset.corpus_embeddings is None:
                    self.dataset.load_corpus_embeddings()
                if self.dataset.query_embeddings is None:
                    self.dataset.load_query_embeddings()
                if self.dataset.relevance is None:
                    self.dataset.load_relevance()
                if self.model is None:
                    self._init_model_and_optimizer()

                self.load(best_path)
                print(f"  Loaded model from step {self.global_step} "
                      f"(best_val_loss={self.best_val_loss:.4f})")
            else:
                # Stage 1: Unsupervised training from scratch
                print("\n--- Stage 1: Unsupervised pre-training ---")
                orig_supervised = self.config.supervised
                self.config.supervised = False
                super().train()
                self.config.supervised = orig_supervised

            # Stage 2: Supervised fine-tuning (always runs)
            print("\n--- Stage 2: Supervised fine-tuning ---")
            # Reset training state but keep model weights
            self.global_step = 0
            self.best_val_loss = float("inf")
            self.patience_counter = 0
            # Lower LR for fine-tuning
            if self.optimizer is not None:
                for g in self.optimizer.param_groups:
                    g["lr"] = self.config.lr * 0.1
            # Bypass checkpoint detection so Stage 2 actually trains
            self._skip_checkpoint_load = True
            super().train()
            self._skip_checkpoint_load = False
        else:
            super().train()
