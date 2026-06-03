"""
Abstract base trainer for Matryoshka adapter and permutation training.

Provides the shared training loop, evaluation, checkpointing, and early
stopping logic. Subclasses implement build_model(), compute_loss(), and
apply_to_embeddings().
"""

import math
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import MatryoshkaTrainingConfig
from .data import EmbeddingDataset

_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from docuverse.utils.timer import timer


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


class TrainingBatch:
    """Container for a training batch."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class BaseMatryoshkaTrainer(ABC):
    """Abstract base class for Matryoshka training methods.

    Subclasses must implement:
    - build_model(): create the trainable model
    - compute_loss(batch): compute the training loss
    - apply_to_embeddings(embeddings, dim): apply the trained transform
    """

    def __init__(self, config: MatryoshkaTrainingConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.dataset = EmbeddingDataset(config)

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Build model (implemented by subclass)
        self.model = None  # Set after embeddings are loaded (need embedding_dim)
        self.optimizer = None

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self._skip_checkpoint_load = False  # Set True to bypass checkpoint detection

    def _init_model_and_optimizer(self):
        """Initialize model and optimizer after embedding dim is known."""
        if self.config.embedding_dim == 0:
            self.config.embedding_dim = self.dataset.embedding_dim

        self.model = self.build_model()
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        # Build optimizer
        params = list(self.model.parameters())
        if not params:
            self.optimizer = None
            return

        if self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.config.lr)
        elif self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=self.config.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    @abstractmethod
    def build_model(self):
        """Create the trainable model. Called after embedding_dim is known."""
        ...

    @abstractmethod
    def compute_loss(self, batch: TrainingBatch) -> torch.Tensor:
        """Compute training loss for a batch.

        Args:
            batch: TrainingBatch with corpus_embs, neighbor_idx, and optionally
                   query_embs, pos_embs, neg_embs

        Returns:
            Scalar loss tensor
        """
        ...

    @abstractmethod
    def apply_to_embeddings(
        self, embeddings: np.ndarray, dim: Optional[int] = None
    ) -> np.ndarray:
        """Apply the trained transformation to embeddings.

        Args:
            embeddings: (N, d) numpy array
            dim: optional prefix size to truncate to

        Returns:
            (N, d) or (N, dim) transformed embeddings
        """
        ...

    def train(self):
        """Main training loop.

        1. Load/compute embeddings
        2. Mine neighbors (for unsupervised losses)
        3. Iterate: sample batch, compute loss, backprop
        4. Periodically evaluate on validation set
        5. Early stopping based on validation loss
        6. Save best checkpoint
        """
        tm = timer("matryoshka_train::train")

        print(f"\n{'='*60}")
        print(f"Training {self.config.method} | device={self.device}")
        print(f"{'='*60}\n")

        # Step 1: Load embeddings (skip if already loaded by subclass)
        if self.dataset.corpus_embeddings is None:
            self.dataset.load_corpus_embeddings()
        if self.config.supervised:
            if self.dataset.query_embeddings is None:
                self.dataset.load_query_embeddings()
            if self.dataset.relevance is None:
                self.dataset.load_relevance()

        tm.add_timing("load_embeddings")

        # Step 2: Initialize model now that we know embedding_dim
        # (skip if already initialized by subclass)
        if self.model is None:
            self._init_model_and_optimizer()

        tm.add_timing("init_model")

        if self.optimizer is None:
            # No trainable params (e.g., variance-sort)
            print("No trainable parameters. Running baseline computation...")
            self._run_baseline()
            return

        # Check for existing checkpoint
        if not self._skip_checkpoint_load:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            if os.path.exists(best_path):
                print(f"\n  Found existing checkpoint: {best_path}")
                print(f"  Loading pre-trained model (skipping training)...")
                self.load(best_path)
                print(f"  Loaded model from step {self.global_step} "
                      f"(best_val_loss={self.best_val_loss:.4f})")

                # Run evaluation only
                corpus_tensor = torch.tensor(
                    self.dataset.corpus_embeddings, dtype=torch.float32
                )
                self.dataset.mine_topk_neighbors()
                self._evaluate_all_dims(corpus_tensor)
                tm.add_timing("checkpoint_load_and_eval")
                return

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
        print(f"Embedding dim: {self.config.embedding_dim}")
        print(f"Matryoshka dims: {self.config.matryoshka_dims}")
        print(f"Max iterations: {self.config.max_iterations}")
        print(f"Batch size: {self.config.batch_size}")
        print()

        # Step 3: Mine neighbors (unsupervised)
        self.dataset.mine_topk_neighbors()

        tm.add_timing("mine_neighbors")

        # Convert corpus embeddings to tensor (keep on CPU, move batches to device)
        corpus_tensor = torch.tensor(
            self.dataset.corpus_embeddings, dtype=torch.float32
        )
        neighbor_tensor = torch.tensor(
            self.dataset.neighbor_indices, dtype=torch.long
        )

        query_tensor = None
        if self.config.supervised and self.dataset.query_embeddings is not None:
            query_tensor = torch.tensor(
                self.dataset.query_embeddings, dtype=torch.float32
            )

        # Step 4: Training loop
        os.makedirs(self.config.output_dir, exist_ok=True)
        train_losses = []

        pbar = tqdm(range(1, self.config.max_iterations + 1), desc="Training")
        for step in pbar:
            self.global_step = step
            self.model.train()

            # Sample batch
            batch = self._sample_batch(corpus_tensor, neighbor_tensor, query_tensor)

            # Forward + backward
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()
            train_losses.append(loss_val)

            # Log
            if step % self.config.log_every == 0:
                avg_loss = np.mean(train_losses[-self.config.log_every:])
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    best_val=f"{self.best_val_loss:.4f}",
                    patience=f"{self.patience_counter}/{self.config.patience}",
                )

            # Validate & checkpoint
            if step % self.config.save_every == 0 or step == self.config.max_iterations:
                val_loss = self._validate(corpus_tensor, neighbor_tensor, query_tensor)
                print(
                    f"\n  Step {step}: train_loss={np.mean(train_losses[-100:]):.4f}, "
                    f"val_loss={val_loss:.4f}"
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save(os.path.join(self.config.output_dir, "best_model.pt"))
                    print(f"  -> New best model saved (val_loss={val_loss:.4f})")
                else:
                    self.patience_counter += self.config.save_every

                # Early stopping
                if self.patience_counter >= self.config.patience:
                    print(f"\n  Early stopping at step {step}")
                    break

        tm.add_timing("training_loop")

        # Load best model
        best_path = os.path.join(self.config.output_dir, "best_model.pt")
        if os.path.exists(best_path):
            self.load(best_path)

        # Final evaluation
        print(f"\nTraining complete ({tm.time_since_beginning()})")
        self._evaluate_all_dims(corpus_tensor)

        tm.add_timing("evaluation")

    def _sample_batch(
        self,
        corpus_tensor: torch.Tensor,
        neighbor_tensor: torch.Tensor,
        query_tensor: Optional[torch.Tensor],
    ) -> TrainingBatch:
        """Sample a training batch.

        Pre-gathers neighbor embeddings from the full corpus tensor so that
        loss functions don't need to index into the full corpus.
        """
        # Sample corpus indices
        indices = self.dataset.sample_corpus_batch(self.config.batch_size, split="train")
        corpus_embs = corpus_tensor[indices].to(self.device)
        neighbor_idx = neighbor_tensor[indices]  # (B, k) — global indices

        # Pre-gather original neighbor embeddings from full corpus
        # neighbor_idx shape: (B, k), values are global corpus indices
        neighbor_embs = corpus_tensor[neighbor_idx].to(self.device)  # (B, k, d)

        batch_kwargs = {
            "corpus_embs": corpus_embs,
            "neighbor_idx": neighbor_idx.to(self.device),
            "neighbor_embs": neighbor_embs,  # (B, k, d) pre-gathered
            "indices": indices,
        }

        # Supervised: add query/positive/negative
        if self.config.supervised and query_tensor is not None:
            q_idx, pos_idx, neg_idx = self.dataset.sample_supervised_batch(
                self.config.batch_size
            )
            if len(q_idx) > 0:
                batch_kwargs["query_embs"] = query_tensor[q_idx].to(self.device)
                batch_kwargs["pos_embs"] = corpus_tensor[pos_idx].to(self.device)
                batch_kwargs["neg_embs"] = corpus_tensor[neg_idx].to(self.device)

        return TrainingBatch(**batch_kwargs)

    def _validate(
        self,
        corpus_tensor: torch.Tensor,
        neighbor_tensor: torch.Tensor,
        query_tensor: Optional[torch.Tensor],
    ) -> float:
        """Compute validation loss."""
        self.model.eval()
        val_indices = self.dataset.val_indices
        if val_indices is None or len(val_indices) == 0:
            return float("inf")

        with torch.no_grad():
            # Sample validation batch
            sample_size = min(len(val_indices), self.config.batch_size * 4)
            idx = np.random.choice(val_indices, size=sample_size, replace=False)
            corpus_embs = corpus_tensor[idx].to(self.device)
            neighbor_idx = neighbor_tensor[idx]
            neighbor_embs = corpus_tensor[neighbor_idx].to(self.device)

            batch = TrainingBatch(
                corpus_embs=corpus_embs,
                neighbor_idx=neighbor_idx.to(self.device),
                neighbor_embs=neighbor_embs,
                indices=idx,
            )

            if self.config.supervised and query_tensor is not None:
                q_idx, pos_idx, neg_idx = self.dataset.sample_supervised_batch(
                    sample_size
                )
                if len(q_idx) > 0:
                    batch.query_embs = query_tensor[q_idx].to(self.device)
                    batch.pos_embs = corpus_tensor[pos_idx].to(self.device)
                    batch.neg_embs = corpus_tensor[neg_idx].to(self.device)

            loss = self.compute_loss(batch)
            return loss.item()

    def _evaluate_all_dims(self, corpus_tensor: torch.Tensor):
        """Evaluate at all matryoshka dimensions using distance metrics."""
        self.model.eval()
        print(f"\nEvaluation at matryoshka dimensions:")
        print(f"{'Dim':>6}  {'Pairwise Dist':>14}  {'Top-k Dist':>12}")
        print(f"{'-'*6}  {'-'*14}  {'-'*12}")

        # Use a subset for evaluation
        N = min(5000, len(corpus_tensor))
        idx = np.random.choice(len(corpus_tensor), N, replace=False)
        embs = corpus_tensor[idx]
        neighbors = torch.tensor(self.dataset.neighbor_indices[idx], dtype=torch.long)

        # Full-dim similarity matrix
        embs_norm = F.normalize(embs, p=2, dim=-1)
        full_sim = embs_norm @ embs_norm.T

        for dim in self.config.matryoshka_dims + [self.config.embedding_dim]:
            with torch.no_grad():
                adapted = self.apply_to_embeddings(embs.numpy(), dim=dim)
                adapted_t = torch.tensor(adapted, dtype=torch.float32)
                adapted_norm = F.normalize(adapted_t, p=2, dim=-1)
                adapted_sim = adapted_norm @ adapted_norm.T

            # Average pairwise distance
            pair_dist = (full_sim - adapted_sim).abs().mean().item()

            # Average top-k distance
            topk_dist = 0.0
            for i in range(N):
                nn_idx = neighbors[i]
                nn_idx = nn_idx[nn_idx < N]  # Filter to eval subset
                if len(nn_idx) > 0:
                    topk_dist += (
                        (full_sim[i, nn_idx] - adapted_sim[i, nn_idx]).abs().mean().item()
                    )
            topk_dist /= N

            label = f"{dim}" if dim < self.config.embedding_dim else f"{dim} (full)"
            print(f"{label:>6}  {pair_dist:>14.4f}  {topk_dist:>12.4f}")

    def _run_baseline(self):
        """Run baseline evaluation (for zero-parameter methods like variance-sort)."""
        corpus_tensor = torch.tensor(
            self.dataset.corpus_embeddings, dtype=torch.float32
        )
        self.dataset.mine_topk_neighbors()
        self._evaluate_all_dims(corpus_tensor)
        self.save(os.path.join(self.config.output_dir, "best_model.pt"))

    def save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "config": self.config,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        if hasattr(self.model, "state_dict"):
            state["model_state_dict"] = self.model.state_dict()
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state_dict" in state and hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
