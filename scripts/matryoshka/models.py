"""
Trainable models for Matryoshka adapter and permutation training.

- AdaptorMLP: Shallow MLP with skip connection (Yoon et al.)
- ScoreVectorPermutation: d-parameter NeuralSort permutation (recommended)
- SinkhornPermutation: d x d Sinkhorn-based permutation
- VarianceSortPermutation: Zero-parameter variance-sort baseline
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class AdaptorMLP(nn.Module):
    """MLP adapter with skip connection for Matryoshka-Adaptor.

    The adapter learns a residual f: R^d -> R^d.  The final adapted
    embedding is: adapted = original + f(original).

    Architecture: d -> h1 -> ReLU -> h2 -> ReLU -> ... -> d
    """

    def __init__(self, embedding_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512]

        layers = []
        in_dim = embedding_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

        # Initialize last layer to near-zero so skip connection starts as identity
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the residual (without skip connection).

        Args:
            x: (B, d) input embeddings

        Returns:
            (B, d) residual to be added to input
        """
        return self.mlp(x)

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter with skip connection.

        Args:
            x: (B, d) input embeddings

        Returns:
            (B, d) adapted embeddings = x + f(x)
        """
        return x + self.forward(x)


class ScoreVectorPermutation(nn.Module):
    """Score-vector permutation with NeuralSort relaxation.

    Learns a single score vector s in R^d where s[i] represents the
    importance of original dimension i. The permutation is argsort(s)
    descending: highest-scored dimensions come first.

    During training, uses NeuralSort to produce a soft permutation matrix:
        P_soft(s; tau)_{ij} = softmax_j(-|s_i - s_j| / tau)

    At inference, uses the hard permutation: pi = argsort(-s).

    Parameters: exactly d scalars.
    """

    def __init__(self, embedding_dim: int, tau: float = 1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scores = nn.Parameter(torch.randn(embedding_dim))
        self.tau = tau

    def set_temperature(self, tau: float):
        self.tau = tau

    def _neuralsort_matrix(self) -> torch.Tensor:
        """Compute the NeuralSort soft permutation matrix.

        P_soft_{ij} = softmax_j(-|s_i - sorted_s_j| / tau)

        More precisely, following Grover et al. (2019):
        Sort s descending to get s_sorted. The soft permutation matrix
        has row i corresponding to output position i, giving a distribution
        over which input dimension should occupy that position.

        Returns:
            (d, d) soft permutation matrix (doubly stochastic approximation)
        """
        d = self.embedding_dim
        s = self.scores  # (d,)

        # Pairwise absolute differences: |s_i - s_j|
        # Shape: (d, d)
        diff = (s.unsqueeze(0) - s.unsqueeze(1)).abs()  # (d, d)

        # NeuralSort: row i gives soft assignment for the i-th output position
        # We want highest scores first, so negate
        # P_{ij} = softmax_j(-diff_{ij} / tau) doesn't quite work for NeuralSort.
        # Correct NeuralSort: sort scores descending, then for each rank position i,
        # compute how likely each original dim j is at that position.

        # Standard NeuralSort (Grover et al.):
        # A_i = (d + 1 - 2i) * s  for each rank position i
        # P_soft = softmax(A / tau, dim=-1)
        ranks = torch.arange(1, d + 1, device=s.device, dtype=s.dtype)  # (d,)
        # A[i, j] = (d + 1 - 2*i) * s[j]
        A = (d + 1 - 2 * ranks).unsqueeze(1) * s.unsqueeze(0)  # (d, d)
        P_soft = F.softmax(A / self.tau, dim=-1)  # (d, d)

        return P_soft

    def soft_permute(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply soft permutation to embeddings (for training).

        Args:
            embeddings: (B, d) input embeddings

        Returns:
            (B, d) soft-permuted embeddings
        """
        P = self._neuralsort_matrix()  # (d, d)
        # P @ e^T: (d, d) @ (d, B) -> (d, B)
        return (P @ embeddings.T).T  # (B, d)

    def soft_permute_prefix(
        self, embeddings: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """Apply soft permutation and take prefix (for training efficiency).

        Args:
            embeddings: (B, d) input embeddings
            dim: prefix size

        Returns:
            (B, dim) soft-permuted prefix
        """
        P = self._neuralsort_matrix()  # (d, d)
        P_prefix = P[:dim, :]  # (dim, d)
        return (P_prefix @ embeddings.T).T  # (B, dim)

    def hard_permutation(self) -> torch.Tensor:
        """Get the hard permutation (for inference).

        Returns:
            (d,) long tensor of indices: permuted[i] = original[perm[i]]
        """
        return torch.argsort(self.scores, descending=True)

    def hard_permute(self, embeddings: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """Apply hard permutation to embeddings (for inference).

        Args:
            embeddings: (..., d) input embeddings
            dim: optional prefix size (truncate after permutation)

        Returns:
            (..., d) or (..., dim) permuted embeddings
        """
        perm = self.hard_permutation()
        result = embeddings[..., perm]
        if dim is not None:
            result = result[..., :dim]
        return result

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass: soft permutation for training."""
        return self.soft_permute(embeddings)


class SinkhornPermutation(nn.Module):
    """Full d x d Sinkhorn-based permutation.

    Learns a log-score matrix S in R^{d x d}. The soft permutation is:
        P_soft = Sinkhorn(exp(S / tau))

    Optionally adds Gumbel noise for exploration:
        P_soft = Sinkhorn(exp((S + G) / tau))  where G ~ Gumbel(0, 1)

    Parameters: d^2 scalars.
    """

    def __init__(
        self,
        embedding_dim: int,
        tau: float = 1.0,
        n_iters: int = 20,
        use_gumbel: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.log_scores = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
        self.tau = tau
        self.n_iters = n_iters
        self.use_gumbel = use_gumbel

    def set_temperature(self, tau: float):
        self.tau = tau

    def _sinkhorn(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Sinkhorn normalization to produce a doubly-stochastic matrix.

        Args:
            log_alpha: (d, d) log-score matrix

        Returns:
            (d, d) approximately doubly-stochastic matrix
        """
        for _ in range(self.n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha)

    def _soft_permutation_matrix(self) -> torch.Tensor:
        """Compute the soft permutation matrix via Sinkhorn.

        Returns:
            (d, d) doubly-stochastic matrix approximating a permutation
        """
        log_alpha = self.log_scores / self.tau

        if self.use_gumbel and self.training:
            # Add Gumbel noise for exploration
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(log_alpha) + 1e-20) + 1e-20
            )
            log_alpha = log_alpha + gumbel_noise

        return self._sinkhorn(log_alpha)

    def soft_permute(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply soft permutation to embeddings.

        Args:
            embeddings: (B, d) input embeddings

        Returns:
            (B, d) soft-permuted embeddings
        """
        P = self._soft_permutation_matrix()  # (d, d)
        return (P @ embeddings.T).T

    def soft_permute_prefix(
        self, embeddings: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """Apply soft permutation and take prefix.

        Args:
            embeddings: (B, d) input embeddings
            dim: prefix size

        Returns:
            (B, dim) soft-permuted prefix
        """
        P = self._soft_permutation_matrix()
        P_prefix = P[:dim, :]
        return (P_prefix @ embeddings.T).T

    def hard_permutation(self) -> torch.Tensor:
        """Extract hard permutation via Hungarian algorithm.

        Returns:
            (d,) long tensor of permutation indices
        """
        from scipy.optimize import linear_sum_assignment

        P = self._soft_permutation_matrix().detach().cpu().numpy()
        # Hungarian algorithm: maximize assignment (minimize negative)
        row_ind, col_ind = linear_sum_assignment(-P)
        perm = torch.zeros(self.embedding_dim, dtype=torch.long)
        perm[row_ind] = torch.tensor(col_ind, dtype=torch.long)
        return perm

    def hard_permute(self, embeddings: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """Apply hard permutation to embeddings.

        Args:
            embeddings: (..., d) input embeddings
            dim: optional prefix size

        Returns:
            (..., d) or (..., dim) permuted embeddings
        """
        perm = self.hard_permutation().to(embeddings.device)
        result = embeddings[..., perm]
        if dim is not None:
            result = result[..., :dim]
        return result

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.soft_permute(embeddings)


class VarianceSortPermutation:
    """Zero-parameter baseline: sort dimensions by variance (descending).

    High-variance dimensions are placed first, under the hypothesis that
    they carry more discriminative information. No training needed.
    """

    def __init__(self, embedding_dim: Optional[int] = None):
        self.embedding_dim = embedding_dim
        self._permutation = None

    def fit(self, embeddings: torch.Tensor):
        """Compute the variance-sort permutation from a corpus of embeddings.

        Args:
            embeddings: (N, d) corpus embeddings
        """
        self.embedding_dim = embeddings.shape[1]
        variances = embeddings.var(dim=0)  # (d,)
        self._permutation = torch.argsort(variances, descending=True)

    def hard_permutation(self) -> torch.Tensor:
        """Get the permutation.

        Returns:
            (d,) long tensor of permutation indices
        """
        if self._permutation is None:
            raise RuntimeError("Call fit() first with corpus embeddings")
        return self._permutation

    def apply(self, embeddings: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """Apply variance-sort permutation.

        Args:
            embeddings: (..., d) input embeddings
            dim: optional prefix size

        Returns:
            (..., d) or (..., dim) permuted embeddings
        """
        perm = self.hard_permutation().to(embeddings.device)
        result = embeddings[..., perm]
        if dim is not None:
            result = result[..., :dim]
        return result

    def parameters(self):
        """Compatibility: no trainable parameters."""
        return iter([])

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def to(self, device):
        if self._permutation is not None:
            self._permutation = self._permutation.to(device)
        return self

    def state_dict(self):
        return {"permutation": self._permutation}

    def load_state_dict(self, state_dict):
        self._permutation = state_dict["permutation"]
        if self._permutation is not None:
            self.embedding_dim = len(self._permutation)
