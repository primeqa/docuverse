"""
Loss functions for Matryoshka adapter and permutation training.

Implements the loss functions from:
- Matryoshka-Adaptor (Yoon et al., EMNLP 2024): Eq. 1-6
- Permutation whitepaper: InfoNCE + neighborhood preservation

All losses operate on pre-computed embedding tensors (not raw text).
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity between two sets of embeddings.

    Args:
        a: (N, d) embeddings
        b: (M, d) embeddings

    Returns:
        (N, M) cosine similarity matrix
    """
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    return a_norm @ b_norm.T


# ---------------------------------------------------------------------------
# Unsupervised losses (Matryoshka-Adaptor paper, Eq. 1-3)
# ---------------------------------------------------------------------------

def pairwise_similarity_loss(
    original_embs: torch.Tensor,
    adapted_embs: torch.Tensor,
    matryoshka_dims: List[int],
    dim_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """L_pair (Eq. 1): Preserve pairwise similarity across matryoshka dimensions.

    L_pair = sum_i sum_j sum_m |Sim(ce_i, ce_j) - Sim(f(ce_i)[:m], f(ce_j)[:m])|

    Args:
        original_embs: (B, d) original corpus embeddings
        adapted_embs: (B, d) adapted corpus embeddings (after adapter)
        matryoshka_dims: list of target prefix sizes
        dim_weights: optional per-dim weights

    Returns:
        Scalar loss
    """
    if dim_weights is None:
        dim_weights = [1.0] * len(matryoshka_dims)

    # Full-dim similarity as target
    orig_sim = cosine_similarity_matrix(original_embs, original_embs)

    loss = torch.tensor(0.0, device=original_embs.device)
    for w, m in zip(dim_weights, matryoshka_dims):
        adapted_prefix = adapted_embs[:, :m]
        adapted_sim = cosine_similarity_matrix(adapted_prefix, adapted_prefix)
        loss = loss + w * (orig_sim - adapted_sim).abs().mean()

    return loss


def topk_similarity_loss(
    original_embs: torch.Tensor,
    adapted_embs: torch.Tensor,
    neighbor_indices: torch.Tensor,
    matryoshka_dims: List[int],
    dim_weights: Optional[List[float]] = None,
    original_neighbor_embs: Optional[torch.Tensor] = None,
    adapted_neighbor_embs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_topk (Eq. 2): Preserve similarity for top-k neighbors.

    Like L_pair but restricted to top-k nearest neighbors of each embedding.

    Args:
        original_embs: (B, d) original corpus embeddings
        adapted_embs: (B, d) adapted corpus embeddings
        neighbor_indices: (B, k) indices — used only if neighbor_embs not provided
        matryoshka_dims: list of target prefix sizes
        dim_weights: optional per-dim weights
        original_neighbor_embs: (B, k, d) pre-gathered original neighbor embeddings
        adapted_neighbor_embs: (B, k, d) pre-gathered adapted neighbor embeddings

    Returns:
        Scalar loss
    """
    if dim_weights is None:
        dim_weights = [1.0] * len(matryoshka_dims)

    # Get neighbor embeddings (pre-gathered or index into batch)
    if original_neighbor_embs is None:
        original_neighbor_embs = original_embs[neighbor_indices]  # (B, k, d)
    if adapted_neighbor_embs is None:
        adapted_neighbor_embs = adapted_embs[neighbor_indices]  # (B, k, d)

    # Compute full-dim similarities for neighbor pairs
    orig_norm = F.normalize(original_embs, p=2, dim=-1)
    neighbor_norm = F.normalize(original_neighbor_embs, p=2, dim=-1)
    # (B, k) pairwise similarities
    orig_neighbor_sim = torch.bmm(
        orig_norm.unsqueeze(1), neighbor_norm.transpose(1, 2)
    ).squeeze(1)  # (B, k)

    loss = torch.tensor(0.0, device=original_embs.device)
    for w, m in zip(dim_weights, matryoshka_dims):
        adapted_prefix = adapted_embs[:, :m]
        adapted_norm = F.normalize(adapted_prefix, p=2, dim=-1)
        nn_adapted_prefix = adapted_neighbor_embs[:, :, :m]
        nn_adapted_norm = F.normalize(nn_adapted_prefix, p=2, dim=-1)
        adapted_neighbor_sim = torch.bmm(
            adapted_norm.unsqueeze(1), nn_adapted_norm.transpose(1, 2)
        ).squeeze(1)  # (B, k)
        loss = loss + w * (orig_neighbor_sim - adapted_neighbor_sim).abs().mean()

    return loss


def reconstruction_loss(
    original_embs: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """L_rec (Eq. 3): Regularize adapter output to stay close to original.

    L_rec = sum_i |ce_i - f(ce_i)| = sum_i |residual_i|

    Args:
        original_embs: (B, d) original embeddings (unused, kept for API consistency)
        residual: (B, d) output of the adapter MLP (before skip connection)

    Returns:
        Scalar loss
    """
    return residual.abs().mean()


# ---------------------------------------------------------------------------
# Supervised loss (Matryoshka-Adaptor paper, Eq. 5)
# ---------------------------------------------------------------------------

def ranking_loss(
    query_embs_adapted: torch.Tensor,
    corpus_embs_adapted: torch.Tensor,
    relevance_scores: torch.Tensor,
    matryoshka_dims: List[int],
    dim_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """L_rank (Eq. 5): Matryoshka ranking loss.

    L_rank = sum_i sum_j sum_k sum_m I(y_ij > y_ik) * (y_ij - y_ik)
             * log(1 + exp(s_ik[:m] - s_ij[:m]))

    where s_ij[:m] is cosine similarity between adapted query i and corpus j
    at prefix size m.

    For simplicity, we treat each (query, positive, negative) triplet:
    positive = corpus with relevance > 0, negative = corpus with relevance = 0.

    Args:
        query_embs_adapted: (Q, d) adapted query embeddings
        corpus_embs_adapted: (C, d) adapted corpus embeddings
        relevance_scores: (Q, C) relevance matrix (binary or graded)
        matryoshka_dims: list of target prefix sizes
        dim_weights: optional per-dim weights

    Returns:
        Scalar loss
    """
    if dim_weights is None:
        dim_weights = [1.0] * len(matryoshka_dims)

    loss = torch.tensor(0.0, device=query_embs_adapted.device)
    for w, m in zip(dim_weights, matryoshka_dims):
        q_prefix = F.normalize(query_embs_adapted[:, :m], p=2, dim=-1)
        c_prefix = F.normalize(corpus_embs_adapted[:, :m], p=2, dim=-1)
        sim = q_prefix @ c_prefix.T  # (Q, C)

        # For each query, compute pairwise margin loss between pos and neg
        for qi in range(sim.shape[0]):
            pos_mask = relevance_scores[qi] > 0
            neg_mask = relevance_scores[qi] == 0
            if not pos_mask.any() or not neg_mask.any():
                continue
            pos_sims = sim[qi][pos_mask]  # (num_pos,)
            neg_sims = sim[qi][neg_mask]  # (num_neg,)
            # Broadcast: (num_pos, 1) vs (1, num_neg) -> (num_pos, num_neg)
            margin = neg_sims.unsqueeze(0) - pos_sims.unsqueeze(1)
            loss = loss + w * torch.log1p(torch.exp(margin)).mean()

    return loss


def ranking_loss_triplet(
    query_embs_adapted: torch.Tensor,
    pos_embs_adapted: torch.Tensor,
    neg_embs_adapted: torch.Tensor,
    matryoshka_dims: List[int],
    dim_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Triplet-style ranking loss across matryoshka dims.

    Simpler variant where each sample is a (query, positive, negative) triplet.

    Args:
        query_embs_adapted: (B, d) adapted query embeddings
        pos_embs_adapted: (B, d) adapted positive corpus embeddings
        neg_embs_adapted: (B, d) adapted negative corpus embeddings
        matryoshka_dims: list of target prefix sizes
        dim_weights: optional per-dim weights

    Returns:
        Scalar loss
    """
    if dim_weights is None:
        dim_weights = [1.0] * len(matryoshka_dims)

    loss = torch.tensor(0.0, device=query_embs_adapted.device)
    for w, m in zip(dim_weights, matryoshka_dims):
        q = F.normalize(query_embs_adapted[:, :m], p=2, dim=-1)
        p = F.normalize(pos_embs_adapted[:, :m], p=2, dim=-1)
        n = F.normalize(neg_embs_adapted[:, :m], p=2, dim=-1)
        pos_sim = (q * p).sum(dim=-1)  # (B,)
        neg_sim = (q * n).sum(dim=-1)  # (B,)
        loss = loss + w * torch.log1p(torch.exp(neg_sim - pos_sim)).mean()

    return loss


# ---------------------------------------------------------------------------
# Permutation-specific losses (whitepaper)
# ---------------------------------------------------------------------------

def infonce_loss(
    query_embs: torch.Tensor,
    pos_embs: torch.Tensor,
    neg_embs: torch.Tensor,
    matryoshka_dims: List[int],
    temperature: float = 0.05,
    dim_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """InfoNCE contrastive loss at each matryoshka prefix size.

    For permutation training: the embeddings are already soft-permuted.

    Args:
        query_embs: (B, d) query embeddings (already permuted)
        pos_embs: (B, d) positive embeddings (already permuted)
        neg_embs: (B, N, d) negative embeddings (already permuted)
        matryoshka_dims: list of target prefix sizes
        temperature: InfoNCE temperature
        dim_weights: optional per-dim weights

    Returns:
        Scalar loss
    """
    if dim_weights is None:
        dim_weights = [1.0] * len(matryoshka_dims)

    loss = torch.tensor(0.0, device=query_embs.device)
    for w, m in zip(dim_weights, matryoshka_dims):
        q = F.normalize(query_embs[:, :m], p=2, dim=-1)  # (B, m)
        p = F.normalize(pos_embs[:, :m], p=2, dim=-1)  # (B, m)

        # Positive similarity
        pos_sim = (q * p).sum(dim=-1, keepdim=True) / temperature  # (B, 1)

        if neg_embs.dim() == 3:
            n = F.normalize(neg_embs[:, :, :m], p=2, dim=-1)  # (B, N, m)
            neg_sim = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1) / temperature  # (B, N)
        else:
            # neg_embs is (N, d) — shared negatives
            n = F.normalize(neg_embs[:, :m], p=2, dim=-1)  # (N, m)
            neg_sim = (q @ n.T) / temperature  # (B, N)

        # Concat pos and neg: (B, 1+N)
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = loss + w * F.cross_entropy(logits, labels)

    return loss


def neighborhood_preservation_loss(
    original_embs: torch.Tensor,
    permuted_embs: torch.Tensor,
    neighbor_indices: torch.Tensor,
    matryoshka_dims: List[int],
    temperature: float = 0.05,
    dim_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Unsupervised loss for permutation: preserve neighborhood structure.

    Uses the full-dim top-k neighbors as pseudo-positive pairs. For each
    embedding e_i and its neighbor e_j, maximize similarity of their
    permuted prefixes while contrasting against non-neighbors.

    Args:
        original_embs: (B, d) original embeddings (for computing target similarities)
        permuted_embs: (B, d) soft-permuted embeddings
        neighbor_indices: (B, k) top-k neighbor indices
        matryoshka_dims: list of target prefix sizes
        temperature: contrastive temperature
        dim_weights: optional per-dim weights

    Returns:
        Scalar loss
    """
    if dim_weights is None:
        dim_weights = [1.0] * len(matryoshka_dims)

    B, k = neighbor_indices.shape

    # Full-dim similarities as targets
    orig_norm = F.normalize(original_embs, p=2, dim=-1)
    orig_sim = orig_norm @ orig_norm.T  # (B, B)

    loss = torch.tensor(0.0, device=original_embs.device)
    for w, m in zip(dim_weights, matryoshka_dims):
        prefix = permuted_embs[:, :m]
        prefix_norm = F.normalize(prefix, p=2, dim=-1)
        prefix_sim = prefix_norm @ prefix_norm.T  # (B, B)

        # For each point, push prefix sim toward original sim for neighbors
        for i in range(B):
            nn_idx = neighbor_indices[i]
            target_sim = orig_sim[i, nn_idx]  # (k,)
            pred_sim = prefix_sim[i, nn_idx]  # (k,)
            loss = loss + w * (target_sim - pred_sim).abs().mean()

    return loss / B
