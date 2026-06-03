"""Configuration for Matryoshka adapter and permutation training."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MatryoshkaTrainingConfig:
    """Configuration for Matryoshka training methods.

    Supports both the Matryoshka-Adaptor (MLP with skip connection) and
    permutation-based (NeuralSort/Sinkhorn) approaches.
    """

    # --- Model ---
    model_name: str = "ibm-granite/granite-embedding-30m-english"
    embedding_dim: int = 0  # Auto-detected from embeddings if 0

    # --- Method selection ---
    # "adaptor": MLP adapter with skip connection (Yoon et al.)
    # "permutation-score": Score-vector + NeuralSort (d params, recommended)
    # "permutation-sinkhorn": Full d x d Sinkhorn matrix (d^2 params)
    # "variance-sort": Sort dims by variance, zero parameters (baseline)
    method: str = "adaptor"

    # --- Matryoshka dimensions ---
    matryoshka_dims: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512]
    )

    # --- Training mode ---
    supervised: bool = False  # If True, use query-corpus pairs + ranking loss

    # --- Data paths ---
    corpus_file: str = ""  # JSONL file with corpus texts
    query_file: str = ""  # JSONL file with queries (supervised mode)
    embeddings_cache: str = ""  # Pre-computed embeddings pickle file
    query_embeddings_cache: str = ""  # Pre-computed query embeddings pickle
    relevance_file: str = ""  # Query-corpus relevance pairs (supervised)
    text_field: str = "text"
    id_field: str = "id"
    query_text_field: str = "text"
    query_id_field: str = "id"

    # --- Training hyperparameters (paper Table 5) ---
    batch_size: int = 128
    corpus_batch_size: int = 50000  # For sampling corpus pairs each iteration
    max_iterations: int = 5000
    patience: int = 500  # Early stopping patience
    lr: float = 0.001
    optimizer: str = "adam"
    val_fraction: float = 0.1  # Fraction of corpus for validation

    # --- Loss weights (paper: all 1.0) ---
    alpha: float = 1.0  # Pairwise similarity loss weight
    beta: float = 1.0  # Reconstruction loss weight
    gamma: float = 1.0  # Ranking loss weight

    # --- Adaptor-specific ---
    hidden_dims: List[int] = field(
        default_factory=lambda: [512]
    )

    # --- Permutation-specific ---
    tau_start: float = 1.0  # Initial temperature for NeuralSort/Sinkhorn
    tau_end: float = 0.05  # Final temperature
    sinkhorn_iters: int = 20  # Number of Sinkhorn normalization iterations
    use_gumbel_noise: bool = False  # Add Gumbel noise before Sinkhorn

    # --- Neighbor mining (unsupervised) ---
    topk_neighbors: int = 10  # k for top-k neighbor mining

    # --- Prefix-size weighting ---
    # "uniform": all prefix sizes weighted equally
    # "log": w_k = 1 / log2(k), normalized
    dim_weighting: str = "uniform"

    # --- Output ---
    output_dir: str = "output"
    save_every: int = 500
    log_every: int = 50

    # --- Device ---
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"

    # --- Misc ---
    seed: int = 42
    num_workers: int = 0  # DataLoader workers

    def __post_init__(self):
        # Validate method
        valid_methods = {"adaptor", "permutation-score", "permutation-sinkhorn", "variance-sort"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{self.method}'")

        # Sort matryoshka dims
        self.matryoshka_dims = sorted(self.matryoshka_dims)

        # Validate supervised mode has required data
        if self.supervised and not self.query_file and not self.query_embeddings_cache:
            raise ValueError("supervised=True requires query_file or query_embeddings_cache")

    def get_dim_weights(self) -> List[float]:
        """Get per-prefix-size weights for the multi-scale loss."""
        import math
        dims = self.matryoshka_dims
        if self.dim_weighting == "uniform":
            return [1.0] * len(dims)
        elif self.dim_weighting == "log":
            raw = [1.0 / math.log2(max(d, 2)) for d in dims]
            total = sum(raw)
            return [w / total * len(dims) for w in raw]
        else:
            raise ValueError(f"Unknown dim_weighting: {self.dim_weighting}")
