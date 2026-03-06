"""
Matryoshka adapter and permutation training for post-hoc dimensionality reduction.

Two approaches for improving Matryoshka properties of frozen embeddings:

1. MatryoshkaAdaptorTrainer: Learns an MLP adapter f: R^d -> R^d with skip connection.
   Based on "Matryoshka-Adaptor" (Yoon et al., EMNLP 2024).

2. PermutationTrainer: Learns a permutation of embedding dimensions via differentiable
   relaxation (NeuralSort/Sinkhorn). Backward-compatible: full-dim similarity is preserved.
   Based on the post-hoc permutation whitepaper.
"""

from .config import MatryoshkaTrainingConfig
from .adaptor_trainer import MatryoshkaAdaptorTrainer
from .permutation_trainer import PermutationTrainer
from .st_module import MatryoshkaAdaptorModule, export_to_sentence_transformer  # noqa: F401
