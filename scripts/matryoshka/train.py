#!/usr/bin/env python3
"""
CLI entry point for Matryoshka adapter and permutation training.

Usage:
    # Matryoshka-Adaptor (unsupervised)
    python -m scripts.matryoshka.train \\
        --method adaptor \\
        --embeddings_cache data/corpus_embeddings.pkl \\
        --matryoshka_dims 64,128,256,512 \\
        --output_dir output/matryoshka_adaptor

    # Matryoshka-Adaptor (supervised)
    python -m scripts.matryoshka.train \\
        --method adaptor --supervised \\
        --embeddings_cache data/corpus_embeddings.pkl \\
        --query_embeddings_cache data/query_embeddings.pkl \\
        --relevance_file data/qrels.jsonl \\
        --output_dir output/matryoshka_adaptor_sup

    # Score-vector permutation (recommended, unsupervised)
    python -m scripts.matryoshka.train \\
        --method permutation-score \\
        --embeddings_cache data/corpus_embeddings.pkl \\
        --matryoshka_dims 64,128,256,512 \\
        --output_dir output/permutation_score

    # Sinkhorn permutation with Gumbel noise
    python -m scripts.matryoshka.train \\
        --method permutation-sinkhorn \\
        --use_gumbel_noise \\
        --embeddings_cache data/corpus_embeddings.pkl \\
        --output_dir output/permutation_sinkhorn

    # Variance-sort baseline (no training)
    python -m scripts.matryoshka.train \\
        --method variance-sort \\
        --embeddings_cache data/corpus_embeddings.pkl \\
        --output_dir output/variance_sort

    # Compute embeddings first, then train
    python -m scripts.matryoshka.train \\
        --method adaptor \\
        --corpus_file data/corpus.jsonl \\
        --model_name ibm-granite/granite-embedding-30m-english \\
        --text_field text --id_field id \\
        --embeddings_cache data/corpus_embeddings.pkl \\
        --output_dir output/matryoshka_adaptor
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from docuverse.utils.timer import timer
from scripts.matryoshka.config import MatryoshkaTrainingConfig
from scripts.matryoshka.adaptor_trainer import MatryoshkaAdaptorTrainer
from scripts.matryoshka.permutation_trainer import PermutationTrainer


def parse_args() -> MatryoshkaTrainingConfig:
    """Parse command-line arguments into a MatryoshkaTrainingConfig."""
    parser = argparse.ArgumentParser(
        description="Matryoshka adapter and permutation training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  adaptor              MLP adapter with skip connection (Yoon et al., EMNLP 2024)
  permutation-score    Score-vector + NeuralSort, d parameters (recommended)
  permutation-sinkhorn Full d x d Sinkhorn permutation, d^2 parameters
  variance-sort        Zero-parameter baseline: sort by variance
        """,
    )

    # Method
    parser.add_argument(
        "--method",
        type=str,
        default="adaptor",
        choices=["adaptor", "permutation-score", "permutation-sinkhorn", "variance-sort"],
        help="Training method (default: adaptor)",
    )

    # Model
    parser.add_argument("--model_name", type=str, default="ibm-granite/granite-embedding-30m-english")
    parser.add_argument("--embedding_dim", type=int, default=0, help="Embedding dim (0=auto)")

    # Matryoshka dims
    parser.add_argument(
        "--matryoshka_dims",
        type=str,
        default="64,128,256,512",
        help="Comma-separated list of target prefix sizes",
    )

    # Training mode
    parser.add_argument("--supervised", action="store_true", help="Enable supervised mode")

    # Data
    parser.add_argument("--corpus_file", type=str, default="", help="JSONL corpus file")
    parser.add_argument("--query_file", type=str, default="", help="JSONL query file (supervised)")
    parser.add_argument("--embeddings_cache", type=str, default="", help="Pre-computed embeddings pickle")
    parser.add_argument("--query_embeddings_cache", type=str, default="", help="Pre-computed query embeddings")
    parser.add_argument("--relevance_file", type=str, default="", help="Query-corpus relevance JSONL")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--id_field", type=str, default="id")
    parser.add_argument("--query_text_field", type=str, default="text")
    parser.add_argument("--query_id_field", type=str, default="id")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--corpus_batch_size", type=int, default=50000)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--val_fraction", type=float, default=0.1)

    # Loss weights
    parser.add_argument("--alpha", type=float, default=1.0, help="Pairwise loss weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Ranking loss weight")

    # Adaptor-specific
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="512",
        help="Comma-separated MLP hidden layer sizes",
    )

    # Permutation-specific
    parser.add_argument("--tau_start", type=float, default=1.0, help="Initial temperature")
    parser.add_argument("--tau_end", type=float, default=0.05, help="Final temperature")
    parser.add_argument("--sinkhorn_iters", type=int, default=20)
    parser.add_argument("--use_gumbel_noise", action="store_true")

    # Neighbor mining
    parser.add_argument("--topk_neighbors", type=int, default=10)

    # Prefix weighting
    parser.add_argument("--dim_weighting", type=str, default="uniform", choices=["uniform", "log"])

    # Output
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)

    # Device
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Build config
    config = MatryoshkaTrainingConfig(
        method=args.method,
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        matryoshka_dims=[int(d) for d in args.matryoshka_dims.split(",")],
        supervised=args.supervised,
        corpus_file=args.corpus_file,
        query_file=args.query_file,
        embeddings_cache=args.embeddings_cache,
        query_embeddings_cache=args.query_embeddings_cache,
        relevance_file=args.relevance_file,
        text_field=args.text_field,
        id_field=args.id_field,
        query_text_field=args.query_text_field,
        query_id_field=args.query_id_field,
        batch_size=args.batch_size,
        corpus_batch_size=args.corpus_batch_size,
        max_iterations=args.max_iterations,
        patience=args.patience,
        lr=args.lr,
        optimizer=args.optimizer,
        val_fraction=args.val_fraction,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        hidden_dims=[int(d) for d in args.hidden_dims.split(",")],
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        sinkhorn_iters=args.sinkhorn_iters,
        use_gumbel_noise=args.use_gumbel_noise,
        topk_neighbors=args.topk_neighbors,
        dim_weighting=args.dim_weighting,
        output_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
        seed=args.seed,
    )

    return config


def create_trainer(config: MatryoshkaTrainingConfig):
    """Factory: create the appropriate trainer based on config.method."""
    if config.method == "adaptor":
        return MatryoshkaAdaptorTrainer(config)
    elif config.method in ("permutation-score", "permutation-sinkhorn", "variance-sort"):
        return PermutationTrainer(config)
    else:
        raise ValueError(f"Unknown method: {config.method}")


def main():
    tm = timer("matryoshka_train")
    config = parse_args()

    print(f"Matryoshka Training")
    print(f"  Method: {config.method}")
    print(f"  Supervised: {config.supervised}")
    print(f"  Matryoshka dims: {config.matryoshka_dims}")
    print(f"  Output: {config.output_dir}")
    print()

    trainer = create_trainer(config)
    tm.add_timing("setup")

    trainer.train()
    tm.add_timing("train")

    print(f"\nDone. Model saved to {config.output_dir}/best_model.pt")

    total_ms = tm.milliseconds_since_beginning()
    print()
    timer.display_timing(total_ms, keys={"iterations": trainer.global_step})


if __name__ == "__main__":
    main()
