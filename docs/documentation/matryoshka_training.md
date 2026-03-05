# Matryoshka Training Module

Post-hoc training methods for improving Matryoshka properties of frozen embedding
models. Located in `scripts/matryoshka/`.

## Overview

Two complementary approaches for enabling efficient prefix-based dimensionality
reduction on embeddings from any pre-trained model:

**1. Matryoshka-Adaptor** (Yoon et al., EMNLP 2024)
Learns an MLP adapter `f: R^d -> R^d` with skip connection. The adapted embedding
is `e' = e + f(e)`. The adapter is trained so that prefix truncation `e'[:m]`
preserves retrieval quality across target dimensions `m`.

**2. Permutation-based approach** (post-hoc permutation whitepaper)
Learns a permutation of embedding dimensions so that the reordered prefix at each
target size yields good retrieval. Key advantage: **backward-compatible** -- full-dim
cosine similarity is exactly preserved, so existing vector indexes can be updated
in-place without re-indexing.

Both approaches work on **frozen embeddings** (the base model is never modified) and
support **unsupervised** (corpus-only) and **supervised** (query-corpus pairs) training.

## Quick Start

```bash
# Unsupervised Matryoshka-Adaptor
python -m scripts.matryoshka.train \
    --method adaptor \
    --embeddings_cache data/corpus_embeddings.pkl \
    --matryoshka_dims 64,128,256,512 \
    --output_dir output/adaptor

# Score-vector permutation (recommended permutation method)
python -m scripts.matryoshka.train \
    --method permutation-score \
    --embeddings_cache data/corpus_embeddings.pkl \
    --matryoshka_dims 64,128,256,512 \
    --output_dir output/permutation

# Variance-sort baseline (no training, instant)
python -m scripts.matryoshka.train \
    --method variance-sort \
    --embeddings_cache data/corpus_embeddings.pkl \
    --output_dir output/variance_sort

# Supervised Matryoshka-Adaptor (two-stage: unsupervised then supervised)
python -m scripts.matryoshka.train \
    --method adaptor --supervised \
    --embeddings_cache data/corpus_embeddings.pkl \
    --query_embeddings_cache data/query_embeddings.pkl \
    --relevance_file data/qrels.jsonl \
    --output_dir output/adaptor_supervised

# Compute embeddings on-the-fly (no pre-cached embeddings)
python -m scripts.matryoshka.train \
    --method adaptor \
    --corpus_file data/corpus.jsonl \
    --model_name ibm-granite/granite-embedding-30m-english \
    --text_field text --id_field id \
    --embeddings_cache data/corpus_embeddings.pkl \
    --output_dir output/adaptor
```

## File Structure

```
scripts/matryoshka/
    __init__.py                 # Package exports
    config.py                   # MatryoshkaTrainingConfig dataclass
    losses.py                   # All loss functions
    models.py                   # AdaptorMLP, ScoreVectorPermutation,
                                #   SinkhornPermutation, VarianceSortPermutation
    data.py                     # EmbeddingDataset: loading, caching, neighbor mining
    base_trainer.py             # BaseMatryoshkaTrainer: shared training loop
    adaptor_trainer.py          # MatryoshkaAdaptorTrainer
    permutation_trainer.py      # PermutationTrainer
    train.py                    # CLI entry point
```

## Class Hierarchy

```
BaseMatryoshkaTrainer (base_trainer.py)
    Abstract base with shared training loop, validation, evaluation,
    early stopping, and checkpointing.

    |-- MatryoshkaAdaptorTrainer (adaptor_trainer.py)
    |     model: AdaptorMLP
    |     losses: L_topk + alpha*L_pair + beta*L_rec [+ gamma*L_rank]
    |     Two-stage training for supervised mode
    |
    |-- PermutationTrainer (permutation_trainer.py)
          model: ScoreVectorPermutation   (d parameters, recommended)
               | SinkhornPermutation      (d^2 parameters)
               | VarianceSortPermutation  (0 parameters, baseline)
          Cosine temperature annealing for differentiable relaxation
          Inner product preservation verification
```

## Methods in Detail

### Matryoshka-Adaptor

**Architecture**: Shallow MLP with skip connection.

```
input e (d dims) --> MLP f(e) --> residual
                                    |
                        adapted = e + f(e)
```

The MLP is `Linear(d, h) -> ReLU -> Linear(h, d)` with the final layer initialized
near zero so the adapter starts as an identity function.

**Unsupervised loss** (Eq. 4 from paper):
```
min_f  L_topk(f) + alpha * L_pair(f) + beta * L_rec(f)
```

| Loss | Description | Equation |
|------|-------------|----------|
| `L_topk` | Preserve cosine similarity between each embedding and its top-k neighbors, across all prefix sizes m | `sum_i sum_{j in NN_k(i)} sum_m \|Sim(e_i, e_j) - Sim(e'_i[:m], e'_j[:m])\|` |
| `L_pair` | Preserve pairwise cosine similarity for random pairs across prefix sizes | `sum_i sum_j sum_m \|Sim(e_i, e_j) - Sim(e'_i[:m], e'_j[:m])\|` |
| `L_rec` | Reconstruction regularizer: keep adapter residual small | `sum_i \|f(e_i)\|` (L1 norm) |

**Supervised loss** (Eq. 6, adds ranking loss):
```
min_f  L_topk(f) + alpha * L_pair(f) + beta * L_rec(f) + gamma * L_rank(f)
```

| Loss | Description |
|------|-------------|
| `L_rank` | Pairwise margin ranking loss across query-corpus pairs and prefix sizes: `log(1 + exp(s_neg[:m] - s_pos[:m]))` |

**Two-stage training**: When supervised, the adaptor is first trained with the
unsupervised objective (stage 1), then fine-tuned with all four losses (stage 2)
at a reduced learning rate (0.1x).

### Score-Vector Permutation (NeuralSort)

**Architecture**: Learns a single score vector `s in R^d` where `s[i]` represents
the importance of original dimension `i`. The permutation is `argsort(s, descending)`.

**Training**: Uses the NeuralSort differentiable relaxation to produce a soft
permutation matrix:

```
P_soft(s; tau) = softmax((d+1 - 2*rank) * s / tau,  dim=-1)
```

As temperature `tau -> 0`, `P_soft` approaches a hard permutation matrix.

**Temperature annealing**: Cosine schedule from `tau_start` (default 1.0) to
`tau_end` (default 0.05) over training.

**Inference**: Simply reorder dimensions by `argsort(-s)`, then prefix-truncate.
This is O(d) index selection -- no matrix multiplication needed.

**Parameters**: Exactly `d` scalars (e.g., 768 for a 768-dim model).

**Key property**: Full-dim cosine similarity is exactly preserved. Existing vector
indexes can be updated by reordering stored vectors in-place.

### Sinkhorn Permutation

**Architecture**: Learns a log-score matrix `S in R^{d x d}`. The soft permutation
is produced by the Sinkhorn operator (alternating row/column normalization):

```
P_soft = Sinkhorn(exp(S / tau))
```

Optional **Gumbel noise** adds exploration during training:
```
P_soft = Sinkhorn(exp((S + G) / tau))   where G ~ Gumbel(0, 1)
```

**Hardening**: After training, the hard permutation is extracted via the Hungarian
algorithm (`scipy.optimize.linear_sum_assignment`).

**Parameters**: `d^2` scalars (e.g., ~590K for d=768). More expressive than the
score-vector approach but adds regularization complexity.

### Variance-Sort Permutation

**Architecture**: Zero-parameter baseline. Sorts dimensions by their variance across
the corpus (descending). High-variance dimensions are placed first.

**No training needed** -- computed from a single pass over corpus embeddings.

## Configuration Reference

`MatryoshkaTrainingConfig` (in `config.py`) is a dataclass with all hyperparameters:

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"adaptor"` | `"adaptor"`, `"permutation-score"`, `"permutation-sinkhorn"`, `"variance-sort"` |
| `model_name` | `"ibm-granite/granite-embedding-30m-english"` | SentenceTransformer model (for computing embeddings) |
| `embedding_dim` | `0` | Auto-detected from embeddings if 0 |
| `matryoshka_dims` | `[64, 128, 256, 512]` | Target prefix sizes |
| `supervised` | `False` | Enable supervised mode with query-corpus pairs |

### Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `corpus_file` | `""` | JSONL corpus file (text input) |
| `embeddings_cache` | `""` | Pre-computed embeddings pickle file |
| `query_file` | `""` | JSONL query file (supervised mode) |
| `query_embeddings_cache` | `""` | Pre-computed query embeddings pickle |
| `relevance_file` | `""` | JSONL with query-corpus relevance pairs |
| `text_field` | `"text"` | Field name in JSONL for text |
| `id_field` | `"id"` | Field name in JSONL for document ID |

### Training Hyperparameters

From the paper (Table 5), used identically for all experiments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `128` | Training batch size |
| `corpus_batch_size` | `50000` | Corpus sample size for pairwise computation |
| `max_iterations` | `5000` | Maximum training iterations |
| `patience` | `500` | Early stopping patience |
| `lr` | `0.001` | Learning rate |
| `optimizer` | `"adam"` | `"adam"` or `"sgd"` |
| `val_fraction` | `0.1` | Fraction of corpus held out for validation |

### Loss Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `1.0` | Pairwise similarity loss weight |
| `beta` | `1.0` | Reconstruction loss weight |
| `gamma` | `1.0` | Ranking loss weight (supervised) |
| `dim_weighting` | `"uniform"` | `"uniform"` or `"log"` (1/log2(k), up-weights small dims) |

### Adaptor-Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[512]` | MLP hidden layer sizes |

### Permutation-Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_start` | `1.0` | Initial NeuralSort/Sinkhorn temperature |
| `tau_end` | `0.05` | Final temperature |
| `sinkhorn_iters` | `20` | Sinkhorn normalization iterations |
| `use_gumbel_noise` | `False` | Add Gumbel noise before Sinkhorn |
| `topk_neighbors` | `10` | k for top-k neighbor mining (unsupervised) |

## Data Formats

### Embeddings Cache (pickle)

The primary input is a pickle file with pre-computed embeddings:

```python
{
    "embeddings": np.ndarray,  # shape (N, d), float32, L2-normalized
    "ids": List[str],          # document IDs, length N
}
```

Can be created with `scripts/compute_embeddings_from_jsonl.py`:
```bash
python scripts/compute_embeddings_from_jsonl.py \
    corpus.jsonl corpus_embeddings.pkl \
    --model ibm-granite/granite-embedding-30m-english \
    --text_field text --id_field id
```

### Relevance File (JSONL, supervised mode)

One JSON object per line with query-corpus relevance:
```json
{"query_id": "q1", "corpus_id": "d42", "relevance": 1.0}
{"query_id": "q1", "corpus_id": "d99", "relevance": 0.0}
```

Fields can alternatively be named `qid`/`docid`/`pid`/`score`.

## Output

Training saves to `output_dir/`:

| File | Contents |
|------|----------|
| `best_model.pt` | Best checkpoint (model state dict, optimizer state, config, training step, best validation loss) |

### Loading a Trained Model

```python
from scripts.matryoshka.config import MatryoshkaTrainingConfig
from scripts.matryoshka.train import create_trainer

config = MatryoshkaTrainingConfig(
    method="adaptor",
    embedding_dim=768,
    matryoshka_dims=[64, 128, 256, 512],
    device="cpu",
)
trainer = create_trainer(config)

# Load embeddings to initialize model dimensions
trainer.dataset.load_corpus_embeddings()
trainer._init_model_and_optimizer()

# Load trained weights
trainer.load("output/adaptor/best_model.pt")

# Apply to new embeddings
import numpy as np
embeddings = np.random.randn(100, 768).astype(np.float32)
adapted_256 = trainer.apply_to_embeddings(embeddings, dim=256)
```

### Permutation Inference

For permutation methods, inference is a simple index selection:

```python
trainer.load("output/permutation/best_model.pt")

# Get the hard permutation (d-length index array)
perm = trainer.model.hard_permutation()

# Apply to any embeddings: just reorder dimensions
permuted = embeddings[:, perm.numpy()]
truncated_128 = permuted[:, :128]

# Verify: full-dim cosine similarity is unchanged
max_diff = trainer.verify_inner_product_preservation(embeddings)
# max_diff should be ~1e-7 (floating point precision)
```

## Evaluation

During and after training, the module evaluates two distance metrics at each
matryoshka dimension (inspired by Fig. 7 in the paper):

| Metric | Description |
|--------|-------------|
| **Pairwise Distance** | Average absolute difference between full-dim and prefix-dim cosine similarities across all pairs |
| **Top-k Distance** | Same metric but restricted to top-k nearest neighbors |

Lower values indicate better Matryoshka properties (prefix similarity tracks
full-dim similarity more closely).

Example output:
```
Evaluation at matryoshka dimensions:
   Dim   Pairwise Dist    Top-k Dist
------  --------------  ------------
    64          0.0812        0.0798
   128          0.0412        0.0395
   256          0.0198        0.0187
   512          0.0051        0.0048
768 (full)      0.0000        0.0000
```

## Method Comparison

| Property | Adaptor | Score-Vector Perm | Sinkhorn Perm | Variance-Sort |
|----------|---------|-------------------|---------------|---------------|
| Parameters | d * h + h * d | d | d^2 | 0 |
| Index compatible | No | Yes | Yes | Yes |
| Interpretable | No | Yes (dim importance) | No | Yes |
| Training cost | Moderate | Low | Moderate | None |
| Unsupervised | Yes | Yes | Yes | Yes |
| Supervised | Yes | Yes | Yes | No |

**When to use which**:
- **Adaptor**: Best retrieval quality, especially supervised. Use when you can
  re-index the corpus.
- **Score-vector permutation**: Recommended when backward compatibility with
  existing indexes is required. Only `d` parameters, fast training.
- **Sinkhorn permutation**: More expressive permutation learning, useful as an
  ablation. Higher risk of overfitting.
- **Variance-sort**: Free baseline. Start here to establish a lower bound before
  investing in learned methods.

## References

- Yoon, J., Sinha, R., Arik, S. O., & Pfister, T. (2024). *Matryoshka-Adaptor:
  Unsupervised and Supervised Tuning for Smaller Embedding Dimensions*. EMNLP.
  `docs/matryoshka_adapter.pdf`

- *Post-Hoc Embedding Permutation for Matryoshka-Style Embeddings* (IBM Research
  whitepaper). `docs/whitepaper_matryoshka_permutation.docx`

- Kusupati, A., et al. (2022). *Matryoshka Representation Learning*. NeurIPS.

- Grover, A., et al. (2019). *Stochastic Optimization of Sorting Networks via
  Continuous Relaxations (NeuralSort)*. ICML.

- Mena, G., et al. (2018). *Learning Latent Permutations with Gumbel-Sinkhorn
  Networks*. ICLR.
