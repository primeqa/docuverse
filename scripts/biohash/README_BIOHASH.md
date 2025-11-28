# BioHash Implementation

Implementation of **"Bio-Inspired Hashing for Unsupervised Similarity Search"** by Ryali et al. (ICML 2020)

Paper: [arXiv:2001.04907v2](https://arxiv.org/abs/2001.04907)

## Overview

BioHash is a locality-sensitive hashing (LSH) algorithm inspired by the fruit fly's olfactory circuit. Unlike classical LSH methods that produce low-dimensional hash codes, BioHash:

- **Produces sparse, high-dimensional hash codes** (m >> d, but only k << m neurons active)
- **Learns from data** using biologically plausible dynamics
- **Uses local Hebbian/anti-Hebbian learning** (no backpropagation)
- **Outperforms classical LSH methods** on similarity search benchmarks

## Key Concepts

### Architecture
- **Input dimension**: d (e.g., 784 for MNIST)
- **Hash layer neurons**: m >> d (sparse expansion)
- **Active neurons**: k << m (sparsity via k-WTA)
- **Activity level**: k/m (typically 0.5-5%)

### Learning Dynamics (Equation 1)

For each neuron μ, the weight update is:

```
τ dW_μi/dt = g[Rank(⟨W_μ, x⟩_μ)] (x_i - ⟨W_μ, x⟩_μ W_μi)
```

where:
- `g[1] = 1` (Hebbian update for winner)
- `g[r] = -Δ` (Anti-Hebbian update for rank r)
- `g[other] = 0` (no update)

**Intuition**: Hidden units are attracted to data peaks and repel each other, tiling the data space adaptively.

### Hash Code Generation (Equation 6)

After training, hash codes are generated via k-Winner-Take-All (k-WTA):

```
y_μ = +1  if ⟨W_μ, x⟩_μ is in top k
y_μ = -1  otherwise
```

## Files

- **`biohash_implementation.py`**: Complete BioHash implementation
  - `BioHash` class with training and hashing
  - `compute_mAP` for evaluation
  - `experiment_mnist` for full MNIST experiments

- **`test_biohash_simple.py`**: Simple tests and examples
  - Synthetic circular data
  - Small MNIST subset
  - Hash property verification

## Installation

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

## Quick Start

### Basic Usage

```python
from biohash_implementation import BioHash
import torch

# Create synthetic data
X_train = torch.randn(1000, 50)  # 1000 samples, 50 dims
X_query = torch.randn(100, 50)

# Initialize BioHash
biohash = BioHash(
    input_dim=50,
    num_neurons=500,  # m = 500 (10x expansion)
    hash_length=25,   # k = 25 (5% activity)
    p=2.0,           # p-norm (2 = spherical K-means)
    delta=0.0,       # Anti-Hebbian weight
    learning_rate=0.02,
    max_epochs=100,
    device='cuda'
)

# Train
biohash.fit(X_train)

# Generate hash codes
hash_train = biohash.hash(X_train)
hash_query = biohash.hash(X_query)

# Compute Hamming distances
distances = torch.cdist(hash_query.float(), hash_train.float(), p=1) / 2
```

### Run Simple Tests

```bash
cd scripts
python test_biohash_simple.py
```

This will run three tests:
1. **Circular data**: Visualizes how neurons tile a circle
2. **MNIST small**: Tests on 1000 MNIST samples
3. **Hash properties**: Verifies sparsity and locality sensitivity

### Run Full MNIST Experiment

```bash
python biohash_implementation.py
```

This reproduces the MNIST results from the paper (Table 1):
- Tests hash lengths k ∈ {2, 4, 8, 16, 32}
- Reports mAP@All scores
- Saves trained models
- Generates t-SNE visualization

Expected results (from paper):
```
k    | mAP@All (%)
-----|------------
2    | 44.38
4    | 49.32
8    | 53.42
16   | 54.92
32   | 55.48
```

## Implementation Details

### Hyperparameters (from paper supplementary)

**Default settings** (used in our implementation):
- `p = 2.0` (spherical K-means variant)
- `Δ = 0.0` (no anti-Hebbian for simplicity)
- `r = 2` (rank for anti-Hebbian)
- Initial learning rate: `ε₀ = 0.02`
- Learning rate decay: `εₜ = ε₀(1 - t/T_max)`
- `T_max = 100` epochs
- Batch size: 100
- Convergence: avg weight norm ≈ 1.06

**Activity levels**:
- MNIST: 5% (k/m = 0.05)
- CIFAR-10: 0.5% (k/m = 0.005)

### Architecture Examples

For MNIST (d=784):
```
k=2:  m=40   (5% activity, 0.05x expansion)
k=4:  m=80   (5% activity, 0.1x expansion)
k=8:  m=160  (5% activity, 0.2x expansion)
k=16: m=320  (5% activity, 0.4x expansion)
k=32: m=640  (5% activity, 0.8x expansion)
```

### Energy Function (Equation 3)

The learning minimizes:

```
E = -Σ_A Σ_μ g[Rank(⟨W_μ, x^A⟩_μ)] ⟨W_μ, x^A⟩_μ / ⟨W_μ, W_μ⟩_μ^((p-1)/p)
```

This energy decreases during training (proven in paper via Cauchy-Schwartz).

### Computational Complexity

**Training**: O(n × m × d) per epoch
- n: number of training samples
- m: number of hash neurons
- d: input dimension

**Hashing**: O(m × d) per query

**Storage**: k log₂(m) bits per database entry (vs k bits for classical LSH)

**Hamming distance**: O(k) per pair (using sparse representation)

## Advanced Usage

### Custom Dataset

```python
import torch
from biohash_implementation import BioHash, compute_mAP

# Load your data
X_train = ...  # (n_train, d)
y_train = ...  # (n_train,) labels for evaluation
X_query = ...  # (n_query, d)
y_query = ...  # (n_query,)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Determine m from desired activity
k = 16
activity = 0.05
m = int(k / activity)

# Train BioHash
biohash = BioHash(
    input_dim=X_train.shape[1],
    num_neurons=m,
    hash_length=k,
    device=device
)

biohash.fit(X_train.to(device))

# Evaluate
hash_train = biohash.hash(X_train.to(device))
hash_query = biohash.hash(X_query.to(device))

mAP = compute_mAP(hash_query, hash_train,
                  y_query.to(device), y_train.to(device),
                  R=1000)  # mAP@1000

print(f"mAP@1000: {mAP:.2f}%")

# Save model
biohash.save('my_biohash.pt')
```

### Hyperparameter Tuning

```python
# Try different p values (sparsity in weights)
biohash_p1 = BioHash(..., p=1.0)  # Sparse weights (L1 norm)
biohash_p2 = BioHash(..., p=2.0)  # Dense weights (L2 norm)

# Try anti-Hebbian updates
biohash_delta = BioHash(..., delta=0.1, r=2)

# Vary activity level
for activity in [0.01, 0.02, 0.05, 0.10]:
    m = int(k / activity)
    biohash = BioHash(..., num_neurons=m, hash_length=k)
    # train and evaluate...
```

## Key Results from Paper

### Table 1: MNIST (mAP@All %)

| Method      | k=2  | k=4  | k=8  | k=16 | k=32 |
|-------------|------|------|------|------|------|
| LSH         | 12.5 | 13.8 | 18.1 | 20.3 | 26.2 |
| FlyHash     | 18.9 | 20.0 | 24.2 | 26.3 | 32.3 |
| ITQ         | 21.9 | 28.5 | 38.4 | 41.2 | 43.6 |
| **BioHash** | **44.4** | **49.3** | **53.4** | **54.9** | **55.5** |

### Table 2: CIFAR-10 (mAP@1000 %)

| Method      | k=2  | k=4  | k=8  | k=16 | k=32 |
|-------------|------|------|------|------|------|
| LSH         | 11.7 | 12.5 | 13.4 | 16.1 | 18.1 |
| FlyHash     | 14.6 | 16.5 | 18.0 | 19.3 | 21.4 |
| ITQ         | 12.6 | 14.5 | 17.1 | 18.7 | 20.6 |
| **BioHash** | **20.5** | **21.6** | **22.6** | **23.4** | **24.0** |

## Biological Interpretation

### Fruit Fly Olfactory Circuit

The algorithm is inspired by *Drosophila*:
- **Projection Neurons** (PNs) ≈ input (d ≈ 50)
- **Kenyon Cells** (KCs) ≈ hash layer (m ≈ 2500, 50x expansion)
- **Sparse activation**: ~50% of PNs, <10% of KCs
- **Random connectivity** in flies, but **learned** in BioHash

### Why Sparse Expansion?

1. **Increases separability**: More dimensions → more decision boundaries
2. **Reduces overlap**: Sparse codes have less collision
3. **Energy efficient**: Metabolic cost ∝ # active neurons (k), not total (m)
4. **Locality sensitive**: Similar inputs → similar sparse codes

### Hebbian Learning

**Hebbian rule**: "Neurons that fire together, wire together"
- Winner neuron: `ΔW ∝ (x - ⟨W,x⟩W)` → W moves toward x
- Anti-Hebbian: `ΔW ∝ -(x - ⟨W,x⟩W)` → W moves away from x
- **Local**: Only uses pre/post-synaptic activity (no global error signal)
- **Online**: Updates after each sample (no batch processing required)

## Comparison with Other Methods

### vs. Classical LSH (SimHash)
- **LSH**: Random projections, dense low-dim codes
- **BioHash**: Learned projections, sparse high-dim codes
- **Advantage**: Better preserves local structure

### vs. FlyHash
- **FlyHash**: Random projections (like biological flies)
- **BioHash**: Learned projections (data-driven)
- **Advantage**: Adapts to data manifold

### vs. Deep Hashing (e.g., GreedyHash)
- **Deep**: Backpropagation, requires labels or pretrained features
- **BioHash**: Local learning, fully unsupervised
- **Advantage**: Faster training, biologically plausible

### vs. SOLHash
- **SOLHash**: Constrained linear program at each step
- **BioHash**: Simple gradient-free dynamics
- **Advantage**: Much faster, scalable to large datasets

## Limitations

1. **Hyperparameter sensitivity**: Activity level affects performance
2. **Storage overhead**: k log₂(m) vs k bits (but still practical)
3. **No GPU acceleration for ranking**: Argmax operations are sequential
4. **Convergence not guaranteed**: Though empirically robust

## Extensions

### BioConvHash (from paper)

For images, learn convolutional filters:
1. Train filters on image patches using Equation (1)
2. Apply cross-channel inhibition (spatial k-WTA)
3. Max-pool
4. Apply BioHash layer

Improves robustness to intensity variations (shadows).

### Possible Improvements

1. **Multi-probe**: Query multiple buckets (vary k at query time)
2. **Hashing ensembles**: Train multiple BioHash models
3. **Quantization**: Compress hash codes further
4. **GPU ranking**: Approximate k-WTA with differentiable operations

## Citation

```bibtex
@inproceedings{ryali2020biohash,
  title={Bio-Inspired Hashing for Unsupervised Similarity Search},
  author={Ryali, Chaitanya K. and Hopfield, John J. and Grinberg, Leopold and Krotov, Dmitry},
  booktitle={International Conference on Machine Learning},
  pages={8295--8306},
  year={2020},
  organization={PMLR}
}
```

## License

This implementation is provided for research and educational purposes.

## Contact

For questions about this implementation, please open an issue in the repository.

For questions about the paper, contact the authors at:
- rckrishn@eng.ucsd.edu
- krotov@ibm.com
