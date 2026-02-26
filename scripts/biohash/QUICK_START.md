# BioHash Quick Start Guide

## Installation

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm
cd /home/raduf/sandbox2/docuverse/scripts
```

## Run Tests (Recommended Order)

### 1. Validation (30 seconds)
```bash
python validate_biohash.py
```
✓ Verifies implementation correctness

### 2. Simple Tests (1-2 minutes)
```bash
python test_biohash_simple.py
```
✓ Circular data visualization
✓ Small MNIST test
✓ Hash properties check

### 3. Full MNIST Experiment (15-30 minutes on CPU)
```bash
python biohash_implementation.py
```
✓ Reproduces paper results
✓ Tests k ∈ {2, 4, 8, 16, 32}
✓ Saves models and visualizations

## Minimal Example

```python
from biohash_implementation import BioHash
import torch

# Generate data
X_train = torch.randn(1000, 50)  # 1000 samples, 50 dims

# Create and train BioHash
biohash = BioHash(
    input_dim=50,
    num_neurons=320,  # m (expansion)
    hash_length=16,   # k (sparsity)
    device='cuda'     # or 'cpu'
)

biohash.fit(X_train.cuda())

# Generate hash codes
hash_codes = biohash.hash(X_train.cuda())
print(hash_codes.shape)  # (1000, 320)
print((hash_codes == 1).sum(dim=1).float().mean())  # Should be 16

# Save
biohash.save('my_biohash.pt')
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | - | Data dimensionality (d) |
| `num_neurons` | - | Hash layer size (m >> d) |
| `hash_length` | - | Active neurons (k << m) |
| `p` | 2.0 | p-norm (2 = spherical K-means) |
| `delta` | 0.0 | Anti-Hebbian weight (Δ) |
| `learning_rate` | 0.02 | Initial LR (decays to 0) |
| `max_epochs` | 100 | Training epochs |

**Rule of thumb**: `m = k / activity`, where activity = 0.05 (5%) for MNIST

## Expected Results (MNIST)

| k | mAP@All (%) |
|---|-------------|
| 2 | ~44 |
| 4 | ~49 |
| 8 | ~53 |
| 16 | ~55 |
| 32 | ~56 |

## Files

- `biohash_implementation.py` - Main implementation
- `test_biohash_simple.py` - Quick tests
- `validate_biohash.py` - Unit tests
- `README_BIOHASH.md` - Full documentation
- `BIOHASH_SUMMARY.md` - Algorithm details

## Algorithm at a Glance

```
1. Initialize: W ∈ ℝ^(m×d) ~ N(0,1)

2. For each training sample x:
   - Compute activations: a = Wx
   - Find winner: μ* = argmax(a)
   - Update: W[μ*] += lr · (x - a[μ*]·W[μ*])

3. Hash: y = k-WTA(Wx) ∈ {-1,+1}^m
```

## Paper Reference

**"Bio-Inspired Hashing for Unsupervised Similarity Search"**
Ryali et al., ICML 2020

PDF: `/home/raduf/sandbox2/docuverse/writeups/2001.04907v2.pdf`

## Troubleshooting

**Issue**: Low mAP scores
→ Adjust activity level (try 1-10%)

**Issue**: Slow training
→ Use GPU (`device='cuda'`)

**Issue**: Out of memory
→ Reduce `num_neurons` or `batch_size`

## Next Steps

1. ✅ Install dependencies
2. ✅ Run `validate_biohash.py`
3. ✅ Try `test_biohash_simple.py`
4. ✅ Experiment with `biohash_implementation.py`
5. ✅ Read `README_BIOHASH.md` for details

Happy hashing! 🧠
