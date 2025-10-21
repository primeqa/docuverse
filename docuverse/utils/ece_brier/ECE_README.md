# Expected Calibration Error (ECE) Calculator

A comprehensive Python implementation for computing Expected Calibration Error and related calibration metrics to evaluate the quality of output probabilities from text encoding rerankers.

## What is Expected Calibration Error (ECE)?

ECE measures how well predicted probabilities align with actual outcomes by:
1. **Binning** predictions by confidence level
2. **Computing accuracy** within each bin
3. **Measuring** the weighted average difference between confidence and accuracy

```
ECE = Σ (n_k/N) × |accuracy_k - confidence_k|
```

Where:
- `n_k` = number of samples in bin k
- `N` = total number of samples
- `accuracy_k` = observed frequency in bin k
- `confidence_k` = average predicted probability in bin k

**Key Properties:**
- **Range**: 0 (perfect calibration) to 1 (worst)
- **Lower is better**
- More interpretable than Brier score for understanding miscalibration

## Quick Start

```python
from ece import expected_calibration_error

# Your reranker predictions (probabilities)
predictions = [0.9, 0.8, 0.3, 0.1, 0.7, 0.95, 0.2, 0.85, 0.15, 0.6]

# Ground truth relevance labels (0 or 1)
actuals = [1, 1, 0, 0, 1, 1, 0, 1, 0, 1]

# Calculate ECE
ece = expected_calibration_error(predictions, actuals, n_bins=10)
print(f"ECE: {ece:.4f}")
```

## Core Functions

### `expected_calibration_error(predictions, actuals, n_bins=10, strategy='uniform')`

Calculates the Expected Calibration Error.

**Parameters:**
- `predictions`: List/array of predicted probabilities (0 to 1)
- `actuals`: List/array of binary outcomes (0 or 1)
- `n_bins`: Number of bins (default: 10)
- `strategy`: 'uniform' (equal width bins) or 'quantile' (equal frequency bins)

**Returns:** Float (0 to 1, lower is better)

**Example:**
```python
ece_uniform = expected_calibration_error(preds, labels, n_bins=10, strategy='uniform')
ece_quantile = expected_calibration_error(preds, labels, n_bins=10, strategy='quantile')
```

### `maximum_calibration_error(predictions, actuals, n_bins=10, strategy='uniform')`

Calculates the Maximum Calibration Error - the largest gap between confidence and accuracy in any bin.

**Why use MCE?**
- More pessimistic than ECE
- Identifies worst-case miscalibration
- Useful for safety-critical applications

**Returns:** Float (0 to 1, lower is better)

### `adaptive_calibration_error(predictions, actuals, n_bins=10)`

Calculates ACE using adaptive (quantile) binning. Equivalent to ECE with `strategy='quantile'`.

**When to use:**
- Predictions have skewed distributions
- Many predictions clustered at extremes (0 or 1)
- Want equal number of samples per bin

### `calibration_curve(predictions, actuals, n_bins=10, strategy='uniform')`

Computes data for plotting reliability diagrams.

**Returns:** Tuple of (mean_predicted_probs, fraction_positives, bin_counts)

### `plot_reliability_diagram(predictions, actuals, n_bins=10, strategy='uniform', save_path=None, show_histogram=True)`

Creates a reliability diagram (calibration plot) visualizing how well probabilities are calibrated.

**Features:**
- Diagonal line shows perfect calibration
- Deviation from diagonal indicates miscalibration
- Gap lines show calibration error for each bin
- Optional histogram of prediction distribution
- Displays ECE and MCE in title

**Returns:** matplotlib Figure object

**Example:**
```python
from ece import plot_reliability_diagram

fig = plot_reliability_diagram(
    predictions, 
    actuals,
    n_bins=10,
    strategy='uniform',
    save_path='my_reliability_plot.png',
    show_histogram=True
)
```

### `classwise_ece(predictions, actuals, n_bins=10)`

Calculates ECE for multi-class predictions (e.g., graded relevance: not relevant, relevant, highly relevant).

**Parameters:**
- `predictions`: 2D array of shape (N, K) where K is number of classes
- `actuals`: Array of class labels (0 to K-1)

**Returns:** Tuple of (overall_ece, list_of_class_eces)

## Interpreting Results

### Calibration Quality Guidelines

| ECE Range | Calibration Quality | Interpretation |
|-----------|-------------------|----------------|
| < 0.05 | Excellent | Near-perfect calibration |
| 0.05 - 0.10 | Good | Acceptable for most applications |
| 0.10 - 0.15 | Moderate | Some miscalibration, consider recalibration |
| > 0.15 | Poor | Significant miscalibration issues |

### Common Patterns

**Overconfident Model (ECE > 0.15, predictions > actuals)**
- Predicts high probabilities but accuracy is lower
- Example: Predicts 0.9 confidence but only 60% accurate
- Solution: Temperature scaling, Platt scaling

**Underconfident Model (ECE > 0.15, predictions < actuals)**
- Predicts low probabilities but accuracy is higher
- Example: Predicts 0.5 confidence but actually 85% accurate
- Less common, usually not problematic

**Well-Calibrated Model (ECE < 0.05)**
- Predicted probabilities match observed frequencies
- When model says 70%, it's correct ~70% of the time
- Ideal for downstream decision-making

## Use Case: Evaluating a Reranker

```python
import numpy as np
from ece import (expected_calibration_error, maximum_calibration_error,
                 plot_reliability_diagram)

# Your reranker outputs
reranker_predictions = [...]  # Probabilities for 1000 query-doc pairs
ground_truth_labels = [...]    # Binary relevance labels

# Calculate ECE
ece = expected_calibration_error(reranker_predictions, ground_truth_labels, 
                                 n_bins=10, strategy='uniform')
print(f"ECE: {ece:.4f}")

# Check worst-case miscalibration
mce = maximum_calibration_error(reranker_predictions, ground_truth_labels,
                                n_bins=10)
print(f"MCE: {mce:.4f}")

# Visualize calibration
fig = plot_reliability_diagram(
    reranker_predictions,
    ground_truth_labels,
    n_bins=10,
    save_path='reranker_calibration.png'
)

# Interpretation
if ece < 0.05:
    print("✅ Excellent calibration - probabilities are trustworthy")
elif ece < 0.10:
    print("✓ Good calibration - acceptable for most uses")
elif ece < 0.15:
    print("⚠ Moderate miscalibration - consider recalibration")
else:
    print("❌ Poor calibration - requires recalibration")
```

## Choosing Binning Strategy

### Uniform Binning (`strategy='uniform'`)
- **Bins**: Equal width (0-0.1, 0.1-0.2, ..., 0.9-1.0)
- **Use when**: Predictions are roughly uniformly distributed
- **Advantage**: Easy to interpret
- **Disadvantage**: Empty bins if predictions are skewed

### Quantile Binning (`strategy='quantile'`)
- **Bins**: Equal number of samples per bin
- **Use when**: Predictions are skewed (most near 0 or 1)
- **Advantage**: All bins have samples
- **Disadvantage**: Bin boundaries less interpretable

**Rule of thumb:** Try both, use quantile if uniform has many empty bins.

## ECE vs Brier Score

| Metric | What it measures | Best for |
|--------|-----------------|----------|
| **ECE** | Average calibration gap across bins | Understanding where/how model is miscalibrated |
| **Brier Score** | Mean squared error of probabilities | Overall probability quality assessment |

**Use both together:**
- Brier Score: Overall probability quality
- ECE: Specific calibration issues
- Reliability diagram: Visual diagnosis

## Fixing Calibration Issues

If your reranker has high ECE:

### 1. Temperature Scaling
```python
# Scale logits before sigmoid
calibrated_probs = sigmoid(logits / temperature)
# Tune temperature on validation set to minimize ECE
```

### 2. Platt Scaling
```python
from sklearn.calibration import calibration_curve
# Fit logistic regression on validation predictions
# Apply to test predictions
```

### 3. Isotonic Regression
```python
from sklearn.isotonic import IsotonicRegression
iso_reg = IsotonicRegression(out_of_bounds='clip')
calibrated_probs = iso_reg.fit_transform(predictions, actuals)
```

### 4. Collect More Training Data
- Calibration often improves with more diverse training examples
- Ensure training data represents test distribution

## Typical Workflow

1. **Train your reranker** on labeled data
2. **Get predictions** on validation/test set
3. **Compute ECE** to assess calibration
4. **Plot reliability diagram** to diagnose issues
5. **Apply recalibration** if ECE > 0.10
6. **Verify improvement** on held-out test set

## Multi-class Example (Graded Relevance)

```python
from ece import classwise_ece

# Predictions for 3 relevance grades: [not_rel, relevant, highly_rel]
predictions = [
    [0.7, 0.2, 0.1],  # Likely not relevant
    [0.1, 0.6, 0.3],  # Likely relevant
    [0.05, 0.15, 0.8],  # Likely highly relevant
    # ... more predictions
]

# True labels (0=not relevant, 1=relevant, 2=highly relevant)
actuals = [0, 1, 2, ...]

# Compute class-wise ECE
overall_ece, class_eces = classwise_ece(predictions, actuals, n_bins=10)

print(f"Overall ECE: {overall_ece:.4f}")
for i, ece in enumerate(class_eces):
    print(f"  Class {i} ECE: {ece:.4f}")
```

## Real-World Impact

Well-calibrated reranker probabilities enable:

✅ **Threshold-based filtering**
- "Only show docs with P(relevant) > 0.8"
- Trustworthy confidence for users

✅ **Cost-optimized cascading**
- Route high-confidence predictions to fast path
- Send uncertain predictions to expensive models

✅ **Risk-aware ranking**
- Balance confidence and relevance
- Critical for high-stakes applications

✅ **User trust**
- Display confidence indicators
- Set appropriate expectations

## Requirements

```bash
pip install numpy matplotlib
```

## Additional Resources

- **Paper**: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
- **TREC Deep Learning Track**: Standard benchmarks for reranker evaluation
- **Reliability diagrams**: Essential visualization tool for calibration

## License

MIT License - use freely for reranker evaluation and improvement!