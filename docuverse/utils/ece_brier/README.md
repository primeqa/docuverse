# Brier Score Calculator for Reranker Evaluation

A Python implementation for computing the Brier score to evaluate the quality of output probabilities from text encoding rerankers.

## What is the Brier Score?

The Brier score measures the mean squared difference between predicted probabilities and actual outcomes:

```
Brier Score = (1/N) × Σ(predicted_probability - actual_outcome)²
```

- **Range**: 0 (perfect) to 1 (worst)
- **Lower is better**: A score near 0 indicates well-calibrated probabilities
- **Benchmark**: Scores < 0.25 are generally considered well-calibrated

## Quick Start

```python
from brier_score import brier_score

# Your reranker predictions (probabilities)
predictions = [0.9, 0.8, 0.3, 0.1, 0.7]

# Ground truth relevance labels (0 or 1)
actuals = [1, 1, 0, 0, 1]

# Calculate Brier score
score = brier_score(predictions, actuals)
print(f"Brier Score: {score:.4f}")
```

## Functions

### `brier_score(predictions, actuals)`
Calculates the basic Brier score for binary outcomes.

**Parameters:**
- `predictions`: List/array of predicted probabilities (0 to 1)
- `actuals`: List/array of binary outcomes (0 or 1)

**Returns:** Float (0 to 1, lower is better)

### `brier_skill_score(predictions, actuals, baseline_predictions=None)`
Measures improvement over a baseline (typically climatology).

**Returns:** Float where:
- 1.0 = perfect predictions
- 0.0 = same as baseline
- < 0 = worse than baseline

### `decompose_brier_score(predictions, actuals, n_bins=10)`
Decomposes the Brier score into three components:

**Returns:** Tuple of (reliability, resolution, uncertainty)
- **Reliability**: Calibration error (lower is better)
- **Resolution**: Ability to separate outcomes (higher is better)  
- **Uncertainty**: Inherent randomness in the data

Formula: `Brier Score = Reliability - Resolution + Uncertainty`

### `brier_score_multiclass(predictions, actuals)`
Calculates Brier score for multi-class predictions (e.g., relevance grades: not relevant, relevant, highly relevant).

**Parameters:**
- `predictions`: 2D array of shape (N, K) where K is number of classes
- `actuals`: Array of class labels (0 to K-1)

## Use Case: Evaluating a Reranker

```python
import numpy as np
from brier_score import brier_score, brier_skill_score, decompose_brier_score

# Simulate reranker outputs for 100 query-document pairs
n_pairs = 100

# Your reranker's predicted relevance probabilities
reranker_probs = [...]  # Array of probabilities

# Ground truth labels from human annotations
true_labels = [...]  # Binary array (1=relevant, 0=not relevant)

# Evaluate calibration
score = brier_score(reranker_probs, true_labels)
print(f"Brier Score: {score:.4f}")

# Compare to baseline
skill = brier_skill_score(reranker_probs, true_labels)
print(f"Brier Skill Score: {skill:.4f}")

# Understand the components
reliability, resolution, uncertainty = decompose_brier_score(
    reranker_probs, true_labels, n_bins=10
)
print(f"Reliability (calibration error): {reliability:.4f}")
print(f"Resolution (discrimination): {resolution:.4f}")
print(f"Uncertainty (data variance): {uncertainty:.4f}")
```

## Interpreting Results

### Good Reranker Probabilities
- Brier Score < 0.25
- Brier Skill Score > 0.5
- Low reliability (< 0.05)
- High resolution (> 0.10)

### Poor Reranker Probabilities  
- Brier Score > 0.30
- Brier Skill Score near 0 or negative
- High reliability (> 0.10) - indicates miscalibration
- Low resolution (< 0.05) - can't discriminate well

## Typical Workflow

1. **Train your reranker** on labeled query-document pairs
2. **Get predictions** on a held-out test set
3. **Compute Brier score** to assess probability calibration
4. **Compare models** - lower Brier score = better calibration
5. **Decompose score** to diagnose issues (reliability vs resolution)

## Why This Matters

Well-calibrated probabilities are crucial for:
- **Confidence thresholds**: Deciding when to show/hide results
- **Multi-stage ranking**: Cascading to more expensive models
- **User trust**: Showing confidence indicators
- **Cost optimization**: Selective processing based on confidence

## Related Metrics

Consider also evaluating:
- **Expected Calibration Error (ECE)**: Bins predictions and measures calibration
- **NDCG/MRR**: Ranking quality (separate from probability quality)
- **Reliability diagrams**: Visual assessment of calibration

## Requirements

```bash
pip install numpy
```

## License

MIT License - feel free to use and modify for your reranker evaluation needs!