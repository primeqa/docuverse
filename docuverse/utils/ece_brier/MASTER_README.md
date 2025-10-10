# Reranker Calibration Evaluation Toolkit

A complete toolkit for evaluating the quality of output probabilities from text encoding rerankers, implementing both **Brier Score** and **Expected Calibration Error (ECE)** metrics.

## 📦 What's Included

### Core Scripts
- **`brier_score.py`** - Brier Score, Brier Skill Score, and Brier decomposition
- **`ece.py`** - ECE, MCE, ACE, and reliability diagram generation
- **`example_evaluation.py`** - Complete end-to-end evaluation example

### Documentation
- **`README.md`** - Brier Score guide
- **`ECE_README.md`** - ECE comprehensive guide
- **`METRICS_COMPARISON.md`** - When to use each metric

### Example Outputs
- **`reliability_diagram.png`** - Example calibration visualization
- **`well-calibrated_model_reliability.png`** - Well-calibrated model example
- **`overconfident_model_reliability.png`** - Overconfident model example
- **`poor_discrimination_model_reliability.png`** - Poor discrimination example

## 🚀 Quick Start

### Install Dependencies
```bash
pip install numpy matplotlib
```

### Basic Usage

```python
from brier_score import brier_score
from ece import expected_calibration_error, plot_reliability_diagram

# Your reranker predictions and ground truth
predictions = [0.9, 0.8, 0.3, 0.1, 0.7, ...]  # Probabilities
actuals = [1, 1, 0, 0, 1, ...]                 # Binary labels

# Compute metrics
brier = brier_score(predictions, actuals)
ece = expected_calibration_error(predictions, actuals, n_bins=10)

print(f"Brier Score: {brier:.4f}")
print(f"ECE: {ece:.4f}")

# Generate reliability diagram
plot_reliability_diagram(predictions, actuals, save_path='my_plot.png')
```

### Complete Evaluation

```python
from example_evaluation import evaluate_reranker

# Comprehensive evaluation with all metrics
metrics = evaluate_reranker(predictions, actuals, model_name="My Reranker")

# Automatically provides:
# - Brier Score & Brier Skill Score
# - ECE, MCE, ACE
# - Brier decomposition
# - Calibration by confidence range
# - Overall assessment & recommendations
# - Reliability diagram
```

## 📊 Metrics Overview

| Metric | What It Measures | When to Use | Good Value |
|--------|-----------------|-------------|------------|
| **Brier Score** | Overall probability quality | Comparing models | < 0.15 |
| **ECE** | Calibration accuracy | Diagnosing miscalibration | < 0.10 |
| **MCE** | Worst-case calibration | Safety-critical apps | < 0.15 |
| **ACE** | Calibration (robust to skew) | Skewed predictions | < 0.10 |
| **BSS** | Improvement over baseline | Showing value-add | > 0.5 |

## 🎯 Typical Use Cases

### 1. Model Development
```python
# Train your reranker
model = train_reranker(training_data)

# Evaluate on validation set
val_preds = model.predict(val_queries, val_docs)
val_labels = get_ground_truth(val_queries, val_docs)

# Check calibration
ece = expected_calibration_error(val_preds, val_labels)

if ece > 0.10:
    print("⚠️ Apply temperature scaling")
    calibrated_model = apply_temperature_scaling(model, val_data)
```

### 2. Model Comparison
```python
from example_evaluation import compare_models

models = {
    "BERT Reranker": bert_predictions,
    "ColBERT": colbert_predictions,
    "Cross-Encoder": cross_encoder_predictions
}

# Compare all models
results = compare_models(models, ground_truth_labels)
# Outputs comparison table with all metrics
```

### 3. Production Monitoring
```python
# Weekly monitoring
def monitor_reranker_calibration(predictions, actuals):
    ece = expected_calibration_error(predictions, actuals)
    bs = brier_score(predictions, actuals)
    
    if ece > 0.12:
        send_alert("Reranker calibration degraded")
        
    if bs > 0.25:
        send_alert("Reranker accuracy degraded")
    
    log_metrics({'ece': ece, 'brier': bs})
```

## 🔍 Interpreting Results

### Scenario Analysis

#### ✅ Well-Calibrated Model
```
Brier Score: 0.08
ECE: 0.04
Resolution: 0.19
```
**Interpretation:** Excellent - ready for production
**Action:** Deploy as-is

#### ⚠️ Overconfident Model
```
Brier Score: 0.14
ECE: 0.18
Resolution: 0.19
```
**Interpretation:** Good discrimination but poor calibration
**Action:** Apply temperature scaling (T > 1)

#### ⚠️ Poor Discrimination
```
Brier Score: 0.27
ECE: 0.03
Resolution: 0.01
```
**Interpretation:** Well-calibrated but can't distinguish relevant/irrelevant
**Action:** Improve model architecture or features

#### ❌ Needs Improvement
```
Brier Score: 0.35
ECE: 0.22
Resolution: 0.05
```
**Interpretation:** Both accuracy and calibration are poor
**Action:** Retrain or try different architecture

## 📈 Visualization

The toolkit generates reliability diagrams that show:
- **Diagonal line** = perfect calibration
- **Points above diagonal** = overconfident (predicts higher than actual)
- **Points below diagonal** = underconfident (predicts lower than actual)
- **Distance from diagonal** = calibration error
- **Histogram** = distribution of predictions

Example output:

```
Reliability Diagram
ECE: 0.0376, MCE: 0.1234, Bins: 10 (uniform)
```

See included `.png` files for examples.

## 🛠️ Advanced Features

### Temperature Scaling
```python
def temperature_scale(logits, temperature):
    """Apply temperature scaling to improve calibration"""
    import torch
    return torch.sigmoid(logits / temperature)

# Find optimal temperature on validation set
best_temp = find_optimal_temperature(val_logits, val_labels)
calibrated_preds = temperature_scale(test_logits, best_temp)
```

### Brier Score Decomposition
```python
from brier_score import decompose_brier_score

reliability, resolution, uncertainty = decompose_brier_score(preds, labels)

# Diagnose issues
if reliability > 0.05:
    print("⚠️ Calibration problem - apply temperature scaling")
    
if resolution < 0.05:
    print("⚠️ Discrimination problem - improve model features")
```

### Multi-class Support
```python
from brier_score import brier_score_multiclass
from ece import classwise_ece

# For graded relevance: 0=not relevant, 1=relevant, 2=highly relevant
predictions = [
    [0.7, 0.2, 0.1],
    [0.1, 0.6, 0.3],
    [0.05, 0.15, 0.8]
]
actuals = [0, 1, 2]

bs_multi = brier_score_multiclass(predictions, actuals)
overall_ece, class_eces = classwise_ece(predictions, actuals)
```

## 📚 Further Reading

### Key Papers
- **Brier Score**: Brier, G.W. (1950). "Verification of forecasts expressed in terms of probability"
- **Calibration**: Guo et al. (2017). "On Calibration of Modern Neural Networks"
- **ECE**: Naeini et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning"

### Benchmarks
- **TREC Deep Learning Track**: Standard evaluation for document/passage ranking
- **MS MARCO**: Large-scale ranking dataset with hundreds of thousands of training queries
- **BEIR**: Diverse retrieval tasks for testing generalization

### Related Metrics
- **NDCG** - Ranking quality (complementary to calibration)
- **MRR** - First relevant result position
- **MAP** - Precision across recall levels

## 💡 Best Practices

1. **Always use both Brier and ECE** - they complement each other
2. **Generate reliability diagrams** - visual inspection is crucial
3. **Check multiple bin counts** - try n_bins=5, 10, 15
4. **Compare to baselines** - use Brier Skill Score
5. **Monitor in production** - track ECE over time
6. **Recalibrate when ECE > 0.10** - temperature scaling is usually sufficient
7. **Don't ignore MCE** - worst-case matters for safety-critical apps

## 🤝 Contributing

This toolkit is designed to be extensible. To add new metrics:

1. Implement in `brier_score.py` or `ece.py`
2. Add tests in `if __name__ == "__main__"` block
3. Update documentation
4. Add example in `example_evaluation.py`

## 🐛 Troubleshooting

### Common Issues

**Issue**: ECE is high but Brier is low
**Solution**: Model discriminates well but is poorly calibrated - apply temperature scaling

**Issue**: Many empty bins with uniform binning
**Solution**: Use `strategy='quantile'` or reduce `n_bins`

**Issue**: MCE much higher than ECE
**Solution**: Some confidence ranges are poorly calibrated - check reliability diagram

**Issue**: ValueError about array lengths
**Solution**: Ensure predictions and actuals have same length and no NaN values

## 📞 Support

For questions or issues:
1. Check the documentation files
2. Review the example outputs
3. Run `example_evaluation.py` to see expected behavior

---

**Happy Evaluating! 🎯**

Remember: Well-calibrated probabilities → Better decisions → Happier users