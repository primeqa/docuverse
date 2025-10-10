# Quick Reference Card

## 🎯 One-Liner Commands

### Calculate Brier Score
```python
from brier_score import brier_score
score = brier_score(predictions, actuals)  # Lower is better (0-1)
```

### Calculate ECE
```python
from ece import expected_calibration_error
ece = expected_calibration_error(predictions, actuals, n_bins=10)  # Lower is better
```

### Generate Reliability Diagram
```python
from ece import plot_reliability_diagram
plot_reliability_diagram(predictions, actuals, save_path='plot.png')
```

### Complete Evaluation
```python
from example_evaluation import evaluate_reranker
metrics = evaluate_reranker(predictions, actuals, "MyModel")
```

## 📊 Interpretation Thresholds

| Metric | Excellent | Good | Moderate | Poor |
|--------|-----------|------|----------|------|
| Brier Score | < 0.10 | 0.10-0.20 | 0.20-0.30 | > 0.30 |
| ECE | < 0.05 | 0.05-0.10 | 0.10-0.15 | > 0.15 |
| MCE | < 0.10 | 0.10-0.20 | 0.20-0.30 | > 0.30 |
| BSS | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |

## 🔧 Common Fixes

### High ECE (> 0.15)
**Problem:** Poor calibration
**Solution:** Temperature scaling
```python
calibrated_probs = sigmoid(logits / temperature)  # temperature > 1
```

### High Brier, Low ECE
**Problem:** Poor discrimination but well-calibrated
**Solution:** Improve model features/architecture

### Low Brier, High ECE
**Problem:** Good discrimination but poorly calibrated
**Solution:** Apply temperature scaling or Platt scaling

## 📁 Files You Need

| Task | File |
|------|------|
| Start learning | [INDEX.md](INDEX.md) |
| Brier Score info | [README.md](README.md) |
| ECE info | [ECE_README.md](ECE_README.md) |
| Choose metrics | [METRICS_COMPARISON.md](METRICS_COMPARISON.md) |
| Code - Brier | [brier_score.py](brier_score.py) |
| Code - ECE | [ece.py](ece.py) |
| Complete example | [example_evaluation.py](example_evaluation.py) |

## ⚡ Copy-Paste Snippets

### Minimal Evaluation
```python
from brier_score import brier_score
from ece import expected_calibration_error

bs = brier_score(preds, labels)
ece = expected_calibration_error(preds, labels)

if ece < 0.10 and bs < 0.15:
    print("✅ Ready for production")
elif ece > 0.10:
    print("⚠️ Needs recalibration")
else:
    print("⚠️ Needs improvement")
```

### Production Monitoring
```python
from ece import expected_calibration_error
from brier_score import brier_score

ece = expected_calibration_error(recent_preds, recent_labels)
bs = brier_score(recent_preds, recent_labels)

if ece > 0.12 or bs > 0.25:
    send_alert("Model degraded")
    
log_metrics({'ece': ece, 'brier': bs})
```

### Model Comparison
```python
from example_evaluation import compare_models

models = {
    "BERT": bert_preds,
    "ColBERT": colbert_preds
}

compare_models(models, ground_truth)
```

## 🎓 Formula Cheat Sheet

### Brier Score
```
BS = (1/N) × Σ(predicted - actual)²
```
- Measures: Overall probability quality
- Range: [0, 1]
- Lower is better

### Expected Calibration Error
```
ECE = Σ (n_k/N) × |accuracy_k - confidence_k|
```
- Measures: Average calibration gap
- Range: [0, 1]
- Lower is better

### Maximum Calibration Error
```
MCE = max|accuracy_k - confidence_k|
```
- Measures: Worst-case calibration
- Range: [0, 1]
- Lower is better

### Brier Decomposition
```
Brier = Reliability - Resolution + Uncertainty
```
- Reliability: Calibration error (lower better)
- Resolution: Discrimination ability (higher better)
- Uncertainty: Inherent data variance

## 🚨 Common Mistakes

❌ **Using only Brier Score**
→ Also check ECE for calibration-specific issues

❌ **Ignoring reliability diagrams**
→ Visual inspection reveals patterns metrics miss

❌ **Not checking MCE**
→ High MCE with low ECE means some bins are poorly calibrated

❌ **Wrong binning strategy**
→ Use quantile binning for skewed predictions

❌ **Not recalibrating**
→ If ECE > 0.10, apply temperature scaling

## 💡 Pro Tips

1. ✅ Report both Brier and ECE
2. ✅ Include reliability diagram
3. ✅ Use n_bins=10 (standard)
4. ✅ Check decomposition for diagnosis
5. ✅ Compare to baseline (BSS)
6. ✅ Monitor in production
7. ✅ Recalibrate if ECE > 0.10

## 📞 Emergency Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| ECE > 0.15 | Miscalibrated | Temperature scaling |
| BS > 0.30 | Poor accuracy | Retrain model |
| MCE >> ECE | Some bins bad | Isotonic regression |
| Empty bins | Skewed preds | Use quantile binning |
| BSS < 0 | Worse than baseline | Check data quality |

## 🎯 Decision Tree

```
Need to evaluate reranker?
├─ Quick check? → Use Brier + ECE
├─ Full assessment? → Use evaluate_reranker()
├─ Compare models? → Use compare_models()
├─ Fix calibration? → Temperature scaling
└─ Monitor production? → Track ECE over time
```

## 📦 Installation

```bash
pip install numpy matplotlib
```

## 🚀 Getting Started (30 seconds)

```python
# 1. Import
from brier_score import brier_score
from ece import expected_calibration_error, plot_reliability_diagram

# 2. Evaluate
bs = brier_score(predictions, actuals)
ece = expected_calibration_error(predictions, actuals)

# 3. Visualize
plot_reliability_diagram(predictions, actuals, save_path='plot.png')

# 4. Decide
if ece < 0.10:
    print("✅ Well-calibrated!")
else:
    print("⚠️ Apply temperature scaling")
```

---

**Print this page for quick reference! 📄**