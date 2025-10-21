# 📚 Reranker Calibration Toolkit - File Index

## 🎯 Start Here
- **[MASTER_README.md](MASTER_README.md)** - Complete overview and quick start guide

## 📖 Documentation by Topic

### Getting Started
1. **[MASTER_README.md](MASTER_README.md)** - Read this first
2. **[example_evaluation.py](example_evaluation.py)** - Run this to see everything in action

### Metric-Specific Guides
- **[README.md](README.md)** - Brier Score comprehensive guide
- **[ECE_README.md](ECE_README.md)** - Expected Calibration Error comprehensive guide
- **[METRICS_COMPARISON.md](METRICS_COMPARISON.md)** - When to use which metric

## 🐍 Python Scripts

### Core Implementation
- **[brier_score.py](brier_score.py)** - Brier Score, BSS, and decomposition
- **[ece.py](ece.py)** - ECE, MCE, ACE, and reliability diagrams
- **[example_evaluation.py](example_evaluation.py)** - Complete evaluation example

### What Each Script Does

#### brier_score.py
```python
# Functions included:
- brier_score()                    # Basic Brier Score
- brier_skill_score()             # Compare to baseline
- brier_score_multiclass()        # Multi-class support
- decompose_brier_score()         # Reliability/Resolution/Uncertainty
```

#### ece.py
```python
# Functions included:
- expected_calibration_error()    # Core ECE metric
- maximum_calibration_error()     # Worst-case MCE
- adaptive_calibration_error()    # Robust to skew
- calibration_curve()             # Data for plotting
- plot_reliability_diagram()      # Visualization
- classwise_ece()                 # Multi-class support
```

#### example_evaluation.py
```python
# Functions included:
- evaluate_reranker()             # Complete single-model evaluation
- compare_models()                # Compare multiple models
```

## 📊 Example Outputs

### Reliability Diagrams (Visual)
- **[reliability_diagram.png](reliability_diagram.png)** - General example
- **[well-calibrated_model_reliability.png](well-calibrated_model_reliability.png)** - Good calibration example
- **[overconfident_model_reliability.png](overconfident_model_reliability.png)** - Overconfident model example
- **[poor_discrimination_model_reliability.png](poor_discrimination_model_reliability.png)** - Poor discrimination example

## 🔍 Find What You Need

### "How do I...?"

#### Calculate Brier Score
→ See [README.md](README.md) or run:
```python
from brier_score import brier_score
score = brier_score(predictions, actuals)
```

#### Calculate ECE
→ See [ECE_README.md](ECE_README.md) or run:
```python
from ece import expected_calibration_error
ece = expected_calibration_error(predictions, actuals)
```

#### Compare Multiple Models
→ See [example_evaluation.py](example_evaluation.py):
```python
from example_evaluation import compare_models
results = compare_models(models_dict, actuals)
```

#### Visualize Calibration
→ See [ECE_README.md](ECE_README.md) or run:
```python
from ece import plot_reliability_diagram
plot_reliability_diagram(preds, actuals, save_path='plot.png')
```

#### Decide Which Metric to Use
→ Read [METRICS_COMPARISON.md](METRICS_COMPARISON.md)

#### Understand Why My Model is Miscalibrated
→ Use decomposition:
```python
from brier_score import decompose_brier_score
reliability, resolution, uncertainty = decompose_brier_score(preds, actuals)
```

#### Fix Poor Calibration
→ See "Fixing Calibration Issues" in [ECE_README.md](ECE_README.md)

## 📈 Usage Patterns

### Pattern 1: Quick Check
```python
from brier_score import brier_score
from ece import expected_calibration_error

bs = brier_score(predictions, actuals)
ece = expected_calibration_error(predictions, actuals)
print(f"Brier: {bs:.4f}, ECE: {ece:.4f}")
```
**When**: Quick sanity check during development

### Pattern 2: Comprehensive Evaluation
```python
from example_evaluation import evaluate_reranker

metrics = evaluate_reranker(predictions, actuals, "My Model")
```
**When**: Full model assessment with all metrics and visualization

### Pattern 3: Model Comparison
```python
from example_evaluation import compare_models

models = {"Model A": preds_a, "Model B": preds_b}
compare_models(models, actuals)
```
**When**: A/B testing or selecting best model

### Pattern 4: Production Monitoring
```python
from ece import expected_calibration_error
from brier_score import brier_score

# In your monitoring pipeline
ece = expected_calibration_error(recent_preds, recent_actuals)
bs = brier_score(recent_preds, recent_actuals)

if ece > 0.12:
    alert("Calibration degraded")
```
**When**: Continuous model quality monitoring

## 📊 Interpretation Quick Reference

### Brier Score
- **< 0.10**: Excellent
- **0.10-0.20**: Good
- **0.20-0.30**: Moderate
- **> 0.30**: Poor

### ECE
- **< 0.05**: Excellent calibration
- **0.05-0.10**: Good calibration
- **0.10-0.15**: Moderate miscalibration
- **> 0.15**: Poor calibration (recalibrate!)

### Brier Skill Score
- **> 0.7**: Much better than baseline
- **0.5-0.7**: Significantly better
- **0.3-0.5**: Moderately better
- **< 0.3**: Marginal improvement

## 🎓 Learning Path

### Beginner
1. Read [MASTER_README.md](MASTER_README.md)
2. Run [example_evaluation.py](example_evaluation.py)
3. Study the generated reliability diagrams

### Intermediate
1. Read [README.md](README.md) for Brier Score details
2. Read [ECE_README.md](ECE_README.md) for ECE details
3. Experiment with your own data

### Advanced
1. Read [METRICS_COMPARISON.md](METRICS_COMPARISON.md)
2. Study decomposition methods
3. Implement custom calibration techniques

## 🔗 External Resources

### Papers
- Brier (1950): "Verification of forecasts expressed in terms of probability"
- Guo et al. (2017): "On Calibration of Modern Neural Networks"
- Naeini et al. (2015): "Obtaining Well Calibrated Probabilities Using Bayesian Binning"

### Benchmarks
- **TREC Deep Learning**: https://microsoft.github.io/msmarco/TREC-Deep-Learning.html
- **MS MARCO**: https://microsoft.github.io/msmarco/
- **BEIR**: https://github.com/beir-cellar/beir

## 💡 Tips

1. **Always start** with [MASTER_README.md](MASTER_README.md)
2. **Run the examples** before using on your own data
3. **Check reliability diagrams** - they reveal issues numbers alone can't
4. **Use both Brier and ECE** - they complement each other
5. **Recalibrate if ECE > 0.10** - temperature scaling usually works

## 📞 Quick Help

### "My ECE is high but Brier is low"
→ Model discriminates well but is poorly calibrated
→ Solution: Apply temperature scaling

### "Many empty bins in my reliability diagram"
→ Predictions are skewed toward 0 or 1
→ Solution: Use `strategy='quantile'` in ECE calculation

### "I get different results with different n_bins"
→ This is normal - ECE is sensitive to bin count
→ Solution: Report results with n_bins=10 (standard)

### "Which metric should I report in my paper?"
→ Report both Brier Score and ECE, plus reliability diagram
→ See [METRICS_COMPARISON.md](METRICS_COMPARISON.md)

---

**Ready to get started?** → [MASTER_README.md](MASTER_README.md)
