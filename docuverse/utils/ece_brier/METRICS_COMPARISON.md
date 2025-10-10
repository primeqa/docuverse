# Calibration Metrics Comparison Guide

A practical guide for choosing the right calibration metrics when evaluating text encoding rerankers.

## Quick Reference Table

| Metric | Formula | Range | Best For | Interpretation |
|--------|---------|-------|----------|----------------|
| **Brier Score** | (1/N)Σ(p-y)² | [0, 1] | Overall probability quality | Lower = better overall |
| **ECE** | Σ(n/N)\|acc-conf\| | [0, 1] | Understanding miscalibration | Lower = better calibration |
| **MCE** | max\|acc-conf\| | [0, 1] | Worst-case calibration | Pessimistic view |
| **ACE** | ECE with quantile bins | [0, 1] | Skewed distributions | Robust to data skew |
| **BSS** | 1 - BS/BS_baseline | [-∞, 1] | Improvement over baseline | > 0 is better |

## When to Use Each Metric

### Use Brier Score when:
✅ You want a **single overall quality score**
✅ Comparing **multiple models** (lower is better)
✅ You care about both **calibration AND discrimination**
✅ Publishing results (widely recognized metric)

**Example scenario:**
```python
# Comparing two rerankers
model_a_brier = brier_score(model_a_preds, labels)  # 0.1234
model_b_brier = brier_score(model_b_preds, labels)  # 0.0987

# Model B is better overall
print(f"Model A: {model_a_brier:.4f}")
print(f"Model B: {model_b_brier:.4f}")
```

### Use ECE when:
✅ You need to **understand WHERE miscalibration occurs**
✅ You want to **diagnose calibration problems**
✅ Deciding whether to **apply recalibration** (ECE > 0.10 → recalibrate)
✅ You care specifically about **probability calibration** (not discrimination)

**Example scenario:**
```python
# Understanding calibration quality
ece = expected_calibration_error(predictions, labels, n_bins=10)

if ece < 0.05:
    print("✅ Excellent - probabilities are trustworthy")
elif ece < 0.10:
    print("✓ Good - probabilities are usable")
else:
    print("❌ Poor - apply temperature scaling")
```

### Use MCE when:
✅ You need **worst-case guarantees**
✅ Working on **safety-critical applications**
✅ Want to identify the **most problematic confidence range**
✅ ECE looks good but you suspect issues in specific bins

**Example scenario:**
```python
# Safety-critical reranker for medical document retrieval
ece = expected_calibration_error(preds, labels)  # 0.08 (looks OK)
mce = maximum_calibration_error(preds, labels)   # 0.35 (concerning!)

# MCE reveals that one bin is very poorly calibrated
# even though average (ECE) is acceptable
```

### Use ACE when:
✅ Predictions are **heavily skewed** (most near 0 or 1)
✅ Uniform bins result in **many empty bins**
✅ You want **equal representation** across confidence ranges
✅ Comparing models with different prediction distributions

**Example scenario:**
```python
# Reranker that's very confident (most preds near 0 or 1)
ece_uniform = expected_calibration_error(preds, labels, strategy='uniform')
ace = adaptive_calibration_error(preds, labels)

# ACE gives better signal when predictions are skewed
print(f"ECE (uniform): {ece_uniform:.4f}")  # May have empty bins
print(f"ACE (quantile): {ace:.4f}")         # All bins populated
```

### Use Brier Skill Score when:
✅ Showing **improvement over a baseline**
✅ Comparing to **simple baselines** (e.g., always predict class frequency)
✅ Want to demonstrate that your model **adds value**
✅ Reporting relative rather than absolute performance

**Example scenario:**
```python
# Show your reranker is better than naive baseline
bss = brier_skill_score(model_preds, labels)  # 0.65

print(f"BSS: {bss:.4f}")
if bss > 0.5:
    print("✅ 65% improvement over climatology baseline")
```

## Combining Metrics for Complete Picture

### Recommended Evaluation Pipeline

```python
from brier_score import brier_score, brier_skill_score, decompose_brier_score
from ece import expected_calibration_error, maximum_calibration_error, plot_reliability_diagram

# 1. Overall quality
bs = brier_score(predictions, actuals)
bss = brier_skill_score(predictions, actuals)

print(f"📊 Overall Quality:")
print(f"  Brier Score: {bs:.4f}")
print(f"  Brier Skill Score: {bss:.4f}")

# 2. Calibration assessment
ece = expected_calibration_error(predictions, actuals, n_bins=10)
mce = maximum_calibration_error(predictions, actuals, n_bins=10)

print(f"\n📊 Calibration:")
print(f"  ECE: {ece:.4f}")
print(f"  MCE: {mce:.4f}")

# 3. Decomposition for diagnosis
reliability, resolution, uncertainty = decompose_brier_score(predictions, actuals)

print(f"\n📊 Brier Decomposition:")
print(f"  Reliability (calibration): {reliability:.4f}")
print(f"  Resolution (discrimination): {resolution:.4f}")
print(f"  Uncertainty (data): {uncertainty:.4f}")

# 4. Visual diagnosis
plot_reliability_diagram(predictions, actuals, save_path='calibration_plot.png')

# 5. Decision
if ece < 0.10 and bs < 0.15:
    print("\n✅ Model is well-calibrated and accurate")
elif ece > 0.10:
    print("\n⚠️  Apply temperature scaling to improve calibration")
else:
    print("\n✓ Acceptable performance")
```

## Interpretation Matrix

### Scenario 1: Low Brier, Low ECE
```
Brier Score: 0.08
ECE: 0.03
```
**Interpretation:** ✅ Excellent model - both accurate and calibrated
**Action:** Deploy as-is

### Scenario 2: Low Brier, High ECE
```
Brier Score: 0.10
ECE: 0.18
```
**Interpretation:** ⚠️ Good discrimination but poor calibration
**Action:** Apply temperature scaling - probabilities need recalibration but ranking is good

### Scenario 3: High Brier, Low ECE
```
Brier Score: 0.35
ECE: 0.04
```
**Interpretation:** ⚠️ Well-calibrated but poor discrimination
**Action:** Improve model architecture/features - calibration is fine but accuracy is low

### Scenario 4: High Brier, High ECE
```
Brier Score: 0.40
ECE: 0.25
```
**Interpretation:** ❌ Both accuracy and calibration are poor
**Action:** Retrain model or try different architecture

## Common Patterns

### Pattern 1: Overconfident Reranker
```python
# Signs:
# - High ECE (> 0.15)
# - MCE even higher
# - Reliability diagram shows points above diagonal
# - Brier decomposition: high reliability component

# Solution: Temperature scaling
# calibrated_probs = sigmoid(logits / T) where T > 1
```

### Pattern 2: Underconfident Reranker
```python
# Signs:
# - Moderate ECE
# - Reliability diagram shows points below diagonal
# - Model performs better than it "thinks"

# Solution: Usually less problematic, but can use T < 1 if needed
```

### Pattern 3: Bimodal Confidence
```python
# Signs:
# - ECE is okay, but MCE is high
# - Some bins well-calibrated, others not
# - Reliability diagram shows non-monotonic pattern

# Solution: 
# - Isotonic regression (more flexible than temperature scaling)
# - Investigate feature/data issues causing inconsistency
```

## Metric Selection Decision Tree

```
START: Need to evaluate reranker probabilities?
│
├─→ Need single summary metric?
│   └─→ Use: Brier Score
│
├─→ Need to diagnose calibration issues?
│   └─→ Use: ECE + Reliability Diagram
│
├─→ Safety-critical application?
│   └─→ Use: MCE (worst-case)
│
├─→ Skewed prediction distribution?
│   └─→ Use: ACE (quantile binning)
│
├─→ Comparing to baseline?
│   └─→ Use: Brier Skill Score
│
└─→ Comprehensive evaluation?
    └─→ Use: All of the above!
```

## Real-World Examples

### Example 1: Production Reranker Monitoring
```python
# Weekly monitoring dashboard
metrics = {
    'brier_score': brier_score(preds, labels),
    'ece': expected_calibration_error(preds, labels),
    'mce': maximum_calibration_error(preds, labels)
}

# Alert if calibration degrades
if metrics['ece'] > 0.12:
    send_alert("Reranker calibration degraded, consider retraining")
```

### Example 2: A/B Test Between Models
```python
# Compare two reranker versions
model_a_metrics = {
    'brier': brier_score(model_a_preds, labels),
    'ece': expected_calibration_error(model_a_preds, labels)
}

model_b_metrics = {
    'brier': brier_score(model_b_preds, labels),
    'ece': expected_calibration_error(model_b_preds, labels)
}

# Choose model with better balance of accuracy and calibration
if model_b_metrics['brier'] < model_a_metrics['brier'] and \
   model_b_metrics['ece'] < model_a_metrics['ece']:
    print("Deploy Model B - better on both metrics")
```

### Example 3: Post-Training Calibration
```python
# Before calibration
bs_before = brier_score(raw_preds, labels)
ece_before = expected_calibration_error(raw_preds, labels)

# Apply temperature scaling
calibrated_preds = temperature_scale(raw_preds, T=1.5)

# After calibration
bs_after = brier_score(calibrated_preds, labels)
ece_after = expected_calibration_error(calibrated_preds, labels)

print(f"Brier: {bs_before:.4f} → {bs_after:.4f}")
print(f"ECE: {ece_before:.4f} → {ece_after:.4f}")
# Expect: ECE improves significantly, Brier improves slightly
```

## Summary Recommendations

| Use Case | Primary Metric | Secondary Metrics |
|----------|---------------|-------------------|
| Model comparison | Brier Score | ECE, NDCG |
| Calibration diagnosis | ECE + Diagram | MCE, Brier Decomposition |
| Production monitoring | ECE | Brier Score |
| Safety-critical | MCE | ECE |
| Research paper | All metrics | Include reliability diagram |
| Quick check | ECE | - |

## Key Takeaways

1. **No single metric is enough** - use multiple metrics for complete picture
2. **Brier Score** = overall quality (accuracy + calibration)
3. **ECE** = specific to calibration, actionable for improvement
4. **MCE** = worst-case view, important for safety
5. **Reliability diagrams** = essential for visual understanding
6. **Always compare** to baselines (BSS) and across bins (MCE)

## Next Steps

1. Start with **ECE and Brier Score** for quick assessment
2. Generate **reliability diagram** to visualize issues
3. Use **decomposition** to understand if issue is calibration or discrimination
4. Apply **appropriate recalibration** technique based on diagnosis
5. **Re-evaluate** to confirm improvement