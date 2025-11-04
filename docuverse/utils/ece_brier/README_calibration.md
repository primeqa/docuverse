# Probability Calibration to Minimize ECE Scores

## Overview

This package provides a complete toolkit for calibrating probability predictions to minimize Expected Calibration Error (ECE). It implements three state-of-the-art calibration methods and includes comprehensive visualization tools.

## What is ECE?

**Expected Calibration Error (ECE)** is a metric that measures how well a model's predicted probabilities match the actual probabilities of outcomes. A well-calibrated model should predict probabilities that align with the true frequency of events.

For example:
- If a model predicts 70% probability for 100 samples, approximately 70 of them should be positive
- ECE measures the average deviation from perfect calibration across probability bins
- Lower ECE = Better calibration

### ECE Calculation

ECE divides predictions into bins and calculates:
```
ECE = Σ (|accuracy_in_bin - confidence_in_bin| × proportion_in_bin)
```

## Calibration Methods Implemented

### 1. Temperature Scaling ⭐ (Recommended for Neural Networks)

**How it works:** Divides logits by a learned temperature parameter T before applying softmax.

**Advantages:**
- Simple and fast (single parameter)
- Preserves model predictions' ranking
- Highly effective for neural networks
- Minimal overfitting risk

**Best for:** Neural networks, deep learning models

**Citation count:** 5000+ citations (Guo et al., 2017)

### 2. Platt Scaling (Sigmoid Calibration)

**How it works:** Fits a logistic regression model to map raw predictions to calibrated probabilities.

**Advantages:**
- Parametric method (two parameters: A and B)
- Works well for SVMs and models with sigmoidal distortions
- Less data required than non-parametric methods

**Best for:** Support Vector Machines, boosted models

### 3. Isotonic Regression

**How it works:** Fits a non-parametric, piecewise-constant, monotonically increasing function.

**Advantages:**
- Most flexible method
- Can correct any monotonic miscalibration
- No assumptions about distribution shape

**Disadvantages:**
- Requires more data
- Can overfit with small datasets
- More computationally expensive

**Best for:** Large datasets, complex miscalibration patterns

## Quick Start

### Basic Usage

```python
from probability_calibration import ProbabilityCalibrator
import numpy as np

# Your predicted probabilities and true labels
probs = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
labels = np.array([1, 1, 0, 1, 0])

# Create calibrator
calibrator = ProbabilityCalibrator(n_bins=15)

# Auto-select best method
calibrated_probs = calibrator.calibrate(probs, labels, method='auto')

# Print results
calibrator.print_results()
```

### Using Specific Methods

```python
# Use temperature scaling specifically
calibrated_probs = calibrator.calibrate(probs, labels, method='temperature_scaling')

# Use Platt scaling
calibrated_probs = calibrator.calibrate(probs, labels, method='platt_scaling')

# Use isotonic regression
calibrated_probs = calibrator.calibrate(probs, labels, method='isotonic')
```

### Individual Calibrators

```python
from probability_calibration import TemperatureScaling, PlattScaling, IsotonicCalibration

# Temperature scaling
temp_scaler = TemperatureScaling()
calibrated = temp_scaler.fit_transform(probs, labels)
print(f"Learned temperature: {temp_scaler.temperature}")

# Platt scaling
platt = PlattScaling()
calibrated = platt.fit_transform(probs, labels)
print(f"Parameters - A: {platt.A}, B: {platt.B}")

# Isotonic regression
iso = IsotonicCalibration()
calibrated = iso.fit_transform(probs, labels)
```

## Visualization Tools

### Create Reliability Diagrams

```python
from calibration_visualizations import plot_reliability_diagram
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
plot_reliability_diagram(probs, labels, n_bins=10, ax=ax)
plt.show()
```

### Compare All Methods

```python
from calibration_visualizations import plot_calibration_comparison

fig = plot_calibration_comparison(probs, labels, n_bins=10)
plt.show()
```

### Comprehensive Report

```python
from calibration_visualizations import create_full_calibration_report

fig = create_full_calibration_report(probs, labels, 
                                    save_path='calibration_report.png')
plt.show()
```

## Understanding the Visualizations

### 1. Reliability Diagram (Calibration Curve)
- **Perfect calibration:** Points fall on the diagonal line y=x
- **Overconfident:** Points below diagonal (predictions too high)
- **Underconfident:** Points above diagonal (predictions too low)

### 2. ECE Bin Visualization
- Shows the calibration error in each probability bin
- Bar height = |confidence - accuracy| in that bin
- Color intensity = number of samples in bin
- Total ECE = weighted sum of all bars

### 3. Histogram of Predictions
- Shows distribution of predicted probabilities
- Well-calibrated models should show predictions across the full range
- Overconfident models cluster near 0 and 1

## Key Research Findings

1. **Temperature scaling is remarkably effective** for neural networks despite using only one parameter
2. **Isotonic regression is most flexible** but needs more data to avoid overfitting
3. **Modern neural networks are poorly calibrated** out of the box (Guo et al., 2017)
4. **Batch normalization and model depth** affect calibration quality
5. **Post-hoc calibration doesn't hurt accuracy** (predictions remain the same, only probabilities change)

## When to Use Calibration

Calibration is critical when:
- **Making decisions based on probabilities** (not just predictions)
- **Risk assessment** applications (medical diagnosis, finance)
- **Comparing models** with different confidence ranges
- **Threshold selection** requires accurate probabilities
- **Safety-critical applications** (autonomous vehicles, medical systems)
- **Uncertainty quantification** is important

## Installation Requirements

```bash
pip install numpy scipy scikit-learn matplotlib
```

## Files Included

1. **probability_calibration.py** - Main calibration module
   - `ProbabilityCalibrator` - Main class with auto-selection
   - `TemperatureScaling` - Temperature scaling implementation
   - `PlattScaling` - Platt scaling implementation  
   - `IsotonicCalibration` - Isotonic regression implementation
   - `ECECalculator` - ECE calculation utility

2. **calibration_visualizations.py** - Visualization tools
   - `plot_reliability_diagram()` - Create calibration curves
   - `plot_calibration_comparison()` - Compare methods
   - `plot_ece_bins()` - Visualize ECE calculation
   - `create_full_calibration_report()` - Comprehensive report

3. **README.md** - This guide

## Example Workflow

```python
import numpy as np
from probability_calibration import ProbabilityCalibrator
from calibration_visualizations import create_full_calibration_report
import matplotlib.pyplot as plt

# 1. Load your model predictions
probs = model.predict_proba(X_test)[:, 1]  # Get probability for positive class
labels = y_test

# 2. Create calibrator
calibrator = ProbabilityCalibrator(n_bins=15)

# 3. Calibrate (auto-selects best method)
calibrated_probs = calibrator.calibrate(probs, labels, method='auto')

# 4. Review results
calibrator.print_results()

# 5. Create visual report
fig = create_full_calibration_report(probs, labels, save_path='report.png')
plt.show()

# 6. Use calibrated probabilities for decision making
threshold = 0.5
predictions = (calibrated_probs > threshold).astype(int)
```

## Best Practices

1. **Always use a held-out calibration set** - Don't calibrate on training data
2. **Start with 'auto' method** - Let the algorithm choose the best approach
3. **Use 10-20 bins for ECE** - Balance between granularity and statistical reliability
4. **Check reliability diagrams** - Visual inspection is valuable
5. **For neural networks, try temperature scaling first** - It's simple and effective
6. **For small datasets, prefer parametric methods** - Temperature or Platt scaling
7. **For large datasets with complex patterns, try isotonic** - More flexible
8. **Always validate on a separate test set** - Avoid overfitting to calibration set

## Common Issues and Solutions

### Issue: ECE increases after calibration
**Solution:** You may have too little calibration data. Try temperature scaling (fewer parameters) or use cross-validation.

### Issue: Isotonic regression overfits
**Solution:** Switch to temperature scaling or Platt scaling, which have fewer parameters.

### Issue: Calibrated probabilities don't change much
**Solution:** Your model may already be well-calibrated. Check the original ECE score.

### Issue: All predictions become 0 or 1
**Solution:** Isotonic regression can be aggressive. Try temperature scaling or ensure you have enough calibration samples.

## Performance Tips

1. **Temperature scaling is fastest** - O(n) optimization for single parameter
2. **Isotonic regression is slowest** - O(n log n) for fitting
3. **For real-time applications** - Pre-compute calibration mapping offline
4. **Batch calibration** - Calibrate in batches for large datasets

## References

1. Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML* 
   - Introduced temperature scaling, showed neural networks are poorly calibrated
   
2. Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines."
   - Original Platt scaling paper

3. Zadrozny, B., & Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." *KDD*
   - Isotonic regression for calibration

4. Naeini, M., et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning." *AAAI*
   - Introduced ECE metric

5. Nixon, J., et al. (2019). "Measuring Calibration in Deep Learning." *CVPR*
   - Discussed limitations of ECE

## License

MIT License - Feel free to use in your projects!

## Support

For questions or issues:
1. Check the examples in the main scripts
2. Review the visualization outputs
3. Try different calibration methods
4. Ensure your data is properly formatted (probabilities in [0,1], labels in {0,1})

---

**Remember:** Good calibration means your model's confidence matches its accuracy. A 70% prediction should be right 70% of the time!
