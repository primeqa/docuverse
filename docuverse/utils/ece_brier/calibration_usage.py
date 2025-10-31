"""
Simple Example: How to Use Probability Calibration

This script demonstrates a complete workflow for calibrating
probabilities to minimize ECE scores.
"""

import numpy as np
from probability_calibration import ProbabilityCalibrator, ECECalculator

# ============================================================================
# EXAMPLE 1: Basic Usage with Auto Method Selection
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: Auto-Select Best Calibration Method")
print("=" * 70)

# Simulate a poorly calibrated model (e.g., overconfident neural network)
np.random.seed(123)
n_samples = 500

# Create synthetic data with overconfident predictions
true_labels = np.random.binomial(1, 0.4, n_samples)
raw_probs = true_labels * np.random.beta(9, 2, n_samples) + \
            (1 - true_labels) * np.random.beta(2, 9, n_samples)
raw_probs = np.clip(raw_probs, 0.01, 0.99)

print(f"\nDataset: {n_samples} samples")
print(f"True positive rate: {true_labels.mean():.3f}")
print(f"Average prediction: {raw_probs.mean():.3f}\n")

# Create calibrator and auto-select best method
calibrator = ProbabilityCalibrator(n_bins=15)
calibrated_probs = calibrator.calibrate(raw_probs, true_labels, method='auto')['test']['probs']

# View results
calibrator.print_results()

print("\nSample Predictions (first 5):")
print("-" * 50)
print(f"{'Original':<12} {'Calibrated':<12} {'True Label':<12}")
print("-" * 50)
for i in range(5):
    print(f"{raw_probs[i]:<12.4f} {calibrated_probs[i]:<12.4f} {true_labels[i]:<12}")


# ============================================================================
# EXAMPLE 2: Using Specific Methods
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Compare Specific Methods")
print("=" * 70)

from probability_calibration import TemperatureScaling, PlattScaling, IsotonicCalibration

# Split data into calibration and test sets
split_idx = int(0.7 * n_samples)
cal_probs, test_probs = raw_probs[:split_idx], raw_probs[split_idx:]
cal_labels, test_labels = true_labels[:split_idx], true_labels[split_idx:]

print(f"\nCalibration set: {len(cal_probs)} samples")
print(f"Test set: {len(test_probs)} samples\n")

# Create ECE calculator
ece_calc = ECECalculator(n_bins=15)
original_ece = ece_calc.calculate(test_probs, test_labels)

print(f"Original Test ECE: {original_ece:.6f}\n")

# Temperature Scaling
temp_scaler = TemperatureScaling()
temp_scaler.fit(cal_probs, cal_labels)
temp_calibrated = temp_scaler.transform(test_probs)
temp_ece = ece_calc.calculate(temp_calibrated, test_labels)
print(f"Temperature Scaling (T={temp_scaler.temperature:.3f}):")
print(f"  Test ECE: {temp_ece:.6f}")
print(f"  Improvement: {(original_ece - temp_ece):.6f} ({(original_ece - temp_ece)/original_ece*100:+.1f}%)\n")

# Platt Scaling
platt = PlattScaling()
platt.fit(cal_probs, cal_labels)
platt_calibrated = platt.transform(test_probs)
platt_ece = ece_calc.calculate(platt_calibrated, test_labels)
print(f"Platt Scaling (A={platt.A:.3f}, B={platt.B:.3f}):")
print(f"  Test ECE: {platt_ece:.6f}")
print(f"  Improvement: {(original_ece - platt_ece):.6f} ({(original_ece - platt_ece)/original_ece*100:+.1f}%)\n")

# Isotonic Regression
iso = IsotonicCalibration()
iso.fit(cal_probs, cal_labels)
iso_calibrated = iso.transform(test_probs)
iso_ece = ece_calc.calculate(iso_calibrated, test_labels)
print(f"Isotonic Regression:")
print(f"  Test ECE: {iso_ece:.6f}")
print(f"  Improvement: {(original_ece - iso_ece):.6f} ({(original_ece - iso_ece)/original_ece*100:+.1f}%)\n")


# ============================================================================
# EXAMPLE 3: Real-World Workflow
# ============================================================================

print("=" * 70)
print("EXAMPLE 3: Complete Real-World Workflow")
print("=" * 70)

# Simulate getting predictions from your ML model
def simulate_ml_model_predictions(n):
    """Simulate predictions from an overconfident classifier"""
    labels = np.random.binomial(1, 0.6, n)
    # Overconfident predictions
    probs = labels * np.random.beta(10, 1.5, n) + \
            (1 - labels) * np.random.beta(1.5, 10, n)
    return np.clip(probs, 0.001, 0.999), labels

# Get validation set predictions (for calibration)
val_probs, val_labels = simulate_ml_model_predictions(800)

# Get test set predictions (for evaluation)
test_probs_real, test_labels_real = simulate_ml_model_predictions(200)

print(f"\nValidation set: {len(val_probs)} samples")
print(f"Test set: {len(test_probs_real)} samples\n")

# Step 1: Choose calibration method on validation set
print("Step 1: Selecting best calibration method...")
calibrator_final = ProbabilityCalibrator(n_bins=15)
_ = calibrator_final.calibrate(val_probs, val_labels, method='auto')
best_method = calibrator_final.best_method
print(f"Selected method: {best_method}\n")

# Step 2: Apply to test set
print("Step 2: Applying calibration to test set...")
test_calibrated = calibrator_final.methods[best_method].transform(test_probs_real)

# Step 3: Evaluate
test_ece_before = ece_calc.calculate(test_probs_real, test_labels_real)
test_ece_after = ece_calc.calculate(test_calibrated, test_labels_real)

print(f"\nTest Set Results:")
print(f"  ECE before calibration: {test_ece_before:.6f}")
print(f"  ECE after calibration:  {test_ece_after:.6f}")
print(f"  Improvement: {test_ece_before - test_ece_after:.6f} ({(test_ece_before - test_ece_after)/test_ece_before*100:+.1f}%)")

# Step 4: Make decisions using calibrated probabilities
threshold = 0.5
predictions_before = (test_probs_real > threshold).astype(int)
predictions_after = (test_calibrated > threshold).astype(int)

accuracy_before = (predictions_before == test_labels_real).mean()
accuracy_after = (predictions_after == test_labels_real).mean()

print(f"\nPrediction Accuracy:")
print(f"  Before calibration: {accuracy_before:.3f}")
print(f"  After calibration:  {accuracy_after:.3f}")
print(f"  Note: Calibration preserves or slightly improves accuracy")


# ============================================================================
# EXAMPLE 4: Detailed ECE Analysis
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Detailed ECE Bin Analysis")
print("=" * 70)

ece, bin_details = ece_calc.calculate(raw_probs, true_labels, return_details=True)

print(f"\nTotal ECE: {ece:.6f}\n")
print("Bin-by-Bin Breakdown:")
print("-" * 70)
print(f"{'Bin Range':<15} {'Count':<10} {'Confidence':<12} {'Accuracy':<12} {'Error':<10}")
print("-" * 70)

for bin_info in bin_details:
    range_str = f"[{bin_info['range'][0]:.2f}, {bin_info['range'][1]:.2f}]"
    print(f"{range_str:<15} {bin_info['count']:<10} "
          f"{bin_info['confidence']:<12.4f} {bin_info['accuracy']:<12.4f} "
          f"{bin_info['error']:<10.4f}")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 70)

print("""
Key Takeaways:

1. ECE measures calibration quality (lower is better)
2. Temperature scaling works best for neural networks
3. Isotonic regression is most flexible but needs more data
4. Always use a separate calibration set
5. Calibration improves probability estimates without changing predictions

Next Steps:

1. Use 'auto' method to find the best calibration approach
2. Create visualizations with calibration_visualizations.py
3. Apply calibration to your ML pipeline
4. Use calibrated probabilities for decision-making

For more details, see README.md
""")

print("=" * 70)
print("Examples completed successfully!")
print("=" * 70)
