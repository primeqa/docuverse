#!/usr/bin/env python3
"""
Complete Reranker Calibration Evaluation Example

This script demonstrates how to comprehensively evaluate a text encoding
reranker's output probabilities using both Brier Score and ECE metrics.
"""

import numpy as np
from brier_score import (brier_score, brier_skill_score, 
                         decompose_brier_score)
from ece import (expected_calibration_error, maximum_calibration_error,
                 adaptive_calibration_error, plot_reliability_diagram)


def evaluate_reranker(predictions, actuals, model_name="Reranker", save_plots=True):
    """
    Comprehensive evaluation of reranker probability calibration.
    
    Args:
        predictions: Array of predicted probabilities
        actuals: Array of binary ground truth labels
        model_name: Name of the model for reporting
        save_plots: Whether to save reliability diagram
    
    Returns:
        Dictionary of all computed metrics
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    print("=" * 70)
    print(f"Calibration Evaluation: {model_name}")
    print("=" * 70)
    
    # Basic statistics
    n_samples = len(predictions)
    n_positive = np.sum(actuals)
    n_negative = n_samples - n_positive
    base_rate = n_positive / n_samples
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {n_samples}")
    print(f"  Positive samples: {n_positive} ({base_rate:.1%})")
    print(f"  Negative samples: {n_negative} ({(1-base_rate):.1%})")
    print(f"  Mean prediction: {np.mean(predictions):.4f}")
    print(f"  Prediction std: {np.std(predictions):.4f}")
    
    # Brier Score metrics
    print(f"\n📊 Brier Score Metrics:")
    bs = brier_score(predictions, actuals)
    bss = brier_skill_score(predictions, actuals)
    print(f"  Brier Score: {bs:.4f}")
    print(f"  Brier Skill Score: {bss:.4f}")
    
    if bs < 0.10:
        print("  ✅ Excellent Brier Score")
    elif bs < 0.20:
        print("  ✓ Good Brier Score")
    elif bs < 0.30:
        print("  ⚠ Moderate Brier Score")
    else:
        print("  ❌ Poor Brier Score")
    
    # Brier decomposition
    reliability, resolution, uncertainty = decompose_brier_score(
        predictions, actuals, n_bins=10
    )
    print(f"\n📊 Brier Decomposition:")
    print(f"  Reliability (calibration error): {reliability:.4f}")
    print(f"  Resolution (discrimination): {resolution:.4f}")
    print(f"  Uncertainty (data variance): {uncertainty:.4f}")
    print(f"  Check: {reliability:.4f} - {resolution:.4f} + {uncertainty:.4f} = {bs:.4f}")
    
    if reliability < 0.03:
        print("  ✅ Low reliability component - well calibrated")
    elif reliability < 0.05:
        print("  ✓ Acceptable reliability component")
    else:
        print("  ⚠ High reliability component - miscalibration detected")
    
    if resolution > 0.15:
        print("  ✅ High resolution - good discrimination")
    elif resolution > 0.08:
        print("  ✓ Moderate resolution")
    else:
        print("  ⚠ Low resolution - poor discrimination")
    
    # Calibration metrics
    print(f"\n📊 Calibration Metrics:")
    ece_uniform = expected_calibration_error(predictions, actuals, 
                                            n_bins=10, strategy='uniform')
    ece_quantile = expected_calibration_error(predictions, actuals,
                                             n_bins=10, strategy='quantile')
    ace = adaptive_calibration_error(predictions, actuals, n_bins=10)
    mce = maximum_calibration_error(predictions, actuals, n_bins=10)
    
    print(f"  ECE (uniform bins): {ece_uniform:.4f}")
    print(f"  ECE (quantile bins): {ece_quantile:.4f}")
    print(f"  ACE: {ace:.4f}")
    print(f"  MCE: {mce:.4f}")
    
    if ece_uniform < 0.05:
        print("  ✅ Excellent calibration (ECE < 0.05)")
    elif ece_uniform < 0.10:
        print("  ✓ Good calibration (ECE < 0.10)")
    elif ece_uniform < 0.15:
        print("  ⚠ Moderate miscalibration (ECE < 0.15)")
    else:
        print("  ❌ Poor calibration (ECE > 0.15) - recalibration needed")
    
    if mce > 2 * ece_uniform:
        print(f"  ⚠ MCE much higher than ECE - some bins poorly calibrated")
    
    # Calibration by confidence range
    print(f"\n📊 Calibration by Confidence Range:")
    ranges = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
    for low, high in ranges:
        mask = (predictions >= low) & (predictions < high)
        if np.any(mask):
            range_accuracy = np.mean(actuals[mask])
            range_confidence = np.mean(predictions[mask])
            range_n = np.sum(mask)
            gap = abs(range_accuracy - range_confidence)
            print(f"  [{low:.1f}, {high:.1f}): n={range_n:4d}, "
                  f"conf={range_confidence:.3f}, acc={range_accuracy:.3f}, "
                  f"gap={gap:.3f}")
    
    # Overall assessment
    print(f"\n📊 Overall Assessment:")
    
    calibration_good = ece_uniform < 0.10
    accuracy_good = bs < 0.20
    
    if calibration_good and accuracy_good:
        assessment = "✅ EXCELLENT - Model is both accurate and well-calibrated"
        recommendation = "Ready for production deployment"
    elif calibration_good and not accuracy_good:
        assessment = "⚠ MIXED - Well-calibrated but accuracy could improve"
        recommendation = "Consider improving model architecture or features"
    elif not calibration_good and accuracy_good:
        assessment = "⚠ MIXED - Good accuracy but poor calibration"
        recommendation = "Apply temperature scaling or Platt scaling"
    else:
        assessment = "❌ NEEDS IMPROVEMENT - Both accuracy and calibration need work"
        recommendation = "Retrain model or try different architecture"
    
    print(f"  {assessment}")
    print(f"  Recommendation: {recommendation}")
    
    # Generate reliability diagram
    if save_plots:
        plot_path = f"{model_name.lower().replace(' ', '_')}_reliability.png"
        print(f"\n📊 Generating reliability diagram: {plot_path}")
        fig = plot_reliability_diagram(
            predictions, actuals,
            n_bins=10,
            strategy='uniform',
            save_path=plot_path,
            show_histogram=True
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    print("=" * 70)
    
    # Return all metrics
    return {
        'n_samples': n_samples,
        'n_positive': n_positive,
        'base_rate': base_rate,
        'brier_score': bs,
        'brier_skill_score': bss,
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty,
        'ece_uniform': ece_uniform,
        'ece_quantile': ece_quantile,
        'ace': ace,
        'mce': mce,
        'assessment': assessment,
        'recommendation': recommendation
    }


def compare_models(model_predictions_dict, actuals):
    """
    Compare multiple reranker models.
    
    Args:
        model_predictions_dict: Dict mapping model names to prediction arrays
        actuals: Ground truth labels
    
    Returns:
        DataFrame-like comparison (printed)
    """
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    
    results = {}
    for model_name, predictions in model_predictions_dict.items():
        metrics = evaluate_reranker(predictions, actuals, model_name, save_plots=False)
        results[model_name] = metrics
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Brier':<8} {'ECE':<8} {'MCE':<8} {'BSS':<8}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} "
              f"{metrics['brier_score']:<8.4f} "
              f"{metrics['ece_uniform']:<8.4f} "
              f"{metrics['mce']:<8.4f} "
              f"{metrics['brier_skill_score']:<8.4f}")
    
    # Determine best model
    print("\n📊 Best Model by Metric:")
    best_brier = min(results.items(), key=lambda x: x[1]['brier_score'])
    best_ece = min(results.items(), key=lambda x: x[1]['ece_uniform'])
    best_bss = max(results.items(), key=lambda x: x[1]['brier_skill_score'])
    
    print(f"  Lowest Brier Score: {best_brier[0]} ({best_brier[1]['brier_score']:.4f})")
    print(f"  Lowest ECE: {best_ece[0]} ({best_ece[1]['ece_uniform']:.4f})")
    print(f"  Highest BSS: {best_bss[0]} ({best_bss[1]['brier_skill_score']:.4f})")
    
    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Complete Reranker Calibration Evaluation Demo")
    print("=" * 70)
    
    # Simulate a realistic reranker scenario
    np.random.seed(42)
    n_samples = 500
    
    # Ground truth: 40% relevant, 60% irrelevant
    actuals = np.concatenate([
        np.ones(200),
        np.zeros(300)
    ]).astype(int)
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    actuals = actuals[shuffle_idx]
    
    print("\n" + "=" * 70)
    print("Scenario 1: Well-Calibrated Reranker")
    print("=" * 70)
    
    # Model A: Well-calibrated (good predictions for relevant docs)
    relevant_mask = actuals == 1
    model_a_preds = np.zeros(n_samples)
    model_a_preds[relevant_mask] = np.random.beta(6, 2, np.sum(relevant_mask))
    model_a_preds[~relevant_mask] = np.random.beta(2, 6, np.sum(~relevant_mask))
    
    metrics_a = evaluate_reranker(model_a_preds, actuals, 
                                  "Well-Calibrated Model", save_plots=True)
    
    print("\n" + "=" * 70)
    print("Scenario 2: Overconfident Reranker")
    print("=" * 70)
    
    # Model B: Overconfident (predictions are 0.2 higher than they should be)
    model_b_preds = np.clip(model_a_preds + 0.2, 0, 1)
    
    metrics_b = evaluate_reranker(model_b_preds, actuals,
                                  "Overconfident Model", save_plots=True)
    
    print("\n" + "=" * 70)
    print("Scenario 3: Poor Discrimination Reranker")
    print("=" * 70)
    
    # Model C: Poor discrimination (can't distinguish relevant/irrelevant well)
    model_c_preds = np.random.uniform(0.3, 0.7, n_samples)
    
    metrics_c = evaluate_reranker(model_c_preds, actuals,
                                  "Poor Discrimination Model", save_plots=True)
    
    # Compare all models
    models = {
        "Well-Calibrated": model_a_preds,
        "Overconfident": model_b_preds,
        "Poor Discrimination": model_c_preds
    }
    
    comparison_results = compare_models(models, actuals)
    
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Use BOTH Brier Score and ECE for complete picture")
    print("  2. ECE < 0.10 indicates good calibration")
    print("  3. Brier Score < 0.20 indicates good overall quality")
    print("  4. Check reliability diagrams for visual diagnosis")
    print("  5. Apply temperature scaling if ECE > 0.10")
    print("=" * 70)