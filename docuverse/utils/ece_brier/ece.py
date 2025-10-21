#!/usr/bin/env python3
"""
Expected Calibration Error (ECE) Calculator for Text Encoding Reranker Evaluation

ECE measures how well predicted probabilities align with actual outcomes by:
1. Binning predictions by confidence level
2. Computing accuracy within each bin
3. Measuring the weighted average difference between confidence and accuracy

ECE = Σ (n_k/N) * |acc_k - conf_k|

where:
- n_k is the number of samples in bin k
- N is the total number of samples
- acc_k is the accuracy (observed frequency) in bin k
- conf_k is the average confidence (predicted probability) in bin k
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional
import warnings


def expected_calibration_error(predictions: Union[List[float], np.ndarray],
                               actuals: Union[List[int], np.ndarray],
                               n_bins: int = 10,
                               strategy: str = 'uniform') -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and observed accuracy
    across different confidence bins.
    
    Args:
        predictions: Predicted probabilities (values between 0 and 1)
        actuals: Actual binary outcomes (0 or 1)
        n_bins: Number of bins to use for grouping predictions
        strategy: Binning strategy - 'uniform' (equal width) or 'quantile' (equal frequency)
    
    Returns:
        ECE score (lower is better, range 0-1)
    
    Raises:
        ValueError: If inputs are invalid
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Validation
    if len(predictions) != len(actuals):
        raise ValueError(f"Length mismatch: predictions ({len(predictions)}) "
                        f"vs actuals ({len(actuals)})")
    
    if len(predictions) == 0:
        raise ValueError("Empty predictions array")
    
    if not np.all((predictions >= 0) & (predictions <= 1)):
        raise ValueError("Predictions must be between 0 and 1")
    
    if not np.all((actuals == 0) | (actuals == 1)):
        raise ValueError("Actuals must be binary (0 or 1)")
    
    if strategy not in ['uniform', 'quantile']:
        raise ValueError("Strategy must be 'uniform' or 'quantile'")
    
    # Create bins
    if strategy == 'uniform':
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins[1:-1])
    else:  # quantile
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(predictions, quantiles * 100)
        bins = np.unique(bins)  # Remove duplicate bin edges
        if len(bins) < 2:
            warnings.warn("Not enough unique values for quantile binning, using uniform")
            bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins[1:-1])
    
    ece = 0.0
    n_total = len(predictions)
    
    for i in range(len(bins) - 1):
        # Get predictions and actuals in this bin
        mask = bin_indices == i
        
        if not np.any(mask):
            continue
        
        n_bin = np.sum(mask)
        
        # Average confidence in this bin
        avg_confidence = np.mean(predictions[mask])
        
        # Accuracy (observed frequency) in this bin
        accuracy = np.mean(actuals[mask])
        
        # Weighted contribution to ECE
        ece += (n_bin / n_total) * np.abs(accuracy - avg_confidence)
    
    return ece


def maximum_calibration_error(predictions: Union[List[float], np.ndarray],
                              actuals: Union[List[int], np.ndarray],
                              n_bins: int = 10,
                              strategy: str = 'uniform') -> float:
    """
    Calculate Maximum Calibration Error (MCE).
    
    MCE is the maximum difference between confidence and accuracy across all bins.
    It's more sensitive to the worst-calibrated bin than ECE.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual binary outcomes
        n_bins: Number of bins
        strategy: Binning strategy - 'uniform' or 'quantile'
    
    Returns:
        MCE score (lower is better, range 0-1)
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Create bins
    if strategy == 'uniform':
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins[1:-1])
    else:  # quantile
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(predictions, quantiles * 100)
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins[1:-1])
    
    max_error = 0.0
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        
        if not np.any(mask):
            continue
        
        avg_confidence = np.mean(predictions[mask])
        accuracy = np.mean(actuals[mask])
        
        error = np.abs(accuracy - avg_confidence)
        max_error = max(max_error, error)
    
    return max_error


def adaptive_calibration_error(predictions: Union[List[float], np.ndarray],
                               actuals: Union[List[int], np.ndarray],
                               n_bins: int = 10) -> float:
    """
    Calculate Adaptive Calibration Error (ACE).
    
    ACE uses adaptive binning where bin boundaries are chosen to have equal
    number of samples, making it more robust than uniform binning.
    
    This is equivalent to ECE with quantile binning strategy.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual binary outcomes
        n_bins: Number of bins
    
    Returns:
        ACE score (lower is better)
    """
    return expected_calibration_error(predictions, actuals, n_bins, strategy='quantile')


def calibration_curve(predictions: Union[List[float], np.ndarray],
                     actuals: Union[List[int], np.ndarray],
                     n_bins: int = 10,
                     strategy: str = 'uniform') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data for plotting reliability diagrams.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual binary outcomes
        n_bins: Number of bins
        strategy: Binning strategy - 'uniform' or 'quantile'
    
    Returns:
        Tuple of (mean_predicted_probs, fraction_positives, bin_counts)
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Create bins
    if strategy == 'uniform':
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins[1:-1])
    else:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(predictions, quantiles * 100)
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins[1:-1])
    
    mean_predicted = []
    fraction_positive = []
    counts = []
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        
        if not np.any(mask):
            continue
        
        mean_predicted.append(np.mean(predictions[mask]))
        fraction_positive.append(np.mean(actuals[mask]))
        counts.append(np.sum(mask))
    
    return np.array(mean_predicted), np.array(fraction_positive), np.array(counts)


def plot_reliability_diagram(predictions: Union[List[float], np.ndarray],
                            actuals: Union[List[int], np.ndarray],
                            n_bins: int = 10,
                            strategy: str = 'uniform',
                            save_path: Optional[str] = None,
                            show_histogram: bool = True) -> plt.Figure:
    """
    Create a reliability diagram (calibration plot).
    
    A reliability diagram plots predicted probability vs observed frequency.
    Perfect calibration appears as a diagonal line.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual binary outcomes
        n_bins: Number of bins
        strategy: Binning strategy
        save_path: Optional path to save the plot
        show_histogram: Whether to show histogram of predictions
    
    Returns:
        matplotlib Figure object
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate ECE for the title
    ece = expected_calibration_error(predictions, actuals, n_bins, strategy)
    mce = maximum_calibration_error(predictions, actuals, n_bins, strategy)
    
    # Get calibration curve data
    mean_pred, frac_pos, counts = calibration_curve(predictions, actuals, n_bins, strategy)
    
    # Create figure
    if show_histogram:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Main calibration plot
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax1.plot(mean_pred, frac_pos, 'o-', markersize=8, linewidth=2, 
             label=f'Model (ECE={ece:.4f})', color='#2E86AB')
    
    # Add gap lines to show calibration error
    for mp, fp in zip(mean_pred, frac_pos):
        ax1.plot([mp, mp], [mp, fp], 'r-', alpha=0.3, linewidth=1)
    
    # Formatting
    ax1.set_xlabel('Predicted Probability (Confidence)', fontsize=12)
    ax1.set_ylabel('Observed Frequency (Accuracy)', fontsize=12)
    ax1.set_title(f'Reliability Diagram\nECE: {ece:.4f}, MCE: {mce:.4f}, Bins: {n_bins} ({strategy})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_aspect('equal')
    
    # Add sample count annotations
    for mp, fp, count in zip(mean_pred, frac_pos, counts):
        ax1.annotate(f'n={count}', (mp, fp), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8, alpha=0.7)
    
    # Histogram of predictions
    if show_histogram:
        ax2.hist(predictions, bins=50, edgecolor='black', alpha=0.7, color='#A23B72')
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Predictions', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reliability diagram saved to: {save_path}")
    
    return fig


def classwise_ece(predictions: Union[List[List[float]], np.ndarray],
                 actuals: Union[List[int], np.ndarray],
                 n_bins: int = 10) -> Tuple[float, List[float]]:
    """
    Calculate class-wise ECE for multi-class predictions.
    
    For each class, treats it as a binary problem (class vs not-class)
    and computes ECE.
    
    Args:
        predictions: Predicted probabilities for each class (N x K array)
        actuals: Actual class labels (0 to K-1)
    
    Returns:
        Tuple of (overall_ece, list_of_class_eces)
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    n_samples, n_classes = predictions.shape
    
    class_eces = []
    
    for class_idx in range(n_classes):
        # Binary labels: 1 if this class, 0 otherwise
        binary_actuals = (actuals == class_idx).astype(int)
        class_predictions = predictions[:, class_idx]
        
        ece = expected_calibration_error(class_predictions, binary_actuals, n_bins)
        class_eces.append(ece)
    
    # Overall ECE is the average across classes
    overall_ece = np.mean(class_eces)
    
    return overall_ece, class_eces


# Example usage and tests
if __name__ == "__main__":
    print("=" * 70)
    print("Expected Calibration Error (ECE) Calculator")
    print("=" * 70)
    
    # Example 1: Perfect calibration
    print("\n📊 Example 1: Perfect Calibration")
    np.random.seed(42)
    n = 1000
    perfect_preds = np.random.uniform(0, 1, n)
    perfect_actuals = (np.random.uniform(0, 1, n) < perfect_preds).astype(int)
    
    ece_perfect = expected_calibration_error(perfect_preds, perfect_actuals, n_bins=10)
    print(f"Number of samples: {n}")
    print(f"ECE (10 bins, uniform): {ece_perfect:.4f}")
    print("✓ Low ECE indicates good calibration")
    
    # Example 2: Overconfident model
    print("\n📊 Example 2: Overconfident Model")
    # Model predicts high confidence but is only 60% accurate
    overconfident_preds = np.random.uniform(0.7, 1.0, 500)
    overconfident_actuals = (np.random.uniform(0, 1, 500) < 0.6).astype(int)
    
    ece_over = expected_calibration_error(overconfident_preds, overconfident_actuals, n_bins=10)
    mce_over = maximum_calibration_error(overconfident_preds, overconfident_actuals, n_bins=10)
    print(f"Predicted confidence: 0.7-1.0")
    print(f"Actual accuracy: ~0.6")
    print(f"ECE: {ece_over:.4f}")
    print(f"MCE: {mce_over:.4f}")
    print("✗ High ECE indicates poor calibration (overconfident)")
    
    # Example 3: Underconfident model
    print("\n📊 Example 3: Underconfident Model")
    # Model predicts low confidence but is actually 80% accurate
    underconfident_preds = np.random.uniform(0.3, 0.6, 500)
    underconfident_actuals = (np.random.uniform(0, 1, 500) < 0.8).astype(int)
    
    ece_under = expected_calibration_error(underconfident_preds, underconfident_actuals, n_bins=10)
    print(f"Predicted confidence: 0.3-0.6")
    print(f"Actual accuracy: ~0.8")
    print(f"ECE: {ece_under:.4f}")
    print("✗ High ECE indicates poor calibration (underconfident)")
    
    # Example 4: Reranker scenario
    print("\n📊 Example 4: Reranker Scenario (Realistic)")
    np.random.seed(123)
    
    # Simulate a reranker with decent but not perfect calibration
    n_samples = 200
    
    # Generate predictions with some miscalibration
    # Relevant docs: slightly overconfident
    relevant_preds = np.clip(np.random.beta(7, 2, 100) + 0.05, 0, 1)
    # Irrelevant docs: slightly underconfident  
    irrelevant_preds = np.clip(np.random.beta(2, 7, 100) - 0.05, 0, 1)
    
    reranker_preds = np.concatenate([relevant_preds, irrelevant_preds])
    reranker_actuals = np.concatenate([np.ones(100), np.zeros(100)]).astype(int)
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    reranker_preds = reranker_preds[shuffle_idx]
    reranker_actuals = reranker_actuals[shuffle_idx]
    
    # Calculate different calibration metrics
    ece_uniform = expected_calibration_error(reranker_preds, reranker_actuals, 
                                            n_bins=10, strategy='uniform')
    ece_quantile = expected_calibration_error(reranker_preds, reranker_actuals,
                                             n_bins=10, strategy='quantile')
    ace = adaptive_calibration_error(reranker_preds, reranker_actuals, n_bins=10)
    mce = maximum_calibration_error(reranker_preds, reranker_actuals, n_bins=10)
    
    print(f"Query-document pairs: {n_samples}")
    print(f"Relevant: {np.sum(reranker_actuals)}, Irrelevant: {n_samples - np.sum(reranker_actuals)}")
    print(f"\nCalibration Metrics:")
    print(f"  ECE (uniform bins):  {ece_uniform:.4f}")
    print(f"  ECE (quantile bins): {ece_quantile:.4f}")
    print(f"  ACE:                 {ace:.4f}")
    print(f"  MCE:                 {mce:.4f}")
    
    # Example 5: Comparing uniform vs quantile binning
    print("\n📊 Example 5: Impact of Binning Strategy")
    # Create skewed predictions (most predictions are low)
    skewed_preds = np.concatenate([
        np.random.uniform(0, 0.3, 800),
        np.random.uniform(0.7, 1.0, 200)
    ])
    skewed_actuals = np.concatenate([
        (np.random.uniform(0, 1, 800) < 0.2).astype(int),
        (np.random.uniform(0, 1, 200) < 0.85).astype(int)
    ])
    
    ece_uniform_skewed = expected_calibration_error(skewed_preds, skewed_actuals,
                                                    n_bins=10, strategy='uniform')
    ece_quantile_skewed = expected_calibration_error(skewed_preds, skewed_actuals,
                                                     n_bins=10, strategy='quantile')
    
    print(f"Skewed predictions (80% low confidence, 20% high confidence)")
    print(f"  ECE (uniform):  {ece_uniform_skewed:.4f}")
    print(f"  ECE (quantile): {ece_quantile_skewed:.4f}")
    print("  → Quantile binning often better for skewed distributions")
    
    # Example 6: Multi-class ECE
    print("\n📊 Example 6: Multi-class ECE (Graded Relevance)")
    # 3 classes: not relevant (0), relevant (1), highly relevant (2)
    n_multi = 150
    multi_preds = np.random.dirichlet(np.ones(3), n_multi)
    multi_actuals = np.random.choice([0, 1, 2], n_multi, p=[0.5, 0.3, 0.2])
    
    overall_ece, class_eces = classwise_ece(multi_preds, multi_actuals, n_bins=10)
    
    print(f"Samples: {n_multi}, Classes: 3 (not relevant, relevant, highly relevant)")
    print(f"Overall ECE: {overall_ece:.4f}")
    print(f"Class-wise ECE:")
    for i, ece in enumerate(class_eces):
        print(f"  Class {i}: {ece:.4f}")
    
    # Generate reliability diagram
    print("\n📊 Generating Reliability Diagram...")
    fig = plot_reliability_diagram(
        reranker_preds, 
        reranker_actuals,
        n_bins=10,
        strategy='uniform',
        save_path='/home/claude/reliability_diagram.png',
        show_histogram=True
    )
    plt.close(fig)
    
    print("\n" + "=" * 70)
    print("✅ Interpretation Guide:")
    print("  • ECE ranges from 0 (perfect) to 1 (worst)")
    print("  • ECE < 0.05 is excellent calibration")
    print("  • ECE < 0.10 is good calibration")
    print("  • ECE > 0.15 indicates significant miscalibration")
    print("  • MCE shows worst-case bin (more pessimistic than ECE)")
    print("  • Use quantile binning for skewed prediction distributions")
    print("  • Reliability diagrams visualize calibration quality")
    print("=" * 70)