#!/usr/bin/env python3
"""
Brier Score Calculator for Text Encoding Reranker Evaluation

The Brier score measures the mean squared difference between predicted 
probabilities and actual outcomes. Lower scores indicate better calibration.

Brier Score = (1/N) * Σ(p_i - y_i)²

where:
- N is the number of predictions
- p_i is the predicted probability
- y_i is the actual outcome (0 or 1)
"""

import numpy as np
from typing import List, Union, Tuple


def brier_score(predictions: Union[List[float], np.ndarray], 
                actuals: Union[List[int], np.ndarray]) -> float:
    """
    Calculate the Brier score for binary predictions.
    
    Args:
        predictions: Predicted probabilities (values between 0 and 1)
        actuals: Actual binary outcomes (0 or 1)
    
    Returns:
        Brier score (lower is better, range 0-1)
    
    Raises:
        ValueError: If predictions and actuals have different lengths,
                   or if values are out of valid ranges
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Validation
    if len(predictions) != len(actuals):
        raise ValueError(f"Length mismatch: predictions ({len(predictions)}) "
                        f"vs actuals ({len(actuals)})")
    
    if not np.all((predictions >= 0) & (predictions <= 1)):
        raise ValueError("Predictions must be between 0 and 1")
    
    if not np.all((actuals == 0) | (actuals == 1)):
        raise ValueError("Actuals must be binary (0 or 1)")
    
    # Calculate Brier score
    return np.mean((predictions - actuals) ** 2)


def brier_score_multiclass(predictions: Union[List[List[float]], np.ndarray],
                          actuals: Union[List[int], np.ndarray]) -> float:
    """
    Calculate the Brier score for multi-class predictions.
    
    Args:
        predictions: Predicted probabilities for each class (N x K array)
        actuals: Actual class labels (0 to K-1)
    
    Returns:
        Brier score (lower is better)
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    n_samples, n_classes = predictions.shape
    
    if len(actuals) != n_samples:
        raise ValueError(f"Length mismatch: {n_samples} predictions "
                        f"vs {len(actuals)} actuals")
    
    # Convert actuals to one-hot encoding
    actuals_one_hot = np.zeros((n_samples, n_classes))
    actuals_one_hot[np.arange(n_samples), actuals] = 1
    
    # Calculate Brier score
    return np.mean(np.sum((predictions - actuals_one_hot) ** 2, axis=1))


def brier_skill_score(predictions: Union[List[float], np.ndarray],
                      actuals: Union[List[int], np.ndarray],
                      baseline_predictions: Union[List[float], np.ndarray] = None) -> float:
    """
    Calculate the Brier Skill Score, which measures improvement over a baseline.
    
    BSS = 1 - (BS_forecast / BS_baseline)
    
    A BSS of 1 means perfect predictions, 0 means same as baseline,
    negative means worse than baseline.
    
    Args:
        predictions: Model predicted probabilities
        actuals: Actual binary outcomes
        baseline_predictions: Baseline probabilities (if None, uses climatology: mean of actuals)
    
    Returns:
        Brier Skill Score
    """
    actuals = np.array(actuals)
    
    # Calculate model Brier score
    bs_model = brier_score(predictions, actuals)
    
    # Calculate baseline Brier score
    if baseline_predictions is None:
        # Climatology baseline: constant probability equal to the mean
        climatology = np.mean(actuals)
        baseline_predictions = np.full_like(actuals, climatology, dtype=float)
    
    bs_baseline = brier_score(baseline_predictions, actuals)
    
    # Calculate skill score
    return 1 - (bs_model / bs_baseline)


def decompose_brier_score(predictions: Union[List[float], np.ndarray],
                          actuals: Union[List[int], np.ndarray],
                          n_bins: int = 10) -> Tuple[float, float, float]:
    """
    Decompose Brier score into reliability, resolution, and uncertainty components.
    
    Brier Score = Reliability - Resolution + Uncertainty
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual binary outcomes
        n_bins: Number of bins for grouping predictions
    
    Returns:
        Tuple of (reliability, resolution, uncertainty)
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bins[1:-1])
    
    # Overall outcome rate
    o_bar = np.mean(actuals)
    
    reliability = 0.0
    resolution = 0.0
    
    for i in range(n_bins):
        # Get predictions and actuals in this bin
        mask = bin_indices == i
        
        if not np.any(mask):
            continue
        
        n_k = np.sum(mask)
        o_k = np.mean(actuals[mask])  # Observed frequency in bin
        f_k = np.mean(predictions[mask])  # Mean forecast in bin
        
        # Reliability: how far forecasts deviate from observed frequencies
        reliability += (n_k / len(predictions)) * (f_k - o_k) ** 2
        
        # Resolution: how much bins differ from overall rate
        resolution += (n_k / len(predictions)) * (o_k - o_bar) ** 2
    
    # Uncertainty: variance of outcomes
    uncertainty = o_bar * (1 - o_bar)
    
    return reliability, resolution, uncertainty


# Example usage and tests
if __name__ == "__main__":
    print("=" * 70)
    print("Brier Score Calculator for Reranker Evaluation")
    print("=" * 70)
    
    # Example 1: Perfect predictions
    print("\n📊 Example 1: Perfect Predictions")
    predictions_perfect = [0.0, 0.0, 1.0, 1.0, 1.0]
    actuals_perfect = [0, 0, 1, 1, 1]
    bs_perfect = brier_score(predictions_perfect, actuals_perfect)
    print(f"Predictions: {predictions_perfect}")
    print(f"Actuals:     {actuals_perfect}")
    print(f"Brier Score: {bs_perfect:.4f} (perfect calibration)")
    
    # Example 2: Random/uncalibrated predictions
    print("\n📊 Example 2: Poorly Calibrated Predictions")
    predictions_poor = [0.5, 0.5, 0.5, 0.5, 0.5]
    actuals_poor = [0, 0, 1, 1, 1]
    bs_poor = brier_score(predictions_poor, actuals_poor)
    print(f"Predictions: {predictions_poor}")
    print(f"Actuals:     {actuals_poor}")
    print(f"Brier Score: {bs_poor:.4f} (poor calibration)")
    
    # Example 3: Overconfident predictions
    print("\n📊 Example 3: Overconfident Predictions")
    predictions_over = [0.1, 0.2, 0.9, 0.95, 0.85]
    actuals_over = [0, 1, 1, 0, 1]
    bs_over = brier_score(predictions_over, actuals_over)
    print(f"Predictions: {predictions_over}")
    print(f"Actuals:     {actuals_over}")
    print(f"Brier Score: {bs_over:.4f}")
    
    # Example 4: Reranker scenario - document relevance prediction
    print("\n📊 Example 4: Reranker Scenario (Query-Document Pairs)")
    print("Simulating a reranker predicting document relevance probabilities")
    
    # Simulate 20 query-document pairs
    np.random.seed(42)
    n_samples = 20
    
    # Generate some realistic reranker predictions
    # Relevant docs (label=1) tend to get higher scores
    relevant_preds = np.random.beta(8, 2, size=10)  # Skewed toward high probs
    irrelevant_preds = np.random.beta(2, 8, size=10)  # Skewed toward low probs
    
    reranker_predictions = np.concatenate([relevant_preds, irrelevant_preds])
    reranker_actuals = np.concatenate([np.ones(10), np.zeros(10)]).astype(int)
    
    # Shuffle to mix relevant and irrelevant
    shuffle_idx = np.random.permutation(n_samples)
    reranker_predictions = reranker_predictions[shuffle_idx]
    reranker_actuals = reranker_actuals[shuffle_idx]
    
    bs_reranker = brier_score(reranker_predictions, reranker_actuals)
    print(f"Number of query-document pairs: {n_samples}")
    print(f"Relevant documents: {np.sum(reranker_actuals)}")
    print(f"Irrelevant documents: {n_samples - np.sum(reranker_actuals)}")
    print(f"Brier Score: {bs_reranker:.4f}")
    
    # Calculate Brier Skill Score
    bss = brier_skill_score(reranker_predictions, reranker_actuals)
    print(f"Brier Skill Score: {bss:.4f} (vs climatology baseline)")
    
    # Decompose the Brier score
    print("\n📊 Brier Score Decomposition:")
    reliability, resolution, uncertainty = decompose_brier_score(
        reranker_predictions, reranker_actuals, n_bins=5
    )
    print(f"Reliability: {reliability:.4f} (lower is better)")
    print(f"Resolution:  {resolution:.4f} (higher is better)")
    print(f"Uncertainty: {uncertainty:.4f} (inherent in data)")
    print(f"Brier Score: {reliability - resolution + uncertainty:.4f}")
    print(f"  = {reliability:.4f} - {resolution:.4f} + {uncertainty:.4f}")
    
    # Example 5: Multi-class scenario (e.g., relevance grades: 0=not relevant, 1=relevant, 2=highly relevant)
    print("\n📊 Example 5: Multi-class Brier Score (Relevance Grades)")
    predictions_multi = [
        [0.7, 0.2, 0.1],  # Predicts class 0
        [0.1, 0.6, 0.3],  # Predicts class 1
        [0.05, 0.15, 0.8],  # Predicts class 2
        [0.5, 0.3, 0.2],  # Predicts class 0
    ]
    actuals_multi = [0, 1, 2, 1]  # True classes
    
    bs_multi = brier_score_multiclass(predictions_multi, actuals_multi)
    print(f"Predicted probabilities shape: {np.array(predictions_multi).shape}")
    print(f"Multi-class Brier Score: {bs_multi:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Interpretation Guide:")
    print("  • Brier Score ranges from 0 (perfect) to 1 (worst)")
    print("  • Score < 0.25 is generally considered well-calibrated")
    print("  • Compare models: lower Brier score = better calibration")
    print("  • Brier Skill Score > 0 means better than baseline")
    print("=" * 70)