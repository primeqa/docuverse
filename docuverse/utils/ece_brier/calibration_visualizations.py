"""
Visualization tools for probability calibration

This module provides functions to create reliability diagrams, 
calibration curves, and other visualizations to understand calibration quality.
"""
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec

from docuverse.utils.ece_brier.probability_calibration import (
    ProbabilityCalibrator, 
    ECECalculator,
    create_reliability_diagram_data
)


def plot_reliability_diagram(probs, labels, n_bins=10, title="Reliability Diagram", ax=None):
    """
    Plot a reliability diagram (calibration curve)
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins
        title: Plot title
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get bin data
    data = create_reliability_diagram_data(probs, labels, n_bins)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax.set_ylim([0, 1])
    # Plot calibration curve
    if len(data['bin_centers']) > 0:
        ax.plot(data['confidences'], data['accuracies'], 'o-', 
               linewidth=2, markersize=8, label='Model')
        
        # Add bar chart showing sample count
        ax2 = ax.twinx()
        ax2.bar(data['bin_centers'], data['counts'], alpha=0.3, 
               width=1.0/n_bins, color='gray', label='Count')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_yscale('log')
        ax2.legend(loc='upper left')
    
    ax.set_xlabel('Confidence (Predicted Probability)', fontsize=12)
    ax.set_ylabel('Accuracy (Fraction of Positives)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    # ax.set_xlim([0, 1])
    # ax.set_ylim(ymin=-0.1, ymax=1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    
    return ax


def plot_calibration_comparison(probs, labels, cal_probs=None, method="auto", n_bins=10, figsize=(16, 5)):
    """
    Plots a comparison of calibration methods using reliability diagrams and Expected Calibration Error (ECE) metrics.

    This function visualizes the calibration performance of various methods, including the original uncalibrated
    model, using reliability diagrams. It computes the ECE for each calibration and displays any improvement
    provided by the specified calibration techniques. The function supports optional pre-calibrated probabilities.

    Args:
        probs (numpy.ndarray): The predicted probabilities from the model, before calibration.
        labels (numpy.ndarray): The true labels associated with the predicted probabilities.
        cal_probs (dict | numpy.ndarray | None): Pre-calibrated probabilities grouped by their respective
            calibration method. If None, calibration will be performed using default methods.
        method (str): Specifies the calibration method when cal_probs is not provided. Default is "auto".
        n_bins (int): The number of bins to use when computing reliability diagrams. Default is 10.
        figsize (tuple): The size of the figure for the reliability diagram plots. Default is (16, 5).

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the calibration comparison plots.
    """
    calibrated_probs = {}
    if cal_probs is None:
        calibrator = ProbabilityCalibrator(n_bins=n_bins)

        # Calibrate using each method
        methods = ['temperature_scaling', 'platt_scaling', 'isotonic']

        for method in methods:
            calibrated_probs[method] = calibrator.methods[method].fit_transform(probs, labels)
    elif isinstance(cal_probs, dict):
        calibrated_probs = cal_probs
        methods = cal_probs.keys()
    else:
        calibrated_probs = {method: cal_probs.copy()}
        methods = {method}
    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Calculate ECE for each
    ece_calc = ECECalculator(n_bins=n_bins)
    
    # Original
    original_ece = ece_calc.calculate(probs, labels)
    plot_reliability_diagram(probs, labels, n_bins, 
                            f"Original\nECE: {original_ece:.4f}", axes[0])
    
    # Calibrated versions
    for idx, method in enumerate(methods, 1):
        calibrated = calibrated_probs[method]
        ece = ece_calc.calculate(calibrated, labels)
        improvement = (original_ece - ece) / original_ece * 100
        
        method_title = method.replace('_', ' ').title()
        plot_reliability_diagram(calibrated, labels, n_bins,
                                f"{method_title}\nECE: {ece:.4f} ({improvement:+.1f}%)",
                                axes[idx])
    
    plt.tight_layout()
    return fig


def plot_ece_bins(probs, labels, n_bins=15, ax=None):
    """
    Visualize ECE calculation by showing error in each bin
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    ece_calc = ECECalculator(n_bins=n_bins)
    ece, bin_details = ece_calc.calculate(probs, labels, return_details=True)
    
    if len(bin_details) == 0:
        ax.text(0.5, 0.5, 'No data in bins', ha='center', va='center')
        return ax
    
    # Extract data
    bin_centers = [(d['range'][0] + d['range'][1]) / 2 for d in bin_details]
    errors = [d['error'] for d in bin_details]
    counts = [d['count'] for d in bin_details]
    confidences = [d['confidence'] for d in bin_details]
    accuracies = [d['accuracy'] for d in bin_details]
    
    # Normalize counts for coloring
    max_count = max(counts) if counts else 1
    colors = plt.cm.Blues([c / max_count for c in counts])
    
    # Plot bars
    bars = ax.bar(bin_centers, errors, width=1.0/n_bins * 0.8, 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Confidence Bin', fontsize=12)
    ax.set_ylabel('|Confidence - Accuracy|', fontsize=12)
    ax.set_title(f'ECE Bin Visualization (Total ECE: {ece:.4f})', 
                fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text showing bin details
    for i, (center, error, count, conf, acc) in enumerate(
        zip(bin_centers, errors, counts, confidences, accuracies)):
        if error > 0.02:  # Only show text for significant bins
            ax.text(center, error + 0.01, f'{count}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                               norm=plt.Normalize(vmin=0, vmax=max_count))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Sample Count', rotation=270, labelpad=20)
    
    return ax


def plot_calibration_curve_with_histogram(probs, labels, n_bins=10, figsize=(12, 8)):
    """
    Create a comprehensive calibration visualization with histogram
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Main reliability diagram
    ax1 = fig.add_subplot(gs[0, :])
    plot_reliability_diagram(probs, labels, n_bins, 
                            "Reliability Diagram with Sample Distribution", ax1)
    
    # Histogram of predicted probabilities
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Predictions', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Histogram by class
    ax3 = fig.add_subplot(gs[1, 1])
    labels_bool = labels.astype(bool)
    ax3.hist([probs[~labels_bool], probs[labels_bool]], 
            bins=30, alpha=0.7, label=['Negative Class', 'Positive Class'],
            color=['red', 'green'], edgecolor='black')
    ax3.set_xlabel('Predicted Probability', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Predictions by True Class', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    return fig


def create_full_calibration_report(probs, labels, calibration_data=None, calibrator=None, save_path=None, n_bins=10):
    """
    Generates a comprehensive calibration report visualizing the reliability of predicted
    probabilities and the effects of various calibration methods.

    The function creates a multi-panel figure displaying the original reliability diagram,
    ECE bin visualization, calibrated versions using several methods, prediction distribution
    histograms, and dataset statistics. It also supports saving the visualization to disk.

    Args:
        probs: numpy.ndarray. Predicted probabilities for the dataset.
        labels: numpy.ndarray. Ground truth labels corresponding to the predicted probabilities.
        calibration_data: Optional[dict]. Precomputed calibration data containing probabilities
            and ECE values for different calibration methods. Default is None.
        calibrator: Optional[ProbabilityCalibrator]. Instance of a probability calibration class to apply
            calibration methods. Default is None.
        save_path: Optional[str]. File path to save the generated calibration report.
            Default is None.
        n_bins: int. Number of bins to use in the reliability diagrams and ECE calculation.
            Default is 10.

    Returns:
        matplotlib.figure.Figure. The generated calibration report figure.
    """

    def plot_class_distribution(method: str, probs,
                                fig: Figure, sb_plot: SubplotSpec,
                                labels_bool):
        ax8 = fig.add_subplot(sb_plot)

        ax8.hist([probs[~labels_bool], probs[labels_bool]],
                 bins=30, alpha=0.7, label=['Class 0', 'Class 1'],
                 color=['red', 'green'], edgecolor='black')
        ax8.set_xlabel('Predicted Probability', fontsize=10)
        ax8.set_ylabel('Frequency', fontsize=10)
        ax8.set_title(f'{method} calibrated, By True Class', fontsize=11, fontweight='bold')
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)

    def plot_ece_diagram(calibration_data, calibrator,
                         compute_calibration: bool,
                         ece_calc: ECECalculator,
                         fig: Figure,
                         ss: SubplotSpec,
                         labels, method: str,
                         original_ece: tuple[float, list[Any]] | float,
                         probs,
                         title: str):
        if compute_calibration:
            cal_probs = calibrator.methods[method].fit_transform(probs, labels)
            cal_ece = ece_calc.calculate(cal_probs, labels)
        else:
            cal_probs = calibration_data['probs'][method]
            cal_ece = calibration_data['ece'][method]

        improvement = (original_ece - cal_ece) / original_ece * 100

        ax = fig.add_subplot(ss)
        ax = plot_reliability_diagram(cal_probs, labels, n_bins=n_bins,
                                      title=f"{title}\nECE: {cal_ece:.4f} ({improvement:+.1f}%)",
                                      ax=ax)

    # Create figure with subplots
    fig = plt.figure(figsize=(40, 40))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # Calculate ECE
    ece_calc = ECECalculator(n_bins=n_bins)
    original_ece = ece_calc.calculate(probs, labels)
    
    # Title
    fig.suptitle(f'Comprehensive Calibration Report\nOriginal ECE: {original_ece:.4f}', 
                fontsize=16, fontweight='bold')
    
    # 1. Original reliability diagram
    ax1 = fig.add_subplot(gs[0, 0])
    plot_reliability_diagram(probs, labels, n_bins=n_bins, title="Original", ax=ax1)
    
    # Calibrate with different methods
    compute_calibration = False
    if calibration_data is None:
        compute_calibration = True
        calibrator = ProbabilityCalibrator(n_bins=n_bins)

    methods = {
        'Original': 'original',
        'Temperature Scaling': 'temperature_scaling',
        'Platt Scaling': 'platt_scaling', 
        'Isotonic Regression': 'isotonic'
    }
    calibration_data['probs']['original'] = probs
    calibration_data['ece']['original'] = original_ece

    # Plot calibrated versions
    positions = [(0, 0), (1, 0), (2, 0), (3, 0)]
    for (title, method), pos in zip(methods.items(), positions):
        plot_ece_diagram(calibration_data, calibrator, compute_calibration, ece_calc, fig,
                         gs[pos[0], pos[1]], labels, method,
                         original_ece, probs, title)

    # ECE bin visualization for original
    positions = [(0,2), (1, 2), (2, 2), (3, 2)]
    for (_, method), pos in zip(methods.items(), positions):
        ax5 = fig.add_subplot(gs[pos[0], pos[1]:])
        plot_ece_bins(calibration_data['probs'][method], labels, n_bins=n_bins, ax=ax5)
    
    # Distribution histograms
    # ax6 = fig.add_subplot(gs[2, 0])
    # ax6.hist(probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    # ax6.set_xlabel('Predicted Probability', fontsize=10)
    # ax6.set_ylabel('Frequency', fontsize=10)
    # ax6.set_title('Prediction Distribution', fontsize=11, fontweight='bold')
    # ax6.grid(True, alpha=0.3)
    
    # Class-wise distributions

    labels_bool = labels.astype(bool)
    if calibrator is not None and hasattr(calibrator, 'best_method'):
        # best_method = calibrator.best_method
        best_method = 'platt_scaling'
        best_probs = calibration_data['probs'][best_method]
    else:
        best_probs = probs
        best_method = "Original"

    # ax7.hist([probs[~labels_bool], probs[labels_bool]],
    #         bins=30, alpha=0.7, label=['Class 0', 'Class 1'],
    #         color=['red', 'green'], edgecolor='black')
    # ax7.set_xlabel('Predicted Probability', fontsize=10)
    # ax7.set_ylabel('Frequency', fontsize=10)
    # ax7.set_title('Uncalibrated, By True Class', fontsize=11, fontweight='bold')
    # ax7.legend(fontsize=9)
    # ax7.grid(True, alpha=0.3)

    positions = [(0, 1), (1, 1), (2, 1), (3, 1)]
    for (title, method), pos in zip(methods.items(), positions):
        plot_class_distribution(method, calibration_data['probs'][method], fig, gs[pos[0],pos[1]], labels_bool)
    # Summary statistics table
        # plot_class_distribution(best_method, best_probs, fig, gs[2,2], labels_bool)
    plt.tight_layout()

    # ax8.axis('off')
    #
    # stats_text = f"""
    # Dataset Statistics:
    # ──────────────────
    # Samples: {len(probs)}
    # Positive Rate: {labels.mean():.3f}
    # Mean Prediction: {probs.mean():.3f}
    # Std Prediction: {probs.std():.3f}
    #
    # Original ECE: {original_ece:.4f}
    #
    # Best Method:
    # {calibrator.best_method if hasattr(calibrator, 'best_method') else 'Not calculated'}
    # """
    #
    # ax8.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
    #         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Report saved to {save_path}")
    
    return fig




# Example usage
if __name__ == "__main__":
    print("Generating calibration visualizations...")
    
    # Generate example data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate poorly calibrated probabilities
    true_labels = np.random.binomial(1, 0.5, n_samples)
    raw_probs = true_labels * np.random.beta(8, 2, n_samples) + \
                (1 - true_labels) * np.random.beta(2, 8, n_samples)
    raw_probs = np.clip(raw_probs, 0.01, 0.99)
    n_bins = 15
    print(f"Generated {n_samples} samples")
    
    # Create visualizations
    print("\n1. Creating reliability diagram...")
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    plot_reliability_diagram(raw_probs, true_labels, n_bins=n_bins, ax=ax1)
    plt.savefig('/home/claude/reliability_diagram.png', dpi=300, bbox_inches='tight')
    print("   Saved to: reliability_diagram.png")
    
    print("\n2. Creating calibration comparison...")
    fig2 = plot_calibration_comparison(raw_probs, true_labels, n_bins=n_bins)
    plt.savefig('/home/claude/calibration_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved to: calibration_comparison.png")
    
    print("\n3. Creating ECE bin visualization...")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    plot_ece_bins(raw_probs, true_labels, n_bins=n_bins, ax=ax3)
    plt.savefig('/home/claude/ece_bins.png', dpi=300, bbox_inches='tight')
    print("   Saved to: ece_bins.png")
    
    print("\n4. Creating comprehensive report...")
    fig4 = create_full_calibration_report(raw_probs, true_labels, 
                                         save_path='/home/claude/calibration_report.png')
    print("   Saved to: calibration_report.png")
    
    print("\n✓ All visualizations created successfully!")
    print("\nVisualization files:")
    print("  - reliability_diagram.png")
    print("  - calibration_comparison.png")
    print("  - ece_bins.png")
    print("  - calibration_report.png")
