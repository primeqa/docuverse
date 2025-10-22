"""
Visualization tools for probability calibration

This module provides functions to create reliability diagrams, 
calibration curves, and other visualizations to understand calibration quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from probability_calibration import (
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
    
    # Plot calibration curve
    if len(data['bin_centers']) > 0:
        ax.plot(data['confidences'], data['accuracies'], 'o-', 
               linewidth=2, markersize=8, label='Model')
        
        # Add bar chart showing sample count
        ax2 = ax.twinx()
        ax2.bar(data['bin_centers'], data['counts'], alpha=0.3, 
               width=1.0/n_bins, color='gray', label='Count')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.legend(loc='upper left')
    
    ax.set_xlabel('Confidence (Predicted Probability)', fontsize=12)
    ax.set_ylabel('Accuracy (Fraction of Positives)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    
    return ax


def plot_calibration_comparison(probs, labels, n_bins=10, figsize=(16, 5)):
    """
    Compare calibration before and after using different methods
    
    Args:
        probs: Predicted probabilities
        labels: True labels  
        n_bins: Number of bins for visualization
        figsize: Figure size
    """
    calibrator = ProbabilityCalibrator(n_bins=15)
    
    # Calibrate using each method
    methods = ['temperature_scaling', 'platt_scaling', 'isotonic']
    calibrated_probs = {}
    
    for method in methods:
        calibrated_probs[method] = calibrator.methods[method].fit_transform(probs, labels)
    
    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Calculate ECE for each
    ece_calc = ECECalculator(n_bins=15)
    
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


def create_full_calibration_report(probs, labels, save_path=None):
    """
    Create a comprehensive calibration report with all visualizations
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        save_path: Path to save the figure (optional)
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Calculate ECE
    ece_calc = ECECalculator(n_bins=15)
    original_ece = ece_calc.calculate(probs, labels)
    
    # Title
    fig.suptitle(f'Comprehensive Calibration Report\nOriginal ECE: {original_ece:.4f}', 
                fontsize=16, fontweight='bold')
    
    # 1. Original reliability diagram
    ax1 = fig.add_subplot(gs[0, 0])
    plot_reliability_diagram(probs, labels, n_bins=10, title="Original", ax=ax1)
    
    # Calibrate with different methods
    calibrator = ProbabilityCalibrator(n_bins=15)
    methods = {
        'Temperature Scaling': 'temperature_scaling',
        'Platt Scaling': 'platt_scaling', 
        'Isotonic Regression': 'isotonic'
    }
    
    # Plot calibrated versions
    positions = [(0, 1), (0, 2), (1, 0)]
    for (title, method), pos in zip(methods.items(), positions):
        cal_probs = calibrator.methods[method].fit_transform(probs, labels)
        cal_ece = ece_calc.calculate(cal_probs, labels)
        improvement = (original_ece - cal_ece) / original_ece * 100
        
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        plot_reliability_diagram(cal_probs, labels, n_bins=10,
                                title=f"{title}\nECE: {cal_ece:.4f} ({improvement:+.1f}%)",
                                ax=ax)
    
    # ECE bin visualization for original
    ax5 = fig.add_subplot(gs[1, 1:])
    plot_ece_bins(probs, labels, n_bins=15, ax=ax5)
    
    # Distribution histograms
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax6.set_xlabel('Predicted Probability', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Prediction Distribution', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Class-wise distributions
    ax7 = fig.add_subplot(gs[2, 1])
    labels_bool = labels.astype(bool)
    ax7.hist([probs[~labels_bool], probs[labels_bool]], 
            bins=30, alpha=0.7, label=['Class 0', 'Class 1'],
            color=['red', 'green'], edgecolor='black')
    ax7.set_xlabel('Predicted Probability', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.set_title('By True Class', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # Summary statistics table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    stats_text = f"""
    Dataset Statistics:
    ──────────────────
    Samples: {len(probs)}
    Positive Rate: {labels.mean():.3f}
    Mean Prediction: {probs.mean():.3f}
    Std Prediction: {probs.std():.3f}
    
    Original ECE: {original_ece:.4f}
    
    Best Method: 
    {calibrator.best_method if hasattr(calibrator, 'best_method') else 'Not calculated'}
    """
    
    ax8.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
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
    
    print(f"Generated {n_samples} samples")
    
    # Create visualizations
    print("\n1. Creating reliability diagram...")
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    plot_reliability_diagram(raw_probs, true_labels, n_bins=10, ax=ax1)
    plt.savefig('/home/claude/reliability_diagram.png', dpi=300, bbox_inches='tight')
    print("   Saved to: reliability_diagram.png")
    
    print("\n2. Creating calibration comparison...")
    fig2 = plot_calibration_comparison(raw_probs, true_labels, n_bins=10)
    plt.savefig('/home/claude/calibration_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved to: calibration_comparison.png")
    
    print("\n3. Creating ECE bin visualization...")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    plot_ece_bins(raw_probs, true_labels, n_bins=15, ax=ax3)
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
