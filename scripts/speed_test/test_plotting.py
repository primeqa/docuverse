#!/usr/bin/env python3
"""
Test script to verify plotting functionality.
Creates mock benchmark results and generates all plots.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_embedding_timing import create_plots

# Create mock benchmark results
print("Creating mock benchmark results...")
mock_results = []

# Simulate API results
for batch_size in [1, 8, 16, 32]:
    mock_results.append({
        "name": "API",
        "batch_size": batch_size,
        "num_samples": 100,
        "total_time": 50.0 / batch_size,  # Simulated: larger batches are faster
        "avg_batch_time": 0.5 / batch_size,
        "std_batch_time": 0.05,
        "throughput": 100.0 / (50.0 / batch_size),
        "avg_latency_ms": (50.0 / batch_size) / 100 * 1000,
        "embedding_dim": 768
    })

# Simulate GPU results (faster than API)
for batch_size in [1, 8, 16, 32]:
    mock_results.append({
        "name": "GPU",
        "batch_size": batch_size,
        "num_samples": 100,
        "total_time": 15.0 / batch_size,  # GPU is ~3x faster
        "avg_batch_time": 0.15 / batch_size,
        "std_batch_time": 0.02,
        "throughput": 100.0 / (15.0 / batch_size),
        "avg_latency_ms": (15.0 / batch_size) / 100 * 1000,
        "embedding_dim": 768
    })

print(f"Created {len(mock_results)} mock results")

# Create plots
output_dir = "test_plots"
print(f"\nGenerating plots to {output_dir}/...")
print("="*80)

try:
    create_plots(mock_results, output_dir, plot_format="png")
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"\nPlots generated in: {output_dir}/")
    print("\nGenerated files:")
    for filename in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  ✓ {filename} ({size_kb:.1f} KB)")

except Exception as e:
    print(f"\n✗ Error generating plots: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
