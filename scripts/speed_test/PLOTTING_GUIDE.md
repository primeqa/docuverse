# Visualization & Plotting Guide

The benchmark script now includes automatic visualization generation to help you analyze and present your results.

## Quick Start

```bash
# Add --plot flag to generate visualizations
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --field_path text \
    --batch_sizes 1,8,16,32 \
    --plot
```

This creates 5 plots in the `benchmark_plots/` directory.

## Command-Line Options

### `--plot`
Enable plot generation (default: disabled)

### `--plot_dir`
Directory to save plots (default: `benchmark_plots`)

### `--plot_format`
Output format: `png`, `pdf`, or `svg` (default: `png`)
- **PNG**: Good for presentations, web, general use
- **PDF**: Best for LaTeX papers, vector graphics
- **SVG**: Best for editing in Illustrator/Inkscape

## Generated Plots

### 1. Throughput Comparison
**File:** `throughput_comparison.png`

Bar chart comparing queries per second across batch sizes.

**Use cases:**
- Compare API vs GPU performance
- Identify optimal batch size
- Show overall performance differences

**Key features:**
- Grouped bars by batch size
- Color-coded by method (API=red, GPU=blue)
- Clear axis labels and legend

### 2. Latency Comparison
**File:** `latency_comparison.png`

Bar chart comparing average latency in milliseconds.

**Use cases:**
- Analyze response time differences
- Identify latency bottlenecks
- Choose appropriate method for time-sensitive applications

**Key features:**
- Lower is better
- Same grouping as throughput plot
- Consistent color scheme

### 3. Speedup Comparison
**File:** `speedup_comparison.png`

Bar chart showing GPU speedup factor over API.

**Use cases:**
- Quantify GPU acceleration benefits
- Justify infrastructure investments
- Show relative performance gains

**Key features:**
- Values labeled on each bar (e.g., "3.5x")
- Baseline at 1.0x (red dashed line)
- Clear speedup multipliers

### 4. Batch Size Scaling
**File:** `batch_scaling.png`

Two-panel line plot showing scaling behavior.

**Panels:**
- **Left:** Throughput vs batch size
- **Right:** Latency vs batch size

**Use cases:**
- Understand scaling characteristics
- Identify diminishing returns
- Optimize batch size selection

**Key features:**
- Log scale on X-axis for better visualization
- Markers on data points
- Shows trend clearly

### 5. Summary Dashboard
**File:** `summary_dashboard.png`

Comprehensive 4-panel overview of all metrics.

**Panels:**
- **Top-left:** Throughput comparison
- **Top-right:** Latency comparison
- **Bottom-left:** Throughput scaling
- **Bottom-right:** Speedup or statistics

**Use cases:**
- Single-page overview
- Executive summaries
- Quick comparisons

**Key features:**
- All key metrics in one view
- Consistent styling across panels
- Professional appearance

## Plot Specifications

All plots are generated with:
- **Resolution:** 300 DPI (publication quality)
- **Style:** Professional seaborn theme
- **Colors:** Consistent across all plots
  - API: Red (#E74C3C)
  - GPU: Blue (#3498DB)
  - Speedup: Green (#27AE60)
- **Fonts:** Bold labels, readable sizes
- **Grid:** Subtle gridlines for readability

## Examples

### Basic Usage
```bash
python benchmark_embedding_timing.py \
    --num_samples 100 \
    --batch_sizes 1,8,16,32 \
    --plot
```

### Custom Output Directory
```bash
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --batch_sizes 1,16,32 \
    --plot \
    --plot_dir results/experiment_1
```

### PDF for Paper
```bash
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --batch_sizes 1,8,16,32 \
    --plot \
    --plot_format pdf \
    --plot_dir paper_figures
```

### SVG for Editing
```bash
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --batch_sizes 1,8,16,32 \
    --plot \
    --plot_format svg \
    --plot_dir editable_figures
```

## Workflow Integration

### Research Papers
1. Run benchmark with `--plot --plot_format pdf`
2. Use generated PDFs directly in LaTeX
3. High quality vector graphics
4. No rasterization issues

```bash
python benchmark_embedding_timing.py \
    --input_file dataset.jsonl \
    --num_samples 500 \
    --batch_sizes 1,8,16,32,64 \
    --plot \
    --plot_format pdf \
    --plot_dir paper/figures
```

### Presentations
1. Run benchmark with `--plot --plot_format png`
2. Insert PNGs into PowerPoint/Google Slides
3. High resolution prevents pixelation

```bash
python benchmark_embedding_timing.py \
    --input_file dataset.jsonl \
    --num_samples 200 \
    --batch_sizes 1,16,32 \
    --plot \
    --plot_dir presentation/slides
```

### Custom Analysis
1. Generate SVG files for editing
2. Open in Illustrator/Inkscape
3. Customize colors, labels, annotations

```bash
python benchmark_embedding_timing.py \
    --input_file dataset.jsonl \
    --batch_sizes 1,8,16,32 \
    --plot \
    --plot_format svg
```

## Progress Monitoring

During benchmarking, you'll see progress bars powered by tqdm:

```
Testing API with batch_size=16 (100 batches)...
  API: 100%|██████████| 100/100 [00:45<00:00,  2.2batch/s, throughput=3.5 q/s]
```

**Progress bar shows:**
- Current method being tested
- Batch progress (X/Y)
- Elapsed time
- Estimated time remaining
- Current throughput

## Output Structure

After running with `--plot`, your directory will look like:

```
benchmark_plots/
├── throughput_comparison.png
├── latency_comparison.png
├── speedup_comparison.png
├── batch_scaling.png
└── summary_dashboard.png
```

## Tips

### Best Practices

1. **Use consistent seeds** for reproducible plots:
   ```bash
   --random_seed 42
   ```

2. **Test multiple batch sizes** for better scaling plots:
   ```bash
   --batch_sizes 1,2,4,8,16,32,64
   ```

3. **Use PDF for papers** to avoid quality loss:
   ```bash
   --plot_format pdf
   ```

4. **Create separate directories** for different experiments:
   ```bash
   --plot_dir experiments/exp_001
   ```

### Common Scenarios

**Scenario:** Need plots for a paper
```bash
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --num_samples 500 \
    --batch_sizes 1,4,8,16,32,64 \
    --random_seed 42 \
    --plot \
    --plot_format pdf \
    --plot_dir paper_submission/figures
```

**Scenario:** Quick visual check
```bash
python benchmark_embedding_timing.py \
    --num_samples 50 \
    --batch_sizes 1,16,32 \
    --skip_api \
    --plot
```

**Scenario:** Compare two models
```bash
# Run 1: Model A
python benchmark_embedding_timing.py \
    --local_model_name model-a \
    --skip_api \
    --plot \
    --plot_dir comparison/model_a

# Run 2: Model B
python benchmark_embedding_timing.py \
    --local_model_name model-b \
    --skip_api \
    --plot \
    --plot_dir comparison/model_b
```

## Testing

Test the plotting functionality:

```bash
python test_plotting.py
```

This creates mock results and generates all plots in `test_plots/`.

## Dependencies

Plotting requires:
- `matplotlib>=3.9.0`
- `numpy`

Already included in `requirements.txt`.

## See Also

- `BENCHMARK_README.md` - Full documentation
- `test_plotting.py` - Test plotting functionality
- `run_embedding_benchmark.sh` - Example scripts
