# Embedding Timing Benchmark

Comprehensive benchmark script to compare OpenAI API-based embeddings vs GPU-based embeddings.

## Features

- ✅ Compare OpenAI-compatible API (RITS) vs local GPU inference
- ✅ Test multiple batch sizes (e.g., 1, 8, 16, 32)
- ✅ Load texts from JSONL or JSONL.bz2 files
- ✅ Support nested field extraction with dot notation (e.g., `document.text`)
- ✅ **Array support**: Extract from arrays with wildcards (`documents[*].text`) or specific indices (`documents[0].text`)
- ✅ **Random sampling**: Sample N random texts from loaded data with reproducible seeds
- ✅ **Visualization**: Automatic plot generation (throughput, latency, speedup, scaling)
- ✅ **Progress bars**: Real-time progress monitoring with tqdm
- ✅ Auto-detect common text fields (`text`, `content`, `question`)
- ✅ Detailed performance metrics (throughput, latency, speedup)
- ✅ Warmup runs to avoid cold-start penalties

## Installation

Ensure you have the required dependencies:

```bash
conda activate gma_rag_rl
# Dependencies: torch, transformers, openai, numpy
```

## Usage

### 1. With Generated Sample Texts

```bash
export RITS_API_KEY="your-api-key"

python benchmark_embedding_timing.py \
    --model_name ibm/slate-125m-english-rtrvr-v2 \
    --endpoint https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/slate-125m-english-rtrvr-v2 \
    --local_model_name ibm-granite/granite-embedding-125m-english \
    --num_samples 100 \
    --batch_sizes 1,8,16,32
```

### 2. With JSONL File (Auto-detect Text Field)

```bash
python benchmark_embedding_timing.py \
    --input_file data/my_texts.jsonl \
    --batch_sizes 1,8,16,32
```

The script will automatically try these fields: `text`, `content`, `question`

### 3. With Custom Field Path

For nested JSON structures, use dot notation:

```bash
# Example: {"document": {"text": "..."}}
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --field_path document.text \
    --batch_sizes 1,16,32
```

### 4. With NQ Dataset

```bash
python benchmark_embedding_timing.py \
    --input_file /proj/rh-inf-scaling/aashka/rag_rl/data/evals/nq-fixed/nq-dev-500-fixed.jsonl \
    --field_path question \
    --max_samples 200 \
    --batch_sizes 1,16,32
```

### 5. With Compressed Files

```bash
python benchmark_embedding_timing.py \
    --input_file data.jsonl.bz2 \
    --field_path text \
    --max_samples 500
```

### 6. GPU Only (Skip API)

```bash
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --skip_api \
    --batch_sizes 1,16,32,64
```

### 7. Random Sampling from Large File

```bash
# Load all texts from file, then randomly sample 200
python benchmark_embedding_timing.py \
    --input_file large_dataset.jsonl \
    --field_path text \
    --num_samples 200 \
    --random_seed 42 \
    --batch_sizes 1,16,32
```

### 8. Combining max_samples and num_samples

```bash
# Read first 1000 lines, then randomly sample 200 from those
python benchmark_embedding_timing.py \
    --input_file huge_dataset.jsonl \
    --field_path text \
    --max_samples 1000 \
    --num_samples 200 \
    --batch_sizes 1,16,32
```

### 9. Generate Plots

```bash
# Run benchmark and generate visualization plots
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --field_path text \
    --num_samples 200 \
    --batch_sizes 1,8,16,32 \
    --plot \
    --plot_dir my_plots \
    --plot_format png
```

## Understanding max_samples vs num_samples

These two options serve different purposes:

### `--max_samples`
- **Purpose**: Limit how many lines to READ from the input file (sequential)
- **When to use**: When you have a huge file and want to quickly read only the first N lines
- **Order**: Sequential (reads from the beginning)
- **Works with**: `--input_file` only

### `--num_samples`
- **Purpose**: Limit how many texts to USE for benchmarking (random sample)
- **When to use**: When you want a representative random sample from your data
- **Order**: Random (uses `--random_seed` for reproducibility)
- **Works with**: Both generated texts and `--input_file`

### Examples:

```bash
# Read first 1000 lines only (fast, but not random)
--input_file data.jsonl --max_samples 1000

# Read entire file, randomly sample 200 (slower to load, but random)
--input_file data.jsonl --num_samples 200

# Read first 1000 lines, then randomly sample 200 from those (best of both)
--input_file data.jsonl --max_samples 1000 --num_samples 200
```

## Command-Line Options

### Required (for API):
- `--model_name`: API model name
- `--endpoint`: API endpoint URL
- Environment variable: `RITS_API_KEY`

### Input Data:
- `--input_file`: Path to JSONL or JSONL.bz2 file
- `--field_path`: Dot-separated path to text field (e.g., `document.text`)
- `--max_samples`: Maximum samples to read from file (sequential, reads first N)
- `--num_samples`: Number of texts to use for benchmarking
  - With `--input_file`: Randomly samples N texts from loaded texts
  - Without `--input_file`: Generates N sample texts
  - Default: 100 for generated texts, all texts for input files
- `--random_seed`: Random seed for sampling (default: 42)

### Models:
- `--local_model_name`: HuggingFace model for GPU (default: `ibm-granite/granite-embedding-125m-english`)
- `--device`: Device for GPU (cuda/cpu, auto-detected if not specified)

### Benchmark Settings:
- `--batch_sizes`: Comma-separated batch sizes (default: `1,8,16,32`)
- `--skip_api`: Skip API benchmarking
- `--skip_gpu`: Skip GPU benchmarking

### Visualization:
- `--plot`: Generate visualization plots from results
- `--plot_dir`: Directory to save plots (default: `benchmark_plots`)
- `--plot_format`: Plot format - png, pdf, or svg (default: `png`)

## Field Path Examples

### Simple field:
```json
{"text": "Hello world"}
```
Use: `--field_path text`

### Nested field:
```json
{"document": {"text": "Nested text"}}
```
Use: `--field_path document.text`

### Deep nesting:
```json
{"metadata": {"content": {"body": {"text": "Deep nested"}}}}
```
Use: `--field_path metadata.content.body.text`

### Array with wildcard (all items):
```json
{"documents": [{"text": "Doc1"}, {"text": "Doc2"}, {"text": "Doc3"}]}
```
Use: `--field_path documents[*].text` or `--field_path documents[].text`

This will extract all three texts: "Doc1", "Doc2", "Doc3"

### Array with specific index:
```json
{"documents": [{"text": "Doc1"}, {"text": "Doc2"}]}
```
Use: `--field_path documents[0].text`

This will extract only "Doc1"

### Array implicit wildcard:
```json
{"documents": [{"text": "Doc1"}, {"text": "Doc2"}]}
```
Use: `--field_path documents.text`

When a field is an array, omitting the index applies to all items (implicit `[*]`)

### Nested arrays:
```json
{"results": [{"items": [{"text": "A1"}, {"text": "A2"}]}, {"items": [{"text": "B1"}]}]}
```
Use: `--field_path results[0].items[*].text`

This extracts texts from the first result's items: "A1", "A2"

### Array of simple values:
```json
{"tags": ["tag1", "tag2", "tag3"]}
```
Use: `--field_path tags[*]`

This extracts all tags as separate texts

### Auto-detection:
```json
{"text": "Will be auto-detected"}
{"content": "Will be auto-detected"}
{"question": "Will be auto-detected"}
```
Use: No `--field_path` needed

## Output

### Console Output

The script produces a formatted comparison table:

```
====================================================================================================
Batch Size: 16
----------------------------------------------------------------------------------------------------
Method               Total Time (s)     Throughput (q/s)     Avg Latency (ms)     Speedup
----------------------------------------------------------------------------------------------------
API                  12.345             8.10                 123.45               1.00x (baseline)
GPU                  3.456              28.94                34.56                3.57x
====================================================================================================
```

### Visualization Plots

When using `--plot`, the script generates 5 publication-ready plots:

#### 1. **Throughput Comparison** (`throughput_comparison.png`)
Bar chart comparing throughput (queries/sec) across batch sizes for each method.
- X-axis: Batch sizes
- Y-axis: Throughput (queries/sec)
- Colors: API (red), GPU (blue)

#### 2. **Latency Comparison** (`latency_comparison.png`)
Bar chart comparing average latency (ms) across batch sizes.
- X-axis: Batch sizes
- Y-axis: Average latency (milliseconds)
- Lower is better

#### 3. **Speedup Comparison** (`speedup_comparison.png`)
Bar chart showing GPU speedup factor over API.
- X-axis: Batch sizes
- Y-axis: Speedup factor (e.g., 3.5x means GPU is 3.5x faster)
- Baseline at 1.0x shown as red dashed line
- Values labeled on bars

#### 4. **Batch Size Scaling** (`batch_scaling.png`)
Two-panel line plot showing how performance scales with batch size.
- Left panel: Throughput scaling
- Right panel: Latency scaling
- Log scale on X-axis
- Shows scaling behavior clearly

#### 5. **Summary Dashboard** (`summary_dashboard.png`)
Comprehensive 4-panel dashboard with all key metrics:
- Top-left: Throughput comparison
- Top-right: Latency comparison
- Bottom-left: Throughput scaling
- Bottom-right: Speedup factors or summary statistics

**Features:**
- High resolution (300 DPI)
- Professional styling
- Clear labels and legends
- Consistent colors across plots
- Ready for papers and presentations

**Example usage:**
```bash
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --batch_sizes 1,8,16,32 \
    --plot \
    --plot_format pdf  # For LaTeX papers
```

## Testing

Run the test suite to verify JSONL reading functionality:

```bash
python test_jsonl_reading.py
```

## Examples in Project

```bash
# Benchmark with NQ dev-500 questions
python benchmark_embedding_timing.py \
    --input_file /proj/rh-inf-scaling/aashka/rag_rl/data/evals/nq-fixed/nq-dev-500-fixed.jsonl \
    --field_path question \
    --batch_sizes 1,8,16,32 \
    --max_samples 100

# Quick GPU-only test
python benchmark_embedding_timing.py \
    --skip_api \
    --num_samples 50 \
    --batch_sizes 8,16,32

# Extract all documents from array field
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --field_path documents[*].text \
    --batch_sizes 1,16,32

# Extract only first document from each line
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --field_path documents[0].text \
    --batch_sizes 1,16,32
```

## Files

- `benchmark_embedding_timing.py` - Main benchmark script
- `run_embedding_benchmark.sh` - Example runner with common configurations
- `test_jsonl_reading.py` - Test suite for JSONL reading functionality
- `BENCHMARK_README.md` - This file
