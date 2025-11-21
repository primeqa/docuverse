# Random Sampling Guide

The benchmark script now supports random sampling from loaded texts, allowing you to get representative samples from large datasets.

## Quick Start

```bash
# Load entire file and randomly sample 200 texts
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --field_path text \
    --num_samples 200 \
    --batch_sizes 1,16,32
```

## Key Options

### `--num_samples`
Number of texts to use for benchmarking.
- **With `--input_file`**: Randomly samples N texts from loaded data
- **Without `--input_file`**: Generates N sample texts
- **Default**: 100 for generated texts, all texts for input files

### `--random_seed`
Seed for random number generator (default: 42)
- Ensures reproducible sampling
- Change to get different random samples

## Common Use Cases

### 1. Representative Sample from Large Dataset

**Scenario**: You have 10,000 questions but want to benchmark with 500 random samples.

```bash
python benchmark_embedding_timing.py \
    --input_file nq-train.jsonl \
    --field_path question \
    --num_samples 500 \
    --random_seed 42 \
    --batch_sizes 1,16,32
```

**What happens**:
1. Loads all 10,000 questions
2. Randomly samples 500 using seed 42
3. Runs benchmark on those 500

### 2. Multiple Random Samples (Different Seeds)

**Scenario**: Run multiple benchmarks with different random samples.

```bash
# First run
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --num_samples 100 \
    --random_seed 42

# Second run (different sample)
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --num_samples 100 \
    --random_seed 123

# Third run (different sample)
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --num_samples 100 \
    --random_seed 456
```

### 3. Efficient Sampling from Huge Files

**Scenario**: File has 1M texts, you want 200 random samples, but don't want to load entire file.

```bash
# Read first 10K lines, then randomly sample 200 from those
python benchmark_embedding_timing.py \
    --input_file huge_file.jsonl \
    --field_path text \
    --max_samples 10000 \
    --num_samples 200 \
    --batch_sizes 1,16,32
```

**What happens**:
1. Reads first 10,000 lines from file (fast)
2. Randomly samples 200 from those 10,000
3. Runs benchmark on those 200

**Trade-off**: Faster loading, but sample is only from first 10K lines, not entire dataset.

### 4. Reproducible Benchmarks

**Scenario**: You want to ensure same results across runs.

```bash
# Run 1 - today
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --num_samples 300 \
    --random_seed 42

# Run 2 - next week (will use same texts)
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --num_samples 300 \
    --random_seed 42  # Same seed = same sample
```

## Understanding the Process

### Without Random Sampling
```
File (1000 texts) → Load all → Benchmark all 1000
```

### With Random Sampling
```
File (1000 texts) → Load all → Random sample (200) → Benchmark 200
```

### With Both max_samples and num_samples
```
File (1M texts) → Load first 10K → Random sample (200) → Benchmark 200
                   (sequential)      (random)
```

## max_samples vs num_samples

| Feature | `--max_samples` | `--num_samples` |
|---------|-----------------|-----------------|
| **Purpose** | Limit lines read from file | Limit texts used for benchmark |
| **Order** | Sequential (first N) | Random (sampled) |
| **Speed** | Fast (stops reading early) | Slower (reads more, then samples) |
| **Representativeness** | Not random (biased to start) | Random (representative) |
| **Works with** | `--input_file` only | Both file and generated |

## Best Practices

### ✅ DO:
- Use `--num_samples` for representative random samples
- Use consistent `--random_seed` for reproducible results
- Combine `--max_samples` + `--num_samples` for huge files
- Document which seed you used in your results

### ❌ DON'T:
- Forget to set `--random_seed` if reproducibility matters
- Use `--max_samples` alone if you need random samples (it's sequential)
- Use extremely large `--num_samples` on huge files without `--max_samples` (slow to load)

## Examples with Arrays

Random sampling works with array extraction too!

```bash
# Extract all documents from arrays, then randomly sample
python benchmark_embedding_timing.py \
    --input_file rag_data.jsonl \
    --field_path "retrieved_docs[*].text" \
    --num_samples 500 \
    --batch_sizes 1,16,32
```

If file has 100 questions with 5 docs each (500 total texts), this will randomly sample 500 texts from those.

## Testing

Run the test suite to verify sampling works:

```bash
python test_random_sampling.py
```

This tests:
- ✓ Random sampling functionality
- ✓ Seed reproducibility
- ✓ Different seeds produce different samples
- ✓ Integration with benchmark script

## Output Example

```
================================================================================
EMBEDDING TIMING BENCHMARK
================================================================================
Batch sizes: [1, 16, 32]
================================================================================

Loading texts from file: data.jsonl
  Using field path: text
✓ Loaded 1000 texts from file
  Preview: This is an example text...

Randomly sampling 200 texts from 1000 available texts
  Random seed: 42
✓ Sampled 200 texts

Final dataset size: 200 texts

✓ Initialized API embedder: ibm/slate-125m-english-rtrvr-v2
✓ Initialized GPU embedder: ibm-granite/granite-embedding-125m-english on cuda
...
```

## See Also

- `BENCHMARK_README.md` - Full documentation
- `test_random_sampling.py` - Test suite
- `run_embedding_benchmark.sh` - Example scripts
