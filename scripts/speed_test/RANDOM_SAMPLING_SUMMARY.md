# Random Sampling Feature Summary

## What Changed

The benchmark script now supports **random sampling** from loaded texts, allowing you to benchmark with representative subsets of large datasets.

## New Parameters

### `--num_samples`
- **Old behavior**: Only controlled generated text count
- **New behavior**:
  - Without `--input_file`: Generates N sample texts (as before)
  - With `--input_file`: Randomly samples N texts from loaded data (NEW!)
  - Default: 100 for generated, all texts for input files

### `--random_seed`
- **Purpose**: Seed for random number generator
- **Default**: 42
- **Use**: Ensures reproducible sampling across runs

## Quick Examples

### Basic Random Sampling
```bash
# Load 1000 texts, randomly sample 200
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --field_path text \
    --num_samples 200
```

### Reproducible Sampling
```bash
# Same seed = same sample every time
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --num_samples 200 \
    --random_seed 42
```

### Efficient Large File Sampling
```bash
# Read first 10K lines, then randomly sample 500
python benchmark_embedding_timing.py \
    --input_file huge.jsonl \
    --max_samples 10000 \
    --num_samples 500
```

## Key Differences: max_samples vs num_samples

| Feature | `--max_samples` | `--num_samples` |
|---------|-----------------|-----------------|
| **What it does** | Limits lines READ from file | Limits texts USED for benchmark |
| **Order** | Sequential (first N) | Random (sampled) |
| **Representativeness** | Biased to start of file | Random sample |
| **Speed** | Fast (stops reading) | Can be slow (may read entire file) |
| **Best for** | Quick tests on huge files | Representative sampling |

## Workflow

### Without Random Sampling (Old)
```
File (1000 texts) → Load all 1000 → Benchmark all 1000
```

### With Random Sampling (New)
```
File (1000 texts) → Load all 1000 → Sample 200 → Benchmark 200
                                      (random)
```

### Combined Approach (Optimal for Large Files)
```
File (1M texts) → Load first 10K → Sample 500 → Benchmark 500
                  (sequential)     (random)
```

## Benefits

✅ **Representative samples**: Not just first N texts
✅ **Reproducible**: Same seed = same sample
✅ **Flexible**: Works with all input methods (files, generated, arrays)
✅ **Efficient**: Benchmark on subset, save time
✅ **Scientific**: Compare different models on same random sample

## Testing

All functionality verified:

```bash
# Run comprehensive tests
python test_random_sampling.py

# Interactive demo
python demo_random_sampling.py
```

Test results:
- ✅ Random sampling works correctly
- ✅ Seed provides reproducibility
- ✅ Different seeds → different samples
- ✅ Samples are subset of original
- ✅ Integration with benchmark script works
- ✅ Edge cases handled (num_samples > available)

## Files Modified/Created

### Modified
- `benchmark_embedding_timing.py` - Added random sampling logic
- `BENCHMARK_README.md` - Updated documentation

### Created
- `RANDOM_SAMPLING_GUIDE.md` - Complete usage guide
- `RANDOM_SAMPLING_SUMMARY.md` - This file
- `test_random_sampling.py` - Comprehensive test suite
- `demo_random_sampling.py` - Interactive demonstration

### Updated
- `run_embedding_benchmark.sh` - Added random sampling examples

## Use Cases

### 1. Large Dataset Benchmarking
You have 10K texts, want to test with 500 random samples:
```bash
--input_file large.jsonl --num_samples 500 --random_seed 42
```

### 2. Reproducible Experiments
Ensure same texts across multiple runs:
```bash
--random_seed 42  # Always use same seed
```

### 3. Quick Representative Test
Test on 100 random texts instead of all 5000:
```bash
--input_file data.jsonl --num_samples 100
```

### 4. Multiple Random Runs
Compare performance across different samples:
```bash
--random_seed 42   # Run 1
--random_seed 123  # Run 2 (different sample)
--random_seed 456  # Run 3 (different sample)
```

## Documentation

For detailed information, see:
- `RANDOM_SAMPLING_GUIDE.md` - Complete guide with examples
- `BENCHMARK_README.md` - Full benchmark documentation
- `test_random_sampling.py` - Test suite
- `demo_random_sampling.py` - Interactive demo

## Example Output

```
Loading texts from file: data.jsonl
  Using field path: text
✓ Loaded 1000 texts from file
  Preview: This is an example...

Randomly sampling 200 texts from 1000 available texts
  Random seed: 42
✓ Sampled 200 texts

Final dataset size: 200 texts
```
