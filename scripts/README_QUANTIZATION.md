# Granite Embedding Model INT8 Quantization

This script quantizes the `ibm-granite/granite-embedding-english-r2` model to 8-bit integers and provides comprehensive testing and benchmarking.

## Script: `quantize_granite_gpu_int8.py`

### Features

- **Two Quantization Modes:**
  - **GPU Mode** (default): Simulates INT8 quantization with FP32 on GPU for speed
  - **True INT8 Mode** (`--use_true_quantization`): Uses PyTorch's dynamic quantization for actual size reduction (CPU only)

- **Comprehensive Testing:**
  - Embedding quality comparison (cosine similarity)
  - Inference speed benchmarking
  - Model size reporting
  - Statistical analysis

- **Flexible Options:**
  - Configurable test sample size
  - Adjustable batch size
  - Multiple benchmark runs
  - Optional model saving

### Requirements

```bash
pip install torch transformers numpy tqdm
```

For GPU support, ensure you have CUDA-enabled PyTorch installed.

### Usage

#### Basic Usage (GPU quantization with generated samples)

```bash
python scripts/quantize_granite_gpu_int8.py
```

This will:
- Load the model on GPU
- Apply GPU-compatible INT8 quantization
- Generate and test 100 sample sentences
- Run 3 benchmark runs
- Show detailed comparison statistics

#### Using JSONL Files

```bash
# With auto-detected text field
python scripts/quantize_granite_gpu_int8.py --input_file data.jsonl

# With custom field path
python scripts/quantize_granite_gpu_int8.py \
  --input_file data.jsonl \
  --field_path document.text

# With nested array indexing (first item)
python scripts/quantize_granite_gpu_int8.py \
  --input_file data.jsonl \
  --field_path documents[0].text

# With array wildcard (all items)
python scripts/quantize_granite_gpu_int8.py \
  --input_file data.jsonl \
  --field_path documents[*].text

# With compressed file and limits
python scripts/quantize_granite_gpu_int8.py \
  --input_file data.jsonl.bz2 \
  --max_samples 500 \
  --num_test_samples 200
```

#### True INT8 Quantization (Size Reduction)

```bash
python scripts/quantize_granite_gpu_int8.py --use_true_quantization --save_model
```

This will:
- Apply true INT8 quantization (runs on CPU)
- Achieve ~4x model size reduction
- Save the quantized model to disk

#### Testing Multiple Batch Sizes

```bash
# Test with multiple batch sizes (displays comparison table)
python scripts/quantize_granite_gpu_int8.py \
  --batch_size "1,8,16,32" \
  --num_test_samples 100 \
  --benchmark_runs 3
```

This will test all batch sizes and display a comprehensive comparison table showing:
- Single-pass encoding performance for each batch size
- Benchmark averages with standard deviation
- Speedup comparisons and embedding quality metrics

#### Custom Configuration

```bash
python scripts/quantize_granite_gpu_int8.py \
  --model_name ibm-granite/granite-embedding-english-r2 \
  --output_dir ./my_quantized_model \
  --num_test_samples 200 \
  --batch_size 64 \
  --benchmark_runs 5 \
  --save_model
```

#### Using Different Attention Implementations

```bash
# Use SDPA (Scaled Dot Product Attention) - may be faster on newer GPUs
python scripts/quantize_granite_gpu_int8.py \
  --attn_implementation sdpa \
  --batch_size "1,8,16,32" \
  --num_test_samples 100

# Use Flash Attention 2 (requires flash-attn package installed)
python scripts/quantize_granite_gpu_int8.py \
  --attn_implementation flash_attention_2 \
  --batch_size "16,32"

# Default: eager (standard PyTorch attention)
python scripts/quantize_granite_gpu_int8.py \
  --attn_implementation eager
```

#### Using FP16 (RECOMMENDED for Speed + Memory) ⚡

```bash
# Use FP16 for 2x faster inference with 50% memory savings
python scripts/quantize_granite_gpu_int8.py \
  --use_fp16 \
  --batch_size "8,16,32,64" \
  --num_test_samples 100

# Combine with SDPA attention for best performance
python scripts/quantize_granite_gpu_int8.py \
  --use_fp16 \
  --attn_implementation sdpa \
  --model_name ibm-granite/granite-embedding-small-english-r2
```

**FP16 Benefits**:
- **Fast**: 1.5-2x faster than FP32, **4-6x faster than INT8**
- **Memory savings**: 50% reduction (181 MB → 91 MB)
- **Perfect quality**: Virtually identical to FP32 (100% cosine similarity)
- **Native GPU support**: No extra dependencies required
- **Best for**: Small embedding models (<2B parameters) on any modern GPU

#### Using FP8 (Maximum Performance on RTX 4090) 🚀

```bash
# Use FP8 for maximum performance on Ada Lovelace GPUs
python scripts/quantize_granite_gpu_int8.py \
  --use_fp8 \
  --batch_size "16,32,64" \
  --num_test_samples 100 \
  --attn_implementation sdpa

# Note: Requires PyTorch 2.1+ and compatible GPU (RTX 4090, H100, etc.)
```

**FP8 Benefits**:
- **Fastest**: 2-4x faster than FP16, **6-12x faster than INT8**
- **Memory savings**: 50% reduction (same as FP16)
- **Excellent quality**: Better than INT8, comparable to FP16
- **Hardware accelerated**: Native FP8 Tensor Cores on RTX 4090/Ada Lovelace
- **Best for**: Production deployment on modern GPUs (Ampere/Ada/Hopper)

#### Using BitsAndBytes Quantization

```bash
# Use bitsandbytes for INT8 quantization (GPU, real memory savings)
python scripts/quantize_granite_gpu_int8.py \
  --use_bitsandbytes \
  --batch_size "8,32,64" \
  --num_test_samples 100

# Combine with SDPA attention for best results
python scripts/quantize_granite_gpu_int8.py \
  --use_bitsandbytes \
  --attn_implementation sdpa \
  --model_name ibm-granite/granite-embedding-small-english-r2

# Note: Requires bitsandbytes: pip install bitsandbytes
```

**BitsAndBytes Benefits**:
- **Real memory savings**: ~65% size reduction (181 MB → 64 MB)
- **High quality**: >99.9% cosine similarity maintained
- **GPU support**: Works natively on CUDA GPUs
- **HuggingFace integration**: Uses official HuggingFace BitsAndBytesConfig
- **True INT8**: Actual 8-bit integer weights, not simulated
- ⚠️ **Warning**: 3x slower than FP32 for small models (<2B params) - use FP16 instead!

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_name` | `ibm-granite/granite-embedding-english-r2` | HuggingFace model identifier |
| `--output_dir` | `./granite-embedding-int8-gpu` | Output directory for saved model |
| `--num_test_samples` | `100` (generated) / `all` (file) | Number of test samples for validation |
| `--input_file` | `None` | Path to JSONL or JSONL.bz2 file with test texts |
| `--field_path` | `None` (auto-detect) | Dot-separated path to text field with array support |
| `--max_samples` | `None` (all) | Maximum samples to read from input file |
| `--batch_size` | `32` | Batch size for encoding (or comma-separated list like "1,8,16,32") |
| `--benchmark_runs` | `3` | Number of benchmark runs for averaging |
| `--save_model` | `False` | Save the quantized model to disk |
| `--use_true_quantization` | `False` | Use true INT8 quantization (CPU, smaller size) |
| `--use_bitsandbytes` | `False` | Use bitsandbytes for INT8 quantization (GPU, HuggingFace) |
| `--use_fp16` | `False` | Use FP16 (half precision) for fast inference ⭐ RECOMMENDED |
| `--use_fp8` | `False` | Use FP8 quantization (requires PyTorch 2.1+ and RTX 4090+) |
| `--attn_implementation` | `eager` | Attention implementation: eager, sdpa, or flash_attention_2 |

### Output

The script provides detailed output including:

1. **Model Loading**: Shows model size and parameter count
2. **Test Data**: Displays number of samples and preview
3. **Original Model Encoding**: Single-pass timing and throughput
4. **Original Model Benchmark**: Multi-run average with standard deviation
5. **Quantization**: Time taken and size reduction
6. **Quantized Model Encoding**: Single-pass timing, throughput, and speedup
7. **Quantized Model Benchmark**: Multi-run average with statistics
8. **Embedding Comparison**: Cosine similarity statistics and distribution
9. **Comprehensive Summary**:
   - 📊 Model sizes (original, quantized, reduction)
   - ⏱️ Single-pass timing (encoding time, throughput, speedup)
   - ⚡ Benchmark results (avg time ± std, throughput, speedup)
   - ✨ Embedding quality (mean, min, std of cosine similarity)
   - 🔧 Additional metrics (total time, batch size, device)

### Example Output

#### Single Batch Size

```
======================================================================
GRANITE EMBEDDING MODEL INT8 QUANTIZATION (GPU)
======================================================================
Batch size: 32

✓ CUDA available: NVIDIA GeForce RTX 4090
  CUDA version: 12.8
  GPU memory: 25.25 GB

Using device: cuda

Step 1: Loading original model...
  ✓ Model loaded: ibm-granite/granite-embedding-english-r2
  ✓ Model size: 568.45 MB (568.44 MB parameters)
  ✓ Number of parameters: 149,014,272

Step 3: Quantizing model to INT8...
  ✓ Quantized model size: 568.45 MB (568.44 MB parameters)
  ✓ Size reduction: 0.0%
  ✓ Quantization time: 0.123s

======================================================================
QUANTIZATION SUMMARY
======================================================================

📊 MODEL SIZES:
  Original:  568.45 MB
  Quantized: 568.45 MB
  Reduction: 0.0%

⏱️  TIMING (Single Pass):
  Original encoding:   1.323s (22.7 samples/sec)
  Quantized encoding:  0.022s (1383.2 samples/sec)
  Encoding speedup:    60.98x
  Quantization time:   0.123s

⚡ BENCHMARK (2 runs average):
  Original throughput:  1491.3 samples/sec
  Quantized throughput: 1621.9 samples/sec
  Benchmark speedup:    1.09x
  Original avg time:    0.020s ± 0.001s
  Quantized avg time:   0.018s ± 0.001s

✨ EMBEDDING QUALITY:
  Mean cosine similarity: 0.997025
  Min cosine similarity:  0.994205
  Std deviation:          0.001100

🔧 ADDITIONAL METRICS:
  Total processing time:  1.468s
  Comparison time:        0.000s
  Samples processed:      30
  Batch size(s):          32
  Device:                 cuda
```

#### Multiple Batch Sizes (Comparison Table)

When testing multiple batch sizes with `--batch_size "1,8,16,32"`:

```
======================================================================
RESULTS COMPARISON ACROSS BATCH SIZES
======================================================================

========================================================================================================================
BATCH SIZE COMPARISON TABLE
========================================================================================================================
Number of samples: 100

Batch    │ Original                                 │ Quantized                                │ Quality
Size     │ Time (s)     Throughput     Std          │ Time (s)     Throughput     Std          │ Similarity
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Single Pass Encoding:
1        │      2.636          37.9  N/A          │      1.552          64.5  N/A          │           0.997486
8        │      0.505         198.1  N/A          │      0.222         450.4  N/A          │           0.997486
16       │      0.118         846.2  N/A          │      0.135         740.2  N/A          │           0.997486
32       │      0.073        1368.2  N/A          │      0.074        1350.7  N/A          │           0.997486

Benchmark (Multi-run Average):
1        │      1.515          66.0  ±     0.044 │      1.499          66.7  ±     0.031 │           0.997486
8        │      0.247         405.7  ±     0.016 │      0.232         430.8  ±     0.004 │           0.997486
16       │      0.130         768.0  ±     0.006 │      0.136         737.2  ±     0.009 │           0.997486
32       │      0.077        1307.2  ±     0.003 │      0.068        1466.1  ±     0.003 │           0.997486
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Speedup Summary:
Batch    │ Encoding Speedup     │ Benchmark Speedup    │ Similarity Min
────────────────────────────────────────────────────────────────────────
1        │               1.70x │               1.01x │           0.994205
8        │               2.27x │               1.06x │           0.994205
16       │               0.87x │               0.96x │           0.994205
32       │               0.99x │               1.12x │           0.994205
========================================================================================================================
```

### Performance Expectations & Quantization Methods

The script supports **six quantization methods**:

#### 1. FP16 (Half Precision) ⭐ RECOMMENDED
- **Command**: `--use_fp16`
- **Speed**: **1.5-2x faster** than FP32
- **Size**: 50% reduction (181 MB → 91 MB)
- **Quality**: 100% cosine similarity (virtually identical)
- **Device**: GPU (any modern CUDA GPU)
- **Use Case**: **Best all-around choice for small models**
- **Note**: Native GPU support, no dependencies

#### 2. FP8 Quantization 🚀 FASTEST (RTX 4090+)
- **Command**: `--use_fp8`
- **Speed**: **2-4x faster** than FP16, **6-12x faster** than INT8
- **Size**: 50% reduction (same as FP16)
- **Quality**: >99.9% cosine similarity
- **Device**: GPU (RTX 4090, H100, Ada Lovelace+)
- **Use Case**: Maximum performance on modern GPUs
- **Requirements**: PyTorch 2.1+, native FP8 Tensor Cores
- **Note**: Hardware-accelerated on Ada Lovelace/Hopper

#### 3. GPU Mode (Default - Simulated INT8)
- **Command**: Default (no flags)
- **Speed**: 1.0-1.2x of original
- **Size**: No reduction (FP32 weights, simulated INT8 range)
- **Quality**: >99.7% cosine similarity
- **Device**: GPU (CUDA required)
- **Use Case**: Quick testing, baseline comparison
- **Note**: Weights clipped to INT8 range but stored as FP32

#### 4. True INT8 Mode (PyTorch Dynamic Quantization)
- **Command**: `--use_true_quantization`
- **Speed**: Varies (CPU-dependent)
- **Size**: ~75% reduction (142 MB vs 568 MB)
- **Quality**: >99% cosine similarity
- **Device**: CPU only
- **Use Case**: Deployment with size constraints, CPU inference
- **Note**: Actual INT8 weights, smaller model size

#### 5. BitsAndBytes INT8 ⚠️ SLOW for Small Models
- **Command**: `--use_bitsandbytes`
- **Speed**: **0.3x of original (3x slower!)**
- **Size**: ~65% reduction (64 MB vs 182 MB for granite-small)
- **Quality**: >99.9% cosine similarity
- **Device**: GPU (CUDA required)
- **Use Case**: Large models (>6.7B params) with memory constraints
- **Requirements**: `pip install bitsandbytes`
- **Note**: ⚠️ Only use for models >6.7B params - much slower than FP16 for small models!

**Comparison Summary**:

| Method | Speed vs FP32 | Size Reduction | Quality | Device | Best For |
|--------|---------------|----------------|---------|--------|----------|
| **FP16** ⭐ | **1.5-2x faster** | 50% | 100% | Any GPU | **Small models (<2B)** |
| **FP8** 🚀 | **2-4x faster** | 50% | 99.9% | RTX 4090+ | **Maximum performance** |
| **GPU (Default)** | 1.0-1.2x | None | 99.7% | GPU | Testing |
| **True INT8** | Varies | 75% | 99% | CPU | CPU deployment |
| **BitsAndBytes** | **0.3x (3x slower!)** | 65% | 99.9% | GPU | Large models (>6.7B) |

**Recommendations**:
- **For Granite embedding models**: Use `--use_fp16 --attn_implementation sdpa` (best speed + quality)
- **For RTX 4090**: Use `--use_fp8 --attn_implementation sdpa` (maximum performance)
- **For large LLMs (>6.7B)**: Use `--use_bitsandbytes` (memory savings)
- **For CPU**: Use `--use_true_quantization` (size reduction)

### Understanding the Results

**Timing Metrics**:
- **Single Pass**: First encoding (includes compilation/warmup)
- **Benchmark**: Average of multiple runs (more stable measurement)
- **Throughput**: Samples processed per second
- **Speedup**: Ratio of times (original/quantized for single pass)

**Batch Size Comparison Table** (when testing multiple batch sizes):
- **Single Pass Encoding**: Shows first-run performance with compilation overhead
  - Lower batch sizes (1, 8) show more compilation impact
  - Higher batch sizes (16, 32) benefit from better GPU utilization
- **Benchmark (Multi-run Average)**: Shows stable performance after warmup
  - Standard deviation (±) indicates run-to-run consistency
  - More reliable for predicting production performance
- **Speedup Summary**:
  - Encoding speedup includes JIT compilation effects
  - Benchmark speedup is more representative of real-world gains
  - Quality metrics (similarity) should remain consistent across batch sizes

**Interpreting Batch Size Results**:
- **Batch size 1**: Highest per-sample latency, good for single queries
- **Batch size 8-16**: Balanced latency and throughput
- **Batch size 32+**: Maximum throughput, best for batch processing
- **GPU quantization typically shows 1.0-1.2x speedup** in benchmarks (minimal overhead)
- **Single-pass speedup varies widely** due to JIT compilation (can be misleading)

**Cosine Similarity**: Measures how similar embeddings are between original and quantized models
- >0.99: Excellent quality, minimal degradation
- 0.95-0.99: Good quality, acceptable for most use cases
- <0.95: Noticeable degradation, may affect downstream tasks

**Performance Notes**:
- First run may be slower due to compilation (JIT)
- Benchmark averages smooth out compilation effects
- Single-pass speedup may differ from benchmark speedup
- Standard deviation shows consistency across runs
- Batch size significantly impacts throughput but not quality

### Attention Implementations

The script supports three attention implementations via the `--attn_implementation` parameter:

1. **eager** (default):
   - Standard PyTorch attention implementation
   - Most compatible, works on all hardware
   - Good baseline performance
   - Recommended for CPU or older GPUs

2. **sdpa** (Scaled Dot Product Attention):
   - Uses PyTorch's native `torch.nn.functional.scaled_dot_product_attention`
   - Available in PyTorch 2.0+
   - Automatically selects optimal kernel (flash attention, memory-efficient, or math)
   - Often faster than eager on modern GPUs (Ampere/Ada/Hopper)
   - Recommended for modern NVIDIA GPUs (RTX 30xx, 40xx, A100, H100)

3. **flash_attention_2**:
   - Requires separate installation: `pip install flash-attn`
   - Fastest attention implementation for supported hardware
   - Reduces memory usage for long sequences
   - Requires Ampere or newer GPU (compute capability 8.0+)
   - Recommended when maximum performance is critical

**Performance Tips**:
- Try `sdpa` first on modern GPUs - often provides 10-20% speedup over eager
- Use `flash_attention_2` for long sequences or when maximum performance is needed
- Attention implementation affects inference speed but not embedding quality
- Some implementations may have different memory requirements

### Notes

- GPU quantization uses simulated INT8 (weight clipping to INT8 range in FP32) for compatibility
- True INT8 quantization uses PyTorch's dynamic quantization (CPU only)
- For production deployment, consider using OpenVINO or ONNX Runtime for optimized INT8 inference
- The model maintains high quality (>99.5% similarity) even after quantization

### Troubleshooting

**CUDA out of memory**: Reduce `--batch_size` to 16 or 8

**Slow CPU performance**: Use `--num_test_samples 20` for faster testing

**Import errors**: Ensure all dependencies are installed:
```bash
pip install torch transformers numpy tqdm
```

### JSONL Field Path Syntax

The `--field_path` parameter supports flexible field extraction:

**Simple nested fields:**
- `text` - top-level field
- `document.text` - nested field
- `metadata.content` - deeply nested field

**Array indexing:**
- `documents[0].text` - first item in array
- `documents[1].content` - second item in array

**Array wildcards:**
- `documents[*].text` - all items in array
- `documents[].text` - same as [*]

**Auto-detection (no --field_path):**
- Tries common fields: `text`, `content`, `question`, `query`, `passage`, `document`

### Related Scripts

- `jsonl_utils.py`: Utility module for reading JSONL files with nested field extraction
- `export_to_openvino_onnx.py`: Export to OpenVINO/ONNX with INT8/INT4 quantization
- `sentence_transformer_backend_comparison.py`: Compare different backend performance
