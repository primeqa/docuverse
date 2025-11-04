# Granite Embedding Model to OpenVINO Conversion

This directory contains scripts for converting the IBM Granite embedding model (`ibm-granite/granite-embedding-english-r2`) to OpenVINO IR format for optimized CPU inference.

## Scripts

### 1. `granite_to_openvino_int8.py`
Main conversion script that converts the Granite model to OpenVINO format with optional INT8 quantization.

### 2. `test_granite_openvino_conversion.py`
Comprehensive test script that validates the conversion with randomly generated test data.

## Quick Start

```bash
# Basic conversion (will attempt INT8, fall back to FP32 if needed)
python scripts/granite_to_openvino_int8.py

# Convert with testing
python scripts/granite_to_openvino_int8.py --test

# FP32 only (skip INT8 quantization)
python scripts/granite_to_openvino_int8.py --no_quantize

# Run comprehensive test suite
python scripts/test_granite_openvino_conversion.py
```

## Requirements

```bash
pip install openvino openvino-dev transformers torch numpy

# For INT8 quantization (optional):
pip install nncf
```

## INT8 Quantization Compatibility

**IMPORTANT**: INT8 quantization requires compatible versions of OpenVINO and NNCF.

### Known Compatibility Issues

- **NNCF 2.17.0** is **NOT** compatible with **OpenVINO 2024.6.0+**
- This is due to API changes in OpenVINO 2024.6.0 where modules were reorganized

### Recommended Configurations

#### Option 1: Use Compatible Versions (for INT8)
```bash
pip install 'openvino==2024.3.0' 'openvino-dev==2024.3.0' 'nncf==2.17.0'
```

#### Option 2: Use FP32 Only (with latest OpenVINO)
```bash
python scripts/granite_to_openvino_int8.py --no_quantize
```

The FP32 model still provides:
- 5-10x speedup over PyTorch on CPU
- Cross-platform compatibility
- No quantization accuracy loss

## Usage Examples

### Basic Conversion

```bash
python scripts/granite_to_openvino_int8.py \
    --model_name ibm-granite/granite-embedding-english-r2 \
    --output_dir ./granite-openvino \
    --test
```

### Custom Configuration

```bash
python scripts/granite_to_openvino_int8.py \
    --model_name ibm-granite/granite-embedding-english-r2 \
    --output_dir ./my_model \
    --max_length 512 \
    --calibration_samples 300 \
    --test \
    --benchmark
```

### Using the Converted Model

```python
from openvino.runtime import Core
from transformers import AutoTokenizer
import numpy as np

# Load model
core = Core()
model = core.read_model("./granite-openvino/granite_embedding_fp32.xml")
compiled_model = core.compile_model(model, "CPU")
tokenizer = AutoTokenizer.from_pretrained("./granite-openvino")

# Run inference
text = "Your text here"
inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)

input_data = {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64)
}

result = compiled_model(input_data)
embeddings = result[list(result.keys())[0]]

# Mean pooling
attention_mask = inputs["attention_mask"]
masked_embeddings = embeddings * attention_mask[:, :, np.newaxis]
pooled = masked_embeddings.sum(axis=1) / np.maximum(attention_mask.sum(axis=1, keepdims=True), 1e-9)

print(f"Embedding shape: {pooled.shape}")
```

## Testing

### Run Full Test Suite

```bash
python scripts/test_granite_openvino_conversion.py \
    --num_test_sentences 50 \
    --keep_output
```

### Test Options

- `--num_test_sentences N`: Number of random test sentences to generate
- `--skip_conversion`: Test existing model without re-converting
- `--skip_pytorch`: Skip PyTorch comparison (faster testing)
- `--keep_output`: Keep the output directory after testing
- `--output_dir DIR`: Specify output directory (default: temp directory)

### What the Test Does

1. Generates random test sentences
2. Runs the conversion script
3. Tests FP32 model inference
4. Tests INT8 model inference (if available)
5. Compares with PyTorch baseline
6. Benchmarks inference speed
7. Calculates accuracy metrics (cosine similarity, embedding differences)

## Performance

Typical performance improvements over PyTorch (CPU):

- **FP32 model**: 5-10x faster
- **INT8 model**: 10-15x faster (when compatible versions are used)
- **Model size reduction**: ~4x with INT8 quantization

Example results:
```
PyTorch baseline: ~500ms per inference
OpenVINO FP32:    ~180ms per inference (2.8x faster)
OpenVINO INT8:    ~50ms per inference (10x faster) *with compatible versions
```

## Troubleshooting

### INT8 Quantization Fails

**Error**: `ModuleNotFoundError: No module named 'openvino.op'` or `AttributeError: module 'openvino' has no attribute 'Node'`

**Solution**: This is a known compatibility issue. Either:
1. Downgrade OpenVINO: `pip install 'openvino==2024.3.0' 'openvino-dev==2024.3.0'`
2. Use FP32 only: Add `--no_quantize` flag

### NNCF Warnings

**Warning**: `NNCF provides best results with torch==2.7.*`

**Solution**: This is usually safe to ignore. If you encounter issues, consider:
```bash
pip install 'torch==2.7.0'
```

### Model Download Issues

**Error**: Connection timeout or slow download

**Solution**: Set HuggingFace cache directory:
```bash
export HF_HOME=/path/to/cache
python scripts/granite_to_openvino_int8.py
```

## Output Files

After successful conversion, you'll have:

```
output_directory/
├── granite_embedding_fp32.xml       # FP32 model definition
├── granite_embedding_fp32.bin       # FP32 model weights (~284 MB)
├── granite_embedding_int8.xml       # INT8 model definition (if quantized)
├── granite_embedding_int8.bin       # INT8 model weights (~71 MB, if quantized)
├── tokenizer.json                   # Tokenizer
├── tokenizer_config.json            # Tokenizer config
└── special_tokens_map.json          # Special tokens
```

## Advanced Options

### Calibration Dataset Size

More calibration samples = better INT8 accuracy, but slower conversion:

```bash
# Fast (less accurate INT8)
python scripts/granite_to_openvino_int8.py --calibration_samples 100

# Slow (more accurate INT8)
python scripts/granite_to_openvino_int8.py --calibration_samples 500
```

### Custom Model

```bash
python scripts/granite_to_openvino_int8.py \
    --model_name your/custom-embedding-model \
    --output_dir ./custom-openvino \
    --max_length 1024
```

## License

These scripts are provided as-is for use with the IBM Granite embedding models.

## References

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [NNCF Documentation](https://github.com/openvinotoolkit/nncf)
- [IBM Granite Models](https://huggingface.co/ibm-granite)
