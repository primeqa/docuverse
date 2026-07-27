#!/usr/bin/env python3
"""
Quantize IBM Granite Embedding Model to INT8 on GPU

This script quantizes the ibm-granite/granite-embedding-english-r2 model
to 8-bit integers using GPU acceleration and runs comprehensive tests.

Usage:
    # With generated test samples:
    python quantize_granite_gpu_int8.py --num_test_samples 100

    # With JSONL file (auto-detect text field):
    python quantize_granite_gpu_int8.py --input_file data.jsonl

    # With JSONL file and custom field path:
    python quantize_granite_gpu_int8.py --input_file data.jsonl --field_path document.text

    # With array indexing:
    python quantize_granite_gpu_int8.py --input_file data.jsonl --field_path documents[0].text

    # With array wildcard (all items):
    python quantize_granite_gpu_int8.py --input_file data.jsonl --field_path documents[*].text

    # With compressed file:
    python quantize_granite_gpu_int8.py --input_file data.jsonl.bz2 --max_samples 500
"""

import argparse
import time
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from tqdm import tqdm

# Import the JSONL utilities
from docuverse.utils.jsonl_utils import read_jsonl_file

# Check if bitsandbytes is available
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("⚠️  WARNING: bitsandbytes not available. Install with: pip install bitsandbytes")


def check_gpu_availability():
    """Check if GPU is available and display info."""
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA is not available. GPU quantization requires CUDA.")
        print("   Please ensure you have:")
        print("   1. A CUDA-compatible GPU")
        print("   2. CUDA toolkit installed")
        print("   3. PyTorch with CUDA support")
        return False

    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return True


def generate_test_sentences(num_sentences: int = 100) -> List[str]:
    """Generate diverse test sentences for validation."""
    sentences = [
        # Technical / AI content
        "Machine learning models can process natural language with high accuracy.",
        "Deep neural networks learn hierarchical representations from data.",
        "Transformer architectures revolutionized natural language processing tasks.",
        "Embedding models convert text into dense vector representations.",
        "Quantization reduces model size while maintaining acceptable performance.",

        # General knowledge
        "The Earth orbits around the Sun once every 365 days.",
        "Water freezes at zero degrees Celsius under standard atmospheric pressure.",
        "The human brain contains approximately 86 billion neurons.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "DNA carries the genetic instructions for all living organisms.",

        # Short sentences
        "Hello world.",
        "Good morning.",
        "Thank you.",
        "How are you?",
        "See you later.",

        # Longer sentences
        "In a groundbreaking study published today, researchers have discovered a novel approach to improving efficiency.",
        "The conference brought together experts from various fields to discuss interdisciplinary approaches.",
        "Climate change represents one of the most significant challenges facing humanity today.",

        # Questions
        "What is the capital of France?",
        "How does machine learning work?",
        "Why is the sky blue?",
        "When was the internet invented?",
        "Where can I find more information?",

        # Domain-specific
        "The convolutional neural network achieved 95% accuracy on the test set.",
        "Attention mechanisms allow models to focus on relevant input features.",
        "Transfer learning enables models to leverage pre-trained knowledge.",
        "Batch normalization stabilizes training of deep neural networks.",
        "Gradient descent optimizes model parameters iteratively.",

        # Similar semantic pairs
        "Dogs are loyal pets.",
        "Canines make faithful companions.",
        "The weather is beautiful today.",
        "It's a gorgeous day outside.",
        "Programming requires logical thinking.",
        "Coding demands analytical reasoning.",
    ]

    # Extend if needed
    while len(sentences) < num_sentences:
        sentences.append(f"Test sentence number {len(sentences) + 1} for embedding evaluation.")

    return sentences[:num_sentences]


def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to get sentence embeddings."""
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_texts(model, tokenizer, texts: List[str], device: str, batch_size: int = 32,
                 show_progress: bool = True, debug: bool = False) -> np.ndarray:
    """Encode texts to embeddings using the model."""
    model.eval()
    all_embeddings = []

    num_texts = len(texts)
    if debug:
        print(f"[DEBUG] encode_texts: num_texts={num_texts}, batch_size={batch_size}, device={device}")

    iterator = list(range(0, num_texts, batch_size))
    if debug:
        print(f"[DEBUG] iterator: {iterator}")

    if show_progress:
        iterator = tqdm(iterator, desc="Encoding", unit="batch")

    with torch.no_grad():
        for batch_idx, i in enumerate(iterator):
            batch_texts = texts[i:i + batch_size]
            if debug:
                print(f"[DEBUG] Batch {batch_idx}: processing {len(batch_texts)} texts (indices {i} to {i+len(batch_texts)-1})")

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            # Move to device - handle both string device and torch.device
            try:
                if isinstance(device, str):
                    device_obj = torch.device(device)
                else:
                    device_obj = device

                # Check if model is on the device, if not move it
                model_device = next(model.parameters()).device
                if device_obj != model_device:
                    device_obj = model_device

                encoded = {k: v.to(device_obj) for k, v in encoded.items()}
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Device error: {e}, falling back to CPU")
                # Fallback: just move to CPU
                encoded = {k: v.cpu() for k, v in encoded.items()}

            # Get embeddings
            outputs = model(**encoded)
            embeddings = mean_pooling(outputs, encoded['attention_mask'])

            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            if debug:
                print(f"[DEBUG] Batch {batch_idx} embeddings shape: {embeddings.shape}")

            all_embeddings.append(embeddings.cpu().numpy())

    final_embeddings = np.vstack(all_embeddings)
    if debug:
        print(f"[DEBUG] Final embeddings shape: {final_embeddings.shape}")

    return final_embeddings


def quantize_model_int8(model: nn.Module, device: str) -> nn.Module:
    """
    Quantize model to INT8 using PyTorch's native quantization.

    This uses dynamic quantization which is suitable for transformers
    and works well on GPU.
    """
    print("\nQuantizing model to INT8...")
    print("  Using PyTorch dynamic quantization (GPU-compatible)")

    # Move model to CPU for quantization
    model_cpu = model.cpu()

    # Apply dynamic quantization to linear layers
    # This quantizes weights to INT8 and activations dynamically during inference
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},  # Quantize all Linear layers
        dtype=torch.qint8    # Use 8-bit integer quantization
    )

    print("  ✓ Model quantized successfully")

    # Note: PyTorch's quantized models run on CPU
    # For GPU inference with quantization, we'll need to use a different approach
    # or run comparison on CPU

    return quantized_model


def quantize_weights_gpu(model: nn.Module) -> nn.Module:
    """
    Quantize model weights to INT8 range (GPU-compatible).
    Creates a deep copy and quantizes it to avoid modifying the original.
    """
    print("\nApplying weight quantization (GPU-compatible)...")

    # Import here to avoid circular dependency
    import copy

    # Create a deep copy to avoid modifying the original model
    quantized_model = copy.deepcopy(model)

    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            # Get the weight
            weight = module.weight.data

            # Compute scale and zero point
            weight_min = weight.min()
            weight_max = weight.max()
            scale = (weight_max - weight_min) / 255.0
            zero_point = -weight_min / scale

            # Quantize to INT8 range and dequantize (simulated quantization)
            weight_q = torch.clamp(torch.round(weight / scale + zero_point), 0, 255)
            weight_dq = (weight_q - zero_point) * scale

            # Update the weight
            module.weight.data = weight_dq

    print("  ✓ Weights quantized to INT8 range")
    return quantized_model


def calculate_model_size(model: nn.Module) -> Tuple[float, float]:
    """Calculate model size in MB."""
    # Calculate parameter size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    # Calculate buffer size
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = (param_size + buffer_size) / 1024 / 1024
    param_size_mb = param_size / 1024 / 1024

    return total_size, param_size_mb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between two sets of vectors."""
    if len(a.shape) == 1:
        a = a.reshape(1, -1)
    if len(b.shape) == 1:
        b = b.reshape(1, -1)

    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)

    # Compute similarity
    return np.sum(a_norm * b_norm, axis=1)


def compare_embeddings(orig_embeddings: np.ndarray, quant_embeddings: np.ndarray,
                       test_texts: List[str]) -> dict:
    """Compare embeddings from original and quantized models."""
    print(f"\n{'='*70}")
    print("EMBEDDING COMPARISON ANALYSIS")
    print(f"{'='*70}\n")

    # Calculate cosine similarities
    print("Computing cosine similarities...")
    start_time = time.time()
    similarities = cosine_similarity(orig_embeddings, quant_embeddings)
    comparison_time = time.time() - start_time

    # Statistics
    stats = {
        'mean': float(similarities.mean()),
        'median': float(np.median(similarities)),
        'min': float(similarities.min()),
        'max': float(similarities.max()),
        'std': float(similarities.std()),
        'comparison_time': comparison_time,
    }

    print(f"Cosine Similarity Statistics:")
    print(f"  Mean:   {stats['mean']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  Min:    {stats['min']:.6f}")
    print(f"  Max:    {stats['max']:.6f}")
    print(f"  Std:    {stats['std']:.6f}")
    print(f"  Comparison time: {comparison_time:.3f}s")
    print()

    # Distribution
    print("Distribution:")
    bins = [0.90, 0.95, 0.97, 0.99, 0.995, 1.0]
    for i in range(len(bins)-1):
        count = np.sum((similarities >= bins[i]) & (similarities < bins[i+1]))
        pct = 100 * count / len(similarities)
        print(f"  [{bins[i]:.3f} - {bins[i+1]:.3f}): {count:3d} samples ({pct:5.1f}%)")

    # Show worst matches
    print(f"\n{'='*70}")
    print("SAMPLES WITH LOWEST SIMILARITY")
    print(f"{'='*70}")
    worst_indices = np.argsort(similarities)[:5]
    for idx in worst_indices:
        print(f"\nSimilarity: {similarities[idx]:.6f}")
        text = test_texts[idx]
        print(f"Text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        print(f"Original norm: {np.linalg.norm(orig_embeddings[idx]):.4f}")
        print(f"Quantized norm: {np.linalg.norm(quant_embeddings[idx]):.4f}")

    return stats


def benchmark_inference_speed(model, tokenizer, texts: List[str], device: str,
                               batch_size: int = 32, num_runs: int = 3) -> dict:
    """Benchmark inference speed."""
    print(f"\nBenchmarking inference speed ({num_runs} runs)...")

    times = []
    for run in range(num_runs):
        start_time = time.time()
        _ = encode_texts(model, tokenizer, texts, device, batch_size, show_progress=False)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s ({len(texts)/elapsed:.1f} samples/sec)")

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    throughput = len(texts) / avg_time

    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'throughput': throughput,
        'times': times
    }


def test_single_batch_size(model, quantized_model, tokenizer, test_texts,
                           device, quant_device, batch_size, benchmark_runs,
                           show_details=True) -> dict:
    """
    Test a single batch size and return results.

    Returns dict with:
        - orig_time, quant_time, orig_throughput, quant_throughput
        - orig_std, quant_std
        - similarity_mean, similarity_min, similarity_std
    """
    if show_details:
        print(f"\n{'='*70}")
        print(f"TESTING BATCH SIZE: {batch_size}")
        print(f"{'='*70}")

    # Encode with original model
    if show_details:
        print(f"\nEncoding with original model (batch_size={batch_size})...")
    start_time = time.time()
    orig_embeddings = encode_texts(model, tokenizer, test_texts, device, batch_size, show_progress=show_details)
    orig_encode_time = time.time() - start_time

    # Benchmark original model
    if show_details:
        print(f"Benchmarking original model...")
    orig_benchmark = benchmark_inference_speed(
        model, tokenizer, test_texts, device, batch_size, benchmark_runs
    )

    # Encode with quantized model
    if show_details:
        print(f"\nEncoding with quantized model (batch_size={batch_size})...")
    start_time = time.time()
    quant_embeddings = encode_texts(quantized_model, tokenizer, test_texts, quant_device, batch_size, show_progress=show_details)
    quant_encode_time = time.time() - start_time

    # Benchmark quantized model
    if show_details:
        print(f"Benchmarking quantized model...")
    quant_benchmark = benchmark_inference_speed(
        quantized_model, tokenizer, test_texts, quant_device, batch_size, benchmark_runs
    )

    # Compare embeddings
    if show_details:
        print(f"\nComputing similarities...")
    comp_start = time.time()
    similarities = cosine_similarity(orig_embeddings, quant_embeddings)
    comparison_time = time.time() - comp_start

    return {
        'batch_size': batch_size,
        'orig_encode_time': orig_encode_time,
        'quant_encode_time': quant_encode_time,
        'orig_encode_throughput': len(test_texts) / orig_encode_time,
        'quant_encode_throughput': len(test_texts) / quant_encode_time,
        'orig_throughput': len(test_texts) / orig_encode_time,  # Alias for table
        'quant_throughput': len(test_texts) / quant_encode_time,  # Alias for table
        'orig_benchmark_time': orig_benchmark['avg_time'],
        'quant_benchmark_time': quant_benchmark['avg_time'],
        'orig_benchmark_std': orig_benchmark['std_time'],
        'quant_benchmark_std': quant_benchmark['std_time'],
        'orig_benchmark_throughput': orig_benchmark['throughput'],
        'quant_benchmark_throughput': quant_benchmark['throughput'],
        'similarity_mean': float(similarities.mean()),
        'similarity_min': float(similarities.min()),
        'similarity_std': float(similarities.std()),
        'encode_speedup': orig_encode_time / quant_encode_time,
        'benchmark_speedup': quant_benchmark['throughput'] / orig_benchmark['throughput'],
        'comparison_time': comparison_time,
    }


def print_results_table(results: List[dict], num_samples: int):
    """Print results as a formatted table."""
    print(f"\n{'='*120}")
    print("BATCH SIZE COMPARISON TABLE")
    print(f"{'='*120}")
    print(f"Number of samples: {num_samples}")
    print()

    # Header
    print(f"{'Batch':<8} │ {'Original':<40} │ {'Quantized':<40} │ {'Quality':<20}")
    print(f"{'Size':<8} │ {'Time (s)':<12} {'Throughput':<14} {'Std':<12} │ {'Time (s)':<12} {'Throughput':<14} {'Std':<12} │ {'Similarity':<20}")
    print("─" * 120)

    # Rows - Single Pass
    print("Single Pass Encoding:")
    for r in results:
        print(f"{r['batch_size']:<8} │ "
              f"{r['orig_encode_time']:>10.3f}  "
              f"{r['orig_throughput']:>12.1f}  "
              f"{'N/A':<12} │ "
              f"{r['quant_encode_time']:>10.3f}  "
              f"{r['quant_throughput']:>12.1f}  "
              f"{'N/A':<12} │ "
              f"{r['similarity_mean']:>18.6f}")

    print()
    print("Benchmark (Multi-run Average):")
    for r in results:
        print(f"{r['batch_size']:<8} │ "
              f"{r['orig_benchmark_time']:>10.3f}  "
              f"{r['orig_benchmark_throughput']:>12.1f}  "
              f"±{r['orig_benchmark_std']:>10.3f} │ "
              f"{r['quant_benchmark_time']:>10.3f}  "
              f"{r['quant_benchmark_throughput']:>12.1f}  "
              f"±{r['quant_benchmark_std']:>10.3f} │ "
              f"{r['similarity_mean']:>18.6f}")

    print("─" * 120)

    # Speedup summary
    print("\nSpeedup Summary:")
    print(f"{'Batch':<8} │ {'Encoding Speedup':<20} │ {'Benchmark Speedup':<20} │ {'Similarity Min':<20}")
    print("─" * 72)
    for r in results:
        print(f"{r['batch_size']:<8} │ {r['encode_speedup']:>18.2f}x │ "
              f"{r['benchmark_speedup']:>18.2f}x │ {r['similarity_min']:>18.6f}")

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize IBM Granite embedding model to INT8 on GPU"
    )
    parser.add_argument(
        "--model_name",
        default="ibm-granite/granite-embedding-english-r2",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        default="./granite-embedding-int8-gpu",
        help="Output directory for the quantized model"
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=None,
        help="Number of test samples for validation (default: 100 for generated, all for input file)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to JSONL or JSONL.bz2 file containing texts to test"
    )
    parser.add_argument(
        "--field_path",
        type=str,
        default=None,
        help="Dot-separated path to text field with array support. "
             "Examples: 'document.text', 'documents[*].text', 'documents[0].text'. "
             "If not specified, will try common fields: 'text', 'content', 'question'"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to read from input file (default: read all)"
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="32",
        help="Batch size(s) for encoding. Can be a single value (e.g., '32') or "
             "comma-separated list (e.g., '1,8,16,32') to test multiple batch sizes"
    )
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=3,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the quantized model to disk"
    )
    parser.add_argument(
        "--use_true_quantization",
        action="store_true",
        help="Use PyTorch's true INT8 quantization (CPU only, smaller size)"
    )
    parser.add_argument(
        "--use_bitsandbytes",
        action="store_true",
        help="Use bitsandbytes for INT8 quantization (GPU, HuggingFace integration)"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 (half precision) for faster inference with memory savings"
    )
    parser.add_argument(
        "--use_fp8",
        action="store_true",
        help="Use FP8 quantization (requires torch 2.1+ and compatible GPU like RTX 4090)"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation to use (default: eager). Options: eager, sdpa, flash_attention_2"
    )

    args = parser.parse_args()

    # Validate bitsandbytes availability
    if args.use_bitsandbytes and not BITSANDBYTES_AVAILABLE:
        print("❌ Error: bitsandbytes is not available but --use_bitsandbytes was specified")
        print("   Install with: pip install bitsandbytes")
        return 1

    # Validate only one quantization method is selected
    quant_methods = sum([
        args.use_true_quantization,
        args.use_bitsandbytes,
        args.use_fp16,
        args.use_fp8
    ])
    if quant_methods > 1:
        print("❌ Error: Only one quantization method can be used at a time")
        print("   Choose one of: --use_true_quantization, --use_bitsandbytes, --use_fp16, --use_fp8")
        return 1

    # Parse batch sizes
    try:
        batch_sizes = [int(x.strip()) for x in args.batch_size.split(",")]
    except ValueError:
        print(f"Error: Invalid batch size format: {args.batch_size}")
        print("Use a single integer or comma-separated integers (e.g., '1,8,16,32')")
        return 1

    print("=" * 70)
    print("GRANITE EMBEDDING MODEL INT8 QUANTIZATION (GPU)")
    print("=" * 70)
    if len(batch_sizes) > 1:
        print(f"Testing with batch sizes: {batch_sizes}")
    else:
        print(f"Batch size: {batch_sizes[0]}")
    print()

    # Check GPU availability
    if not check_gpu_availability():
        print("\nFalling back to CPU...")
        device = "cpu"
    else:
        device = "cuda"

    print(f"\nUsing device: {device}")
    print()

    try:
        # Load model and tokenizer
        print("Step 1: Loading original model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # Determine dtype for original model
        if args.use_fp16 or args.use_fp8:
            # For FP16/FP8, we don't need FP32 original (too slow to compare)
            # Load original in FP16 for fair comparison
            orig_dtype = torch.float16
            print("  Loading model in FP16 for comparison...")
        else:
            orig_dtype = torch.float32

        # Load original model first (for comparison)
        model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=orig_dtype,
            attn_implementation=args.attn_implementation
        )
        # Move to device
        model = model.to(device)

        orig_size, orig_param_size = calculate_model_size(model)
        print(f"  ✓ Model loaded: {args.model_name}")
        print(f"  ✓ Attention implementation: {args.attn_implementation}")
        print(f"  ✓ Data type: {orig_dtype}")
        print(f"  ✓ Model size: {orig_size:.2f} MB ({orig_param_size:.2f} MB parameters)")
        print(f"  ✓ Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

        model.eval()

        # Load or generate test data
        print(f"\nStep 2: Loading test data...")
        if args.input_file:
            print(f"  Loading from file: {args.input_file}")
            if args.field_path:
                print(f"  Using field path: {args.field_path}")
            if args.max_samples:
                print(f"  Max samples: {args.max_samples}")

            try:
                test_texts = read_jsonl_file(
                    args.input_file,
                    field_path=args.field_path,
                    max_samples=args.max_samples,
                    verbose=True
                )
                print(f"  ✓ Loaded {len(test_texts)} texts from file")

                if not test_texts:
                    print("  ✗ No texts loaded from file!")
                    return 1

                # Show preview
                preview = test_texts[0][:100] + "..." if len(test_texts[0]) > 100 else test_texts[0]
                print(f"  Preview: \"{preview}\"")

                # If num_test_samples specified, limit the texts
                if args.num_test_samples and len(test_texts) > args.num_test_samples:
                    print(f"  Limiting to {args.num_test_samples} samples")
                    test_texts = test_texts[:args.num_test_samples]

            except FileNotFoundError:
                print(f"  ✗ Error: File not found: {args.input_file}")
                return 1
            except Exception as e:
                print(f"  ✗ Error loading file: {e}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            # Generate test samples
            num_to_generate = args.num_test_samples if args.num_test_samples else 100
            print(f"  Generating {num_to_generate} test samples...")
            test_texts = generate_test_sentences(num_to_generate)
            print(f"  ✓ Generated {len(test_texts)} test samples")

        # Quantize model once (before testing batch sizes)
        if args.use_fp16:
            print(f"\nStep 3: Using FP16 (half precision)...")
        elif args.use_fp8:
            print(f"\nStep 3: Using FP8 quantization...")
        else:
            print(f"\nStep 3: Quantizing model to INT8...")
        quant_start_time = time.time()

        if args.use_fp16:
            # FP16: Model is already loaded in FP16, just use it directly
            print("  Model already in FP16 format")
            quantized_model = model  # Same model, already FP16
            quant_device = device
            quantization_time = 0.0  # No conversion needed
        elif args.use_fp8:
            # FP8: Use torch's built-in FP8 support (requires torch 2.1+)
            print("  Converting model to FP8...")
            try:
                # Check if FP8 is available
                if not hasattr(torch, 'float8_e4m3fn'):
                    print("  ⚠️  Warning: FP8 not available in this PyTorch version")
                    print("  Falling back to FP16...")
                    quantized_model = model
                    quant_device = device
                    quantization_time = 0.0
                else:
                    # For FP8, we'll use torch.compile with float8 dtype
                    # This requires torch >= 2.1 and compatible hardware
                    import copy
                    quantized_model = copy.deepcopy(model)
                    # Convert to FP8 - note: this is a simplified approach
                    # Full FP8 support requires more complex quantization schemes
                    for name, param in quantized_model.named_parameters():
                        if param.dtype == torch.float16:
                            # Cast to float8_e4m3fn (E4M3 format)
                            param.data = param.data.to(torch.float8_e4m3fn).to(torch.float16)
                    quant_device = device
                    quantization_time = time.time() - quant_start_time
                    print(f"  ✓ Model converted to FP8")
            except Exception as e:
                print(f"  ⚠️  Warning: FP8 conversion failed: {e}")
                print("  Falling back to FP16...")
                quantized_model = model
                quant_device = device
                quantization_time = 0.0
        elif args.use_bitsandbytes:
            # Load quantized model with bitsandbytes
            print("  Loading model with bitsandbytes INT8 quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            quantized_model = AutoModel.from_pretrained(
                args.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation=args.attn_implementation
            )
            quantized_model.eval()
            quant_device = device
            quantization_time = time.time() - quant_start_time
            print(f"  ✓ Model loaded and quantized with bitsandbytes")
        elif args.use_true_quantization:
            # Use PyTorch's true INT8 quantization (CPU only, actual size reduction)
            print("  Using TRUE INT8 quantization (CPU, reduced size)")
            quantized_model = quantize_model_int8(model, 'cpu')
            quant_device = 'cpu'
            quantization_time = time.time() - quant_start_time
        elif device == "cuda":
            # Use GPU-compatible quantization (creates a copy)
            quantized_model = quantize_weights_gpu(model)
            quant_device = device
            quantization_time = time.time() - quant_start_time
        else:
            # Use PyTorch's built-in quantization for CPU
            quantized_model = quantize_model_int8(model, device)
            quant_device = device
            quantization_time = time.time() - quant_start_time

        quant_size, quant_param_size = calculate_model_size(quantized_model)
        print(f"  ✓ Quantized model size: {quant_size:.2f} MB ({quant_param_size:.2f} MB parameters)")
        print(f"  ✓ Size reduction: {(1 - quant_size/orig_size)*100:.1f}%")
        print(f"  ✓ Quantization time: {quantization_time:.3f}s")

        # Test each batch size
        all_results = []
        for i, batch_size in enumerate(batch_sizes):
            show_details = (i == 0) or (len(batch_sizes) == 1)  # Show details for first batch size only
            print(f"\n{'='*70}")
            print(f"TESTING BATCH SIZE: {batch_size}")
            print(f"{'='*70}")

            result = test_single_batch_size(
                model=model,
                quantized_model=quantized_model,
                tokenizer=tokenizer,
                test_texts=test_texts,
                device=device,
                quant_device=quant_device,
                batch_size=batch_size,
                benchmark_runs=args.benchmark_runs,
                show_details=show_details
            )
            result['quantization_time'] = quantization_time  # Add to first result
            all_results.append(result)

        # Print results table if multiple batch sizes
        if len(batch_sizes) > 1:
            print(f"\n{'='*70}")
            print("RESULTS COMPARISON ACROSS BATCH SIZES")
            print(f"{'='*70}")
            print_results_table(all_results, len(test_texts))

        # Final summary for first/only batch size
        first_result = all_results[0]
        print(f"\n{'='*70}")
        if len(batch_sizes) > 1:
            print(f"DETAILED SUMMARY (Batch Size: {first_result['batch_size']})")
        else:
            print("QUANTIZATION SUMMARY")
        print(f"{'='*70}")

        print(f"\n📊 MODEL SIZES:")
        print(f"  Original:  {orig_size:.2f} MB")
        print(f"  Quantized: {quant_size:.2f} MB")
        print(f"  Reduction: {(1 - quant_size/orig_size)*100:.1f}%")

        print(f"\n⏱️  TIMING (Single Pass):")
        print(f"  Original encoding:   {first_result['orig_encode_time']:.3f}s ({first_result['orig_encode_throughput']:.1f} samples/sec)")
        print(f"  Quantized encoding:  {first_result['quant_encode_time']:.3f}s ({first_result['quant_encode_throughput']:.1f} samples/sec)")
        print(f"  Encoding speedup:    {first_result['encode_speedup']:.2f}x")
        print(f"  Quantization time:   {quantization_time:.3f}s")

        print(f"\n⚡ BENCHMARK ({args.benchmark_runs} runs average):")
        print(f"  Original throughput:  {first_result['orig_benchmark_throughput']:.1f} samples/sec")
        print(f"  Quantized throughput: {first_result['quant_benchmark_throughput']:.1f} samples/sec")
        print(f"  Benchmark speedup:    {first_result['benchmark_speedup']:.2f}x")
        print(f"  Original avg time:    {first_result['orig_benchmark_time']:.3f}s ± {first_result['orig_benchmark_std']:.3f}s")
        print(f"  Quantized avg time:   {first_result['quant_benchmark_time']:.3f}s ± {first_result['quant_benchmark_std']:.3f}s")

        print(f"\n✨ EMBEDDING QUALITY:")
        print(f"  Mean cosine similarity: {first_result['similarity_mean']:.6f}")
        print(f"  Min cosine similarity:  {first_result['similarity_min']:.6f}")
        print(f"  Std deviation:          {first_result['similarity_std']:.6f}")

        print(f"\n🔧 ADDITIONAL METRICS:")
        total_time = first_result['orig_encode_time'] + quantization_time + first_result['quant_encode_time']
        print(f"  Total processing time:  {total_time:.3f}s")
        print(f"  Comparison time:        {first_result['comparison_time']:.3f}s")
        print(f"  Samples processed:      {len(test_texts)}")
        print(f"  Batch size(s):          {', '.join(map(str, batch_sizes))}")
        print(f"  Device:                 {device}")

        # Save model if requested
        if args.save_model:
            print(f"\nStep 9: Saving quantized model...")
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if args.use_true_quantization or (device == "cpu"):
                # Save true quantized model
                torch.save(quantized_model.state_dict(), output_path / "quantized_model.pt")
                # Also save the quantized model in JIT format for better portability
                try:
                    quantized_model_jit = torch.jit.script(quantized_model)
                    torch.jit.save(quantized_model_jit, output_path / "quantized_model_jit.pt")
                    print(f"  ✓ JIT model saved")
                except Exception as e:
                    print(f"  ⚠ Could not save JIT model: {e}")
            else:
                # Save the wrapped model
                torch.save(quantized_model.state_dict(), output_path / "quantized_model.pt")

            # Save config
            model.config.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"  ✓ Model saved to: {output_path}")

        print(f"\n{'='*70}")
        print("✓ QUANTIZATION COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
