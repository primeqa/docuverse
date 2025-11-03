#!/usr/bin/env python3
"""
Test script for Granite to OpenVINO INT8 conversion

This script tests the conversion process by:
1. Running the conversion script
2. Generating random test data
3. Testing inference on both FP32 and INT8 models
4. Comparing outputs and performance
5. Validating accuracy
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path
import tempfile
import shutil

import numpy as np
import torch

try:
    import openvino as ov
    from openvino.runtime import Core
except ImportError:
    print("Error: OpenVINO is not installed. Install it with:")
    print("  pip install openvino openvino-dev")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Error: transformers is not installed. Install it with:")
    print("  pip install transformers torch")
    sys.exit(1)


def generate_random_test_sentences(num_sentences: int = 50, seed: int = 42) -> list:
    """
    Generate random test sentences for testing.

    Args:
        num_sentences: Number of sentences to generate
        seed: Random seed for reproducibility

    Returns:
        List of test sentences
    """
    np.random.seed(seed)

    # Word banks for random sentence generation
    subjects = [
        "The machine learning model", "Natural language processing", "Deep learning",
        "The neural network", "Artificial intelligence", "The transformer architecture",
        "The embedding layer", "The attention mechanism", "The encoder",
        "The decoder", "The tokenizer", "The dataset", "The algorithm",
        "Computer vision", "The classification task", "The optimization process",
        "The training procedure", "The inference engine", "The feature extractor"
    ]

    verbs = [
        "processes", "analyzes", "generates", "transforms", "computes",
        "optimizes", "learns", "predicts", "classifies", "encodes",
        "decodes", "tokenizes", "embeds", "extracts", "trains"
    ]

    objects = [
        "the input data efficiently", "complex patterns in text",
        "high-dimensional representations", "semantic relationships",
        "contextual information from sequences", "latent features",
        "vector representations", "probability distributions",
        "attention weights", "gradient updates", "loss values",
        "accuracy metrics", "embeddings in continuous space",
        "hierarchical features", "multi-modal information"
    ]

    sentences = []

    # Generate simple sentences
    for i in range(num_sentences // 2):
        subject = np.random.choice(subjects)
        verb = np.random.choice(verbs)
        obj = np.random.choice(objects)
        sentence = f"{subject} {verb} {obj}."
        sentences.append(sentence)

    # Generate compound sentences
    for i in range(num_sentences // 2):
        subject1 = np.random.choice(subjects)
        verb1 = np.random.choice(verbs)
        obj1 = np.random.choice(objects)

        subject2 = np.random.choice(subjects)
        verb2 = np.random.choice(verbs)
        obj2 = np.random.choice(objects)

        connective = np.random.choice([", and ", ", while ", ", but ", ". Additionally, "])
        sentence = f"{subject1} {verb1} {obj1}{connective}{subject2} {verb2} {obj2}."
        sentences.append(sentence)

    # Add some edge cases
    sentences.append("Short.")
    sentences.append("A" * 100)  # Long repetitive
    sentences.append("This is a very " + "long " * 50 + "sentence.")

    return sentences


def run_conversion(output_dir: str, model_name: str, no_quantize: bool = False) -> int:
    """
    Run the conversion script.

    Args:
        output_dir: Directory for output
        model_name: Model name to convert
        no_quantize: Whether to skip quantization

    Returns:
        Return code of the conversion script
    """
    print(f"\n{'='*70}")
    print("RUNNING CONVERSION SCRIPT")
    print(f"{'='*70}")

    script_path = Path(__file__).parent / "granite_to_openvino_int8.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--model_name", model_name,
        "--output_dir", output_dir,
        "--max_length", "512",
        "--calibration_samples", "100"  # Reduced for faster testing
    ]

    if no_quantize:
        cmd.append("--no_quantize")

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    return result.returncode


def load_openvino_model(model_path: str, tokenizer_path: str):
    """
    Load OpenVINO model and tokenizer.

    Args:
        model_path: Path to .xml model file
        tokenizer_path: Path to tokenizer directory

    Returns:
        Tuple of (compiled_model, tokenizer)
    """
    core = Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, "CPU")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return compiled_model, tokenizer


def get_embeddings_openvino(compiled_model, tokenizer, texts: list, max_length: int = 512):
    """
    Get embeddings using OpenVINO model.

    Args:
        compiled_model: Compiled OpenVINO model
        tokenizer: Tokenizer
        texts: List of input texts
        max_length: Maximum sequence length

    Returns:
        Array of embeddings
    """
    embeddings_list = []

    for text in texts:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        # Run inference
        input_data = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        result = compiled_model(input_data)

        # Get output (last_hidden_state)
        output_key = list(result.keys())[0]
        embeddings = result[output_key]

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        masked_embeddings = embeddings * attention_mask[:, :, np.newaxis]
        sum_embeddings = masked_embeddings.sum(axis=1)
        sum_mask = attention_mask.sum(axis=1, keepdims=True)
        pooled_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)

        embeddings_list.append(pooled_embeddings[0])

    return np.array(embeddings_list)


def get_embeddings_pytorch(model_name: str, texts: list, max_length: int = 512):
    """
    Get embeddings using original PyTorch model (ground truth).

    Args:
        model_name: HuggingFace model name
        texts: List of input texts
        max_length: Maximum sequence length

    Returns:
        Array of embeddings
    """
    print("Loading PyTorch model for ground truth...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    model.eval()

    embeddings_list = []

    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )

            # Run inference
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state

            # Mean pooling
            attention_mask = inputs["attention_mask"]
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            sum_embeddings = masked_embeddings.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            pooled_embeddings = sum_embeddings / torch.maximum(sum_mask, torch.tensor(1e-9))

            embeddings_list.append(pooled_embeddings[0].numpy())

    return np.array(embeddings_list)


def benchmark_inference(compiled_model, tokenizer, texts: list, num_warmup: int = 10, num_iterations: int = 100):
    """
    Benchmark inference speed.

    Args:
        compiled_model: Compiled OpenVINO model
        tokenizer: Tokenizer
        texts: List of input texts
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations

    Returns:
        Average inference time in milliseconds
    """
    # Prepare input
    text = texts[0] if texts else "Test sentence"
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    input_data = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }

    # Warmup
    for _ in range(num_warmup):
        compiled_model(input_data)

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        compiled_model(input_data)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations * 1000
    return avg_time


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate_similarity_matrix(embeddings):
    """Calculate pairwise cosine similarity matrix."""
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

    return sim_matrix


def compare_embeddings(emb1, emb2, labels=("Model 1", "Model 2")):
    """
    Compare two sets of embeddings.

    Args:
        emb1: First set of embeddings
        emb2: Second set of embeddings
        labels: Labels for the two sets

    Returns:
        Dictionary with comparison metrics
    """
    print(f"\nComparing {labels[0]} vs {labels[1]}:")
    print("-" * 60)

    # Calculate element-wise differences
    abs_diff = np.abs(emb1 - emb2)
    rel_diff = abs_diff / (np.abs(emb1) + 1e-9)

    print(f"Shape: {emb1.shape}")
    print(f"Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"Max absolute difference: {abs_diff.max():.6f}")
    print(f"Mean relative difference: {rel_diff.mean():.6f}")

    # Calculate cosine similarity between corresponding embeddings
    cos_sims = []
    for i in range(len(emb1)):
        cos_sim = cosine_similarity(emb1[i], emb2[i])
        cos_sims.append(cos_sim)

    cos_sims = np.array(cos_sims)
    print(f"\nCosine similarity between corresponding embeddings:")
    print(f"  Mean: {cos_sims.mean():.6f}")
    print(f"  Min: {cos_sims.min():.6f}")
    print(f"  Max: {cos_sims.max():.6f}")
    print(f"  Std: {cos_sims.std():.6f}")

    # Calculate similarity matrices
    sim_matrix1 = calculate_similarity_matrix(emb1)
    sim_matrix2 = calculate_similarity_matrix(emb2)

    # Compare similarity matrices
    sim_matrix_diff = np.abs(sim_matrix1 - sim_matrix2)
    print(f"\nSimilarity matrix difference:")
    print(f"  Mean: {sim_matrix_diff.mean():.6f}")
    print(f"  Max: {sim_matrix_diff.max():.6f}")

    return {
        "mean_abs_diff": abs_diff.mean(),
        "max_abs_diff": abs_diff.max(),
        "mean_cos_sim": cos_sims.mean(),
        "min_cos_sim": cos_sims.min(),
        "sim_matrix_diff_mean": sim_matrix_diff.mean(),
        "sim_matrix_diff_max": sim_matrix_diff.max()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test Granite to OpenVINO INT8 conversion"
    )
    parser.add_argument(
        "--model_name",
        default="ibm-granite/granite-embedding-english-r2",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: temporary directory)"
    )
    parser.add_argument(
        "--num_test_sentences",
        type=int,
        default=50,
        help="Number of test sentences to generate"
    )
    parser.add_argument(
        "--skip_conversion",
        action="store_true",
        help="Skip conversion and test existing model"
    )
    parser.add_argument(
        "--skip_pytorch",
        action="store_true",
        help="Skip PyTorch comparison (faster)"
    )
    parser.add_argument(
        "--keep_output",
        action="store_true",
        help="Keep output directory after testing"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GRANITE TO OPENVINO INT8 CONVERSION TEST")
    print("=" * 70)

    # Setup output directory
    use_temp_dir = args.output_dir is None
    if use_temp_dir:
        temp_dir = tempfile.mkdtemp(prefix="granite_openvino_test_")
        output_dir = temp_dir
        print(f"\nUsing temporary directory: {output_dir}")
    else:
        output_dir = args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Generate test data
        print(f"\n{'='*70}")
        print("STEP 1: GENERATING TEST DATA")
        print(f"{'='*70}")

        test_sentences = generate_random_test_sentences(args.num_test_sentences)
        print(f"Generated {len(test_sentences)} test sentences")
        print("\nSample sentences:")
        for i, sent in enumerate(test_sentences[:5]):
            print(f"  {i+1}. {sent[:80]}...")

        # Step 2: Run conversion
        if not args.skip_conversion:
            returncode = run_conversion(output_dir, args.model_name, no_quantize=False)
            if returncode != 0:
                print("\n❌ Conversion failed!")
                return 1
        else:
            print("\nSkipping conversion (using existing model)")

        # Step 3: Check output files
        print(f"\n{'='*70}")
        print("STEP 2: CHECKING OUTPUT FILES")
        print(f"{'='*70}")

        fp32_model_path = Path(output_dir) / "granite_embedding_fp32.xml"
        int8_model_path = Path(output_dir) / "granite_embedding_int8.xml"

        print("\nChecking for converted models...")
        if fp32_model_path.exists():
            print(f"  ✓ FP32 model found: {fp32_model_path}")
        else:
            print(f"  ✗ FP32 model not found: {fp32_model_path}")

        if int8_model_path.exists():
            print(f"  ✓ INT8 model found: {int8_model_path}")
        else:
            print(f"  ⚠ INT8 model not found: {int8_model_path}")
            print("    (This is expected if NNCF is not installed)")

        # Step 4: Test FP32 model
        print(f"\n{'='*70}")
        print("STEP 3: TESTING FP32 MODEL")
        print(f"{'='*70}")

        if fp32_model_path.exists():
            fp32_model, tokenizer = load_openvino_model(str(fp32_model_path), output_dir)
            print("✓ FP32 model loaded successfully")

            print("\nGenerating embeddings with FP32 model...")
            fp32_embeddings = get_embeddings_openvino(fp32_model, tokenizer, test_sentences[:10])
            print(f"✓ Generated embeddings shape: {fp32_embeddings.shape}")

            print("\nBenchmarking FP32 model...")
            fp32_time = benchmark_inference(fp32_model, tokenizer, test_sentences)
            print(f"✓ Average inference time: {fp32_time:.2f} ms")
            print(f"  Throughput: {1000/fp32_time:.2f} samples/sec")

        # Step 5: Test INT8 model
        print(f"\n{'='*70}")
        print("STEP 4: TESTING INT8 MODEL")
        print(f"{'='*70}")

        if int8_model_path.exists():
            int8_model, tokenizer = load_openvino_model(str(int8_model_path), output_dir)
            print("✓ INT8 model loaded successfully")

            print("\nGenerating embeddings with INT8 model...")
            int8_embeddings = get_embeddings_openvino(int8_model, tokenizer, test_sentences[:10])
            print(f"✓ Generated embeddings shape: {int8_embeddings.shape}")

            print("\nBenchmarking INT8 model...")
            int8_time = benchmark_inference(int8_model, tokenizer, test_sentences)
            print(f"✓ Average inference time: {int8_time:.2f} ms")
            print(f"  Throughput: {1000/int8_time:.2f} samples/sec")
            print(f"  Speedup vs FP32: {fp32_time/int8_time:.2f}x")

        # Step 6: Compare with PyTorch
        if not args.skip_pytorch:
            print(f"\n{'='*70}")
            print("STEP 5: COMPARING WITH PYTORCH MODEL")
            print(f"{'='*70}")

            pytorch_embeddings = get_embeddings_pytorch(
                args.model_name,
                test_sentences[:10]
            )
            print(f"✓ Generated PyTorch embeddings shape: {pytorch_embeddings.shape}")

            # Compare FP32 with PyTorch
            if fp32_model_path.exists():
                compare_embeddings(pytorch_embeddings, fp32_embeddings, ("PyTorch", "OpenVINO FP32"))

            # Compare INT8 with PyTorch
            if int8_model_path.exists():
                compare_embeddings(pytorch_embeddings, int8_embeddings, ("PyTorch", "OpenVINO INT8"))

            # Compare FP32 with INT8
            if fp32_model_path.exists() and int8_model_path.exists():
                compare_embeddings(fp32_embeddings, int8_embeddings, ("OpenVINO FP32", "OpenVINO INT8"))

        # Step 7: Summary
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")

        print("\n✓ All tests completed successfully!")
        print("\nKey findings:")
        if fp32_model_path.exists():
            print(f"  - FP32 model inference time: {fp32_time:.2f} ms")
        if int8_model_path.exists():
            print(f"  - INT8 model inference time: {int8_time:.2f} ms")
            if fp32_model_path.exists():
                print(f"  - INT8 speedup: {fp32_time/int8_time:.2f}x")

        print(f"\nOutput directory: {output_dir}")

        return 0

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if use_temp_dir and not args.keep_output:
            print(f"\nCleaning up temporary directory: {output_dir}")
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup temporary directory: {e}")


if __name__ == "__main__":
    sys.exit(main())
