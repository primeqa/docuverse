#!/usr/bin/env python3
"""
Convert IBM Granite Embedding Model to OpenVINO INT8/INT4 Format

This script converts the ibm-granite/granite-embedding-english-r2 model
to OpenVINO IR format with INT8 or INT4 quantization for optimized inference on Intel hardware.

The script uses SentenceTransformers library to load and handle the embedding model,
which provides a high-level interface with built-in pooling and normalization.

Quantization Options:
- INT8: Full model quantization with calibration for optimal accuracy/speed balance
- INT4: Weight-only quantization for maximum compression (4x smaller than FP32)
- FP32: No quantization (baseline)

COMPATIBILITY NOTE:
INT8 quantization requires compatible versions of OpenVINO and NNCF:
- NNCF 2.17.0 is NOT compatible with OpenVINO 2024.6.0+
- Recommended: OpenVINO 2024.1.0 with NNCF 2.11.0
- The script will automatically fall back to FP32 if quantization fails
- FP32 models still provide significant speedup over PyTorch on CPU

To fix INT8 quantization issues:
1. Use compatible versions: pip install 'openvino==2024.1.0' 'openvino-dev==2024.1.0' 'nncf==2.11.0'
2. Or use --no_quantize flag to only generate FP32 model

Requirements:
- sentence-transformers
- openvino
- openvino-dev
- nncf (optional, for INT8/INT4 quantization)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import shutil

import torch
import numpy as np
from transformers import AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers is not installed. Install it with:")
    print("  pip install sentence-transformers")
    sys.exit(1)

try:
    import openvino as ov
    from openvino.tools import mo
    from openvino.runtime import Core, serialize

    # Apply compatibility monkey patch BEFORE importing NNCF
    # This fixes compatibility issues between NNCF 2.17.0 and OpenVINO 2024.6.0
    try:
        from openvino.runtime import Node
        if not hasattr(ov, 'Node'):
            ov.Node = Node
    except Exception:
        pass

except ImportError:
    print("Error: OpenVINO is not installed. Install it with:")
    print("  pip install openvino openvino-dev")
    sys.exit(1)

try:
    import nncf
except ImportError:
    print("Warning: NNCF not installed. INT8 quantization will not be available.")
    print("Install it with: pip install nncf")
    nncf = None


def convert_to_openvino(
    model_name: str = "ibm-granite/granite-embedding-english-r2",
    output_dir: str = "./granite-embedding-openvino-int8",
    max_length: int = 512,
    quantize_int8: bool = False,
    quantize_int4: bool = False,
    calibration_samples: int = 300
) -> str:
    """
    Convert the Granite embedding model to OpenVINO IR format with optional quantization.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save the OpenVINO model
        max_length: Maximum sequence length for the model
        quantize_int8: Whether to apply INT8 quantization (full model)
        quantize_int4: Whether to apply INT4 quantization (weights only)
        calibration_samples: Number of samples for calibration dataset (INT8 only)

    Returns:
        Path to the converted model
    """
    print(f"Converting {model_name} to OpenVINO IR format...")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model using SentenceTransformers
    print("\n1. Loading model with SentenceTransformers...")

    # Try newer 'dtype' parameter first, fall back to 'torch_dtype' for older versions
    try:
        st_model = SentenceTransformer(
            model_name,
            device='cpu',
            model_kwargs={
                'dtype': torch.float32,
                'attn_implementation': 'eager'  # Use eager attention for ONNX/OpenVINO compatibility
            }
        )
    except TypeError:
        # Older versions of sentence-transformers use 'torch_dtype' instead of 'dtype'
        st_model = SentenceTransformer(
            model_name,
            device='cpu',
            model_kwargs={
                'torch_dtype': torch.float32,
                'attn_implementation': 'eager'
            }
        )

    # Set model to evaluation mode
    st_model.eval()

    # Get the underlying transformer model and tokenizer
    # SentenceTransformer stores these as _modules
    tokenizer = st_model.tokenizer

    # Save the entire SentenceTransformer model (includes tokenizer)
    st_model.save_pretrained(output_dir)
    print(f"✓ SentenceTransformer model and tokenizer saved to {output_dir}")

    # Create example input for conversion
    print("\n2. Creating example input for model conversion...")
    example_text = "This is a sample text for embedding generation."
    example_inputs = tokenizer(
        example_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )

    # Convert to OpenVINO format
    print("\n3. Converting to OpenVINO IR format...")
    try:
        # Prepare inputs for conversion
        input_dict = {
            "input_ids": example_inputs["input_ids"],
            "attention_mask": example_inputs["attention_mask"]
        }

        # Extract the underlying transformer model from SentenceTransformer
        # For most models, the transformer is the first module
        if hasattr(st_model, '_modules') and '0' in st_model._modules:
            transformer_module = st_model._modules['0']
            if hasattr(transformer_module, 'auto_model'):
                base_model = transformer_module.auto_model
            else:
                base_model = st_model
        else:
            base_model = st_model

        # Convert using OpenVINO's convert_model
        ov_model = ov.convert_model(
            base_model,
            example_input=input_dict,
            input=[
                ("input_ids", [1, max_length], np.int64),
                ("attention_mask", [1, max_length], np.int64)
            ]
        )

        # Save FP32 model
        fp32_model_path = output_path / "granite_embedding_fp32.xml"
        ov.save_model(ov_model, str(fp32_model_path))
        print(f"✓ FP32 model saved to {fp32_model_path}")

    except Exception as e:
        print(f"Error during OpenVINO conversion: {e}")
        print("\nTrying alternative conversion method via ONNX...")

        # Alternative: Export to ONNX first, then convert to OpenVINO
        onnx_path = output_path / "temp_model.onnx"

        with torch.no_grad():
            torch.onnx.export(
                base_model,
                (example_inputs["input_ids"], example_inputs["attention_mask"]),
                str(onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=14,
                do_constant_folding=True
            )

        print(f"✓ ONNX model created at {onnx_path}")

        # Convert ONNX to OpenVINO
        ov_model = ov.convert_model(str(onnx_path))
        fp32_model_path = output_path / "granite_embedding_fp32.xml"
        ov.save_model(ov_model, str(fp32_model_path))
        print(f"✓ OpenVINO FP32 model saved to {fp32_model_path}")

        # Clean up temporary ONNX file
        if onnx_path.exists():
            onnx_path.unlink()

    # Apply quantization if requested
    if quantize_int8 and quantize_int4:
        print("\nWarning: Both INT8 and INT4 quantization requested. Using INT8 (higher accuracy).")
        quantize_int4 = False

    if quantize_int8:
        if nncf is None:
            print("\nWarning: NNCF not available. Skipping INT8 quantization.")
            print("Install with: pip install nncf")
            return str(fp32_model_path)

        print("\n4. Applying INT8 quantization...")
        int8_model_path = quantize_model_int8(
            model=base_model,
            tokenizer=tokenizer,
            ov_model=ov_model,
            output_path=output_path,
            max_length=max_length,
            calibration_samples=calibration_samples
        )
        return int8_model_path

    if quantize_int4:
        if nncf is None:
            print("\nWarning: NNCF not available. Skipping INT4 quantization.")
            print("Install with: pip install nncf")
            return str(fp32_model_path)

        print("\n4. Applying INT4 weight quantization...")
        int4_model_path = quantize_model_int4(
            ov_model=ov_model,
            output_path=output_path
        )
        return int4_model_path

    return str(fp32_model_path)


def quantize_model_int8(
    model,
    tokenizer,
    ov_model,
    output_path: Path,
    max_length: int = 512,
    calibration_samples: int = 300
) -> str:
    """
    Quantize the OpenVINO model to INT8 using NNCF.

    Args:
        model: Original PyTorch model
        tokenizer: Tokenizer for the model
        ov_model: OpenVINO model
        output_path: Directory to save the quantized model
        max_length: Maximum sequence length
        calibration_samples: Number of calibration samples

    Returns:
        Path to the quantized model
    """
    print("  Creating calibration dataset...")

    # Create calibration dataset
    calibration_data = []

    # Sample texts for calibration (mix of different lengths and types)
    sample_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models use neural networks with multiple layers.",
        "Embeddings represent text as dense vectors in a continuous space.",
        "Transfer learning allows models to leverage knowledge from related tasks.",
        "Attention mechanisms help models focus on relevant parts of the input.",
        "Transformers have revolutionized natural language processing tasks.",
        "Pre-trained models can be fine-tuned for specific downstream tasks.",
        "Tokenization is the process of splitting text into smaller units.",
        "Semantic similarity measures how closely related two texts are in meaning.",
    ]

    # Generate calibration data by repeating and varying the samples
    for i in range(calibration_samples):
        text = sample_texts[i % len(sample_texts)]
        # Add some variation
        if i % 3 == 0:
            text = text + " Additional context for variation."
        elif i % 3 == 1:
            text = "Introduction: " + text

        inputs = tokenizer(
            text,
            return_tensors="np",
            max_length=max_length,
            padding="max_length",
            truncation=True
        )

        calibration_data.append({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })

    print(f"  Generated {len(calibration_data)} calibration samples")

    # Create calibration dataset generator
    def transform_fn(data_item):
        """Transform function for calibration."""
        return {
            "input_ids": data_item["input_ids"],
            "attention_mask": data_item["attention_mask"]
        }

    # Wrap calibration data
    calibration_dataset = nncf.Dataset(calibration_data, transform_fn)

    # Quantize the model
    print("  Quantizing model to INT8 (this may take a few minutes)...")
    try:
        # Use simpler quantization config for better compatibility
        quantized_model = nncf.quantize(
            ov_model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,  # MIXED for balance between speed and accuracy
            subset_size=min(calibration_samples, 300),
            model_type=nncf.ModelType.TRANSFORMER
        )

        # Save quantized model
        int8_model_path = output_path / "granite_embedding_int8.xml"
        ov.save_model(quantized_model, str(int8_model_path))
        print(f"✓ INT8 quantized model saved to {int8_model_path}")

        # Get file sizes for comparison
        fp32_path = output_path / "granite_embedding_fp32.xml"
        if fp32_path.exists():
            fp32_size = fp32_path.stat().st_size + (output_path / "granite_embedding_fp32.bin").stat().st_size
            int8_size = int8_model_path.stat().st_size + (output_path / "granite_embedding_int8.bin").stat().st_size

            print(f"\n  Model size comparison:")
            print(f"    FP32: {fp32_size / (1024*1024):.2f} MB")
            print(f"    INT8: {int8_size / (1024*1024):.2f} MB")
            print(f"    Compression ratio: {fp32_size / int8_size:.2f}x")

        return str(int8_model_path)

    except Exception as e:
        print(f"\n  ⚠ Error during quantization: {e}")
        print("\n  " + "=" * 70)
        print("  COMPATIBILITY ISSUE DETECTED")
        print("  " + "=" * 70)
        if "openvino.op" in str(e) or "ov.Node" in str(e):
            print("  NNCF 2.17.0 is NOT compatible with OpenVINO 2024.6.0+")
            print("\n  To enable INT8 quantization:")
            print("    Option 1: Downgrade OpenVINO")
            print("      pip install 'openvino==2024.3.0' 'openvino-dev==2024.3.0'")
            print("\n    Option 2: Use --no_quantize flag (FP32 only)")
            print("\n  FP32 model still provides significant speedup over PyTorch!")
        else:
            print(f"  Unexpected error during quantization")
            print(f"  Use --no_quantize flag to skip quantization")
        print("  " + "=" * 70)
        print("\n  Falling back to FP32 model...")
        return str(output_path / "granite_embedding_fp32.xml")


def quantize_model_int4(
    ov_model,
    output_path: Path
) -> str:
    """
    Quantize the OpenVINO model to INT4 using weight compression.

    INT4 quantization uses weight-only compression which provides:
    - Maximum compression (up to 4x smaller than FP32)
    - No calibration data required
    - Faster inference with minimal accuracy loss
    - Lower memory bandwidth requirements

    Args:
        ov_model: OpenVINO model
        output_path: Directory to save the quantized model

    Returns:
        Path to the quantized model
    """
    print("  Applying INT4 weight compression...")

    try:
        # Apply INT4 weight compression
        compressed_model = nncf.compress_weights(
            ov_model,
            mode=nncf.CompressWeightsMode.INT4_SYM,  # Symmetric INT4 quantization
        )

        # Save compressed model
        int4_model_path = output_path / "granite_embedding_int4.xml"
        ov.save_model(compressed_model, str(int4_model_path))
        print(f"✓ INT4 compressed model saved to {int4_model_path}")

        # Get file sizes for comparison
        fp32_path = output_path / "granite_embedding_fp32.xml"
        if fp32_path.exists():
            fp32_size = fp32_path.stat().st_size + (output_path / "granite_embedding_fp32.bin").stat().st_size
            int4_size = int4_model_path.stat().st_size + (output_path / "granite_embedding_int4.bin").stat().st_size

            print(f"\n  Model size comparison:")
            print(f"    FP32: {fp32_size / (1024*1024):.2f} MB")
            print(f"    INT4: {int4_size / (1024*1024):.2f} MB")
            print(f"    Compression ratio: {fp32_size / int4_size:.2f}x")

        return str(int4_model_path)

    except Exception as e:
        print(f"\n  ⚠ Error during INT4 compression: {e}")
        print("\n  Falling back to FP32 model...")
        return str(output_path / "granite_embedding_fp32.xml")


def test_openvino_model(model_path: str, tokenizer_path: str, model_type: str = "FP32") -> None:
    """
    Test the converted OpenVINO model to ensure it works correctly.

    Args:
        model_path: Path to the OpenVINO model (.xml file)
        tokenizer_path: Path to the tokenizer directory
        model_type: Type of model (FP32, INT8, or INT4)
    """
    print(f"\n5. Testing converted OpenVINO {model_type} model...")

    try:
        # Initialize OpenVINO runtime
        core = Core()

        # Load model
        model = core.read_model(model_path)
        compiled_model = core.compile_model(model, "CPU")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Test with sample texts
        test_texts = [
            "This is a test sentence for embedding generation.",
            "Another example text to verify the model works correctly.",
            "Machine learning models can be optimized with OpenVINO."
        ]

        print("Testing with sample texts...")
        for i, text in enumerate(test_texts):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=512
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

            # Pool embeddings (mean pooling, ignoring padding)
            attention_mask = inputs["attention_mask"]
            masked_embeddings = embeddings * attention_mask[:, :, np.newaxis]
            sum_embeddings = masked_embeddings.sum(axis=1)
            sum_mask = attention_mask.sum(axis=1, keepdims=True)
            pooled_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)

            print(f"  Text {i+1}: {text[:50]}...")
            print(f"    Embedding shape: {pooled_embeddings.shape}")
            print(f"    Embedding norm: {np.linalg.norm(pooled_embeddings):.4f}")

        print("✓ OpenVINO model test completed successfully!")

        # Benchmark inference speed
        print("\n6. Benchmarking inference speed...")
        import time

        # Warm-up
        for _ in range(10):
            compiled_model(input_data)

        # Benchmark
        num_iterations = 100
        start_time = time.time()
        for _ in range(num_iterations):
            compiled_model(input_data)
        end_time = time.time()

        avg_time = (end_time - start_time) / num_iterations * 1000
        throughput = 1000 / avg_time

        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  Throughput: {throughput:.2f} samples/sec")

    except Exception as e:
        print(f"Error testing OpenVINO model: {e}")
        import traceback
        traceback.print_exc()


def get_model_info(model_path: str) -> None:
    """
    Display information about the converted OpenVINO model.

    Args:
        model_path: Path to the OpenVINO model (.xml file)
    """
    print("\n7. Model Information:")
    print("-" * 60)

    try:
        core = Core()
        model = core.read_model(model_path)

        model_xml = Path(model_path)
        model_bin = model_xml.with_suffix('.bin')

        print(f"Model files:")
        print(f"  XML: {model_xml} ({model_xml.stat().st_size / 1024:.2f} KB)")
        print(f"  BIN: {model_bin} ({model_bin.stat().st_size / (1024*1024):.2f} MB)")
        print(f"\nTotal model size: {(model_xml.stat().st_size + model_bin.stat().st_size) / (1024*1024):.2f} MB")

        print(f"\nModel inputs:")
        for input_tensor in model.inputs:
            print(f"  - {input_tensor.any_name}: {input_tensor.shape} ({input_tensor.element_type})")

        print(f"\nModel outputs:")
        for output_tensor in model.outputs:
            print(f"  - {output_tensor.any_name}: {output_tensor.shape} ({output_tensor.element_type})")

        # List all files in the output directory
        output_dir = model_xml.parent
        print(f"\nFiles in output directory:")
        for file in sorted(output_dir.glob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024*1024)
                print(f"  - {file.name}: {size_mb:.2f} MB")

    except Exception as e:
        print(f"Error getting model info: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert IBM Granite embedding model to OpenVINO INT8/INT4 format"
    )
    parser.add_argument(
        "--model_name",
        default="ibm-granite/granite-embedding-english-r2",
        help="HuggingFace model name (default: ibm-granite/granite-embedding-english-r2)"
    )
    parser.add_argument(
        "--output_dir",
        default="./granite-embedding-openvino-int8",
        help="Output directory for OpenVINO model (default: ./granite-embedding-openvino-int8)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )

    # Quantization options
    quantize_group = parser.add_mutually_exclusive_group()
    quantize_group.add_argument(
        "--int8",
        action="store_true",
        help="Apply INT8 quantization (default if no option specified)"
    )
    quantize_group.add_argument(
        "--int4",
        action="store_true",
        help="Apply INT4 weight compression for maximum size reduction"
    )
    quantize_group.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable quantization (keep FP32)"
    )

    parser.add_argument(
        "--calibration_samples",
        type=int,
        default=300,
        help="Number of samples for calibration dataset (INT8 only, default: 300)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the converted model after conversion"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after conversion"
    )

    args = parser.parse_args()

    # Default to INT8 if no quantization option specified
    if not args.int8 and not args.int4 and not args.no_quantize:
        args.int8 = True

    print("=" * 70)
    print("IBM GRANITE EMBEDDING MODEL TO OPENVINO CONVERTER")
    print("=" * 70)

    # Check dependencies
    print("\nChecking dependencies...")
    try:
        from sentence_transformers import SentenceTransformer
        import sentence_transformers
        print(f"  ✓ SentenceTransformers version: {sentence_transformers.__version__}")
    except ImportError:
        print("  ✗ SentenceTransformers not found. Install with: pip install sentence-transformers")
        return 1

    try:
        import openvino as ov
        print(f"  ✓ OpenVINO version: {ov.__version__}")
    except ImportError:
        print("  ✗ OpenVINO not found. Install with: pip install openvino openvino-dev")
        return 1

    if args.int8 or args.int4:
        try:
            import nncf
            print(f"  ✓ NNCF version: {nncf.__version__}")
        except ImportError:
            print("  ✗ NNCF not found. Install with: pip install nncf")
            print("    Quantization will be disabled.")

    try:
        # Convert the model
        model_path = convert_to_openvino(
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_length=args.max_length,
            quantize_int8=args.int8,
            quantize_int4=args.int4,
            calibration_samples=args.calibration_samples
        )

        # Determine model type for testing
        if args.int8:
            model_type = "INT8"
        elif args.int4:
            model_type = "INT4"
        else:
            model_type = "FP32"

        # Test the model if requested
        if args.test or args.benchmark:
            test_openvino_model(
                model_path,
                args.output_dir,
                model_type=model_type
            )

        # Display model information
        get_model_info(model_path)

        print("\n" + "=" * 70)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Model saved to: {model_path}")
        print(f"\nYou can now use the OpenVINO model for optimized inference.")
        print("\nUsage example:")
        print(f"  from openvino.runtime import Core")
        print(f"  from transformers import AutoTokenizer")
        print(f"  ")
        print(f"  core = Core()")
        print(f"  model = core.read_model('{model_path}')")
        print(f"  compiled_model = core.compile_model(model, 'CPU')")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")
        print(f"  ")
        print(f"  # Tokenize and run inference")
        print(f"  inputs = tokenizer('Your text here', return_tensors='np')")
        print(f"  result = compiled_model(inputs)")

        return 0

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Conversion failed!")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
