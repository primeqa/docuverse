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

Conversion Paths (in order of preference):
1. SentenceTransformers OpenVINO backend (via optimum-intel) - recommended
   Handles ModernBert export correctly, avoids TorchScript tracing issues
2. Manual ONNX export + OpenVINO conversion - fallback
   Used when optimum-intel is not available

Quantization Paths (in order of preference):
1. optimum-intel OVWeightQuantizationConfig - recommended
   Works independently of NNCF/OpenVINO version compatibility
2. NNCF nncf.quantize() - fallback
   Requires compatible NNCF + OpenVINO versions (e.g., NNCF 2.x + OV 2024.x, or NNCF 3.x + OV 2025.x)
3. FP32 (no quantization) - final fallback

Requirements:
- sentence-transformers
- openvino
- optimum-intel (recommended, for primary conversion/quantization path)
- nncf (optional, fallback for quantization when optimum-intel is unavailable)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
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
    from openvino import Core, serialize

    # Apply compatibility monkey patch BEFORE importing NNCF
    # This fixes compatibility issues between NNCF 2.x and OpenVINO 2024.x
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

# Lazy-import NNCF (may not be installed)
nncf = None
try:
    import nncf as _nncf
    nncf = _nncf
except ImportError:
    pass


def _parse_major_version(version_str: str) -> Optional[int]:
    """Extract major version number from a version string like '3.1.0' or '2024.6.0'."""
    try:
        return int(version_str.split('.')[0])
    except (ValueError, IndexError):
        return None


def check_version_compatibility() -> Dict[str, object]:
    """
    Check installed package versions and detect known incompatibilities.

    Returns a dict with:
        - 'optimum_intel_available': bool
        - 'nncf_available': bool
        - 'nncf_ov_compatible': bool (True if NNCF and OpenVINO versions are compatible)
        - 'versions': dict of version strings
        - 'warnings': list of warning messages
    """
    result = {
        'optimum_intel_available': False,
        'nncf_available': nncf is not None,
        'nncf_ov_compatible': True,
        'versions': {},
        'warnings': [],
    }

    # OpenVINO version
    ov_version = getattr(ov, '__version__', 'unknown')
    result['versions']['openvino'] = ov_version

    # NNCF version
    if nncf is not None:
        nncf_version = getattr(nncf, '__version__', 'unknown')
        result['versions']['nncf'] = nncf_version

        # Detect NNCF 3.x + OpenVINO 2024.x mismatch
        nncf_major = _parse_major_version(nncf_version)
        ov_major = _parse_major_version(ov_version)

        if nncf_major is not None and ov_major is not None:
            if nncf_major >= 3 and ov_major < 2025:
                result['nncf_ov_compatible'] = False
                result['warnings'].append(
                    f"NNCF {nncf_version} (3.x) requires OpenVINO 2025.x+, "
                    f"but OpenVINO {ov_version} is installed.\n"
                    f"  NNCF quantization will fail with 'No module named openvino.op'.\n"
                    f"  Fix: pip install 'openvino>=2025.0' OR pip install 'nncf<3.0'"
                )
            elif nncf_major < 3 and ov_major >= 2025:
                result['nncf_ov_compatible'] = False
                result['warnings'].append(
                    f"NNCF {nncf_version} (2.x) may not be compatible with "
                    f"OpenVINO {ov_version} (2025.x+).\n"
                    f"  Fix: pip install 'nncf>=3.0' OR pip install 'openvino<2025.0'"
                )

    # Check optimum-intel availability
    try:
        import optimum.intel
        result['optimum_intel_available'] = True
        oi_version = getattr(optimum.intel, '__version__', 'unknown')
        result['versions']['optimum-intel'] = oi_version
    except ImportError:
        result['warnings'].append(
            "optimum-intel is not installed. The recommended conversion path is unavailable.\n"
            "  Install with: pip install optimum-intel"
        )

    # SentenceTransformers version
    try:
        import sentence_transformers
        result['versions']['sentence-transformers'] = sentence_transformers.__version__
    except Exception:
        pass

    return result


def convert_to_openvino(
    model_name: str = "ibm-granite/granite-embedding-english-r2",
    output_dir: str = "./granite-embedding-openvino-int8",
    max_length: int = 512,
    quantize_int8: bool = False,
    quantize_int4: bool = False,
    calibration_samples: int = 300,
    compat_info: Optional[Dict] = None
) -> Tuple[str, Optional[object]]:
    """
    Convert the Granite embedding model to OpenVINO IR format with optional quantization.

    Uses two conversion strategies:
    1. Primary: SentenceTransformers OpenVINO backend (via optimum-intel)
    2. Fallback: Manual ONNX export + OpenVINO conversion

    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save the OpenVINO model
        max_length: Maximum sequence length for the model
        quantize_int8: Whether to apply INT8 quantization (full model)
        quantize_int4: Whether to apply INT4 quantization (weights only)
        calibration_samples: Number of samples for calibration dataset (INT8 only)
        compat_info: Version compatibility info from check_version_compatibility()

    Returns:
        Tuple of (path to the converted model, SentenceTransformer model or None)
    """
    if compat_info is None:
        compat_info = check_version_compatibility()

    print(f"Converting {model_name} to OpenVINO IR format...")
    print(f"Output directory: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if quantize_int8 and quantize_int4:
        print("\nWarning: Both INT8 and INT4 quantization requested. Using INT8 (higher accuracy).")
        quantize_int4 = False

    # ---- Primary path: SentenceTransformers OpenVINO backend ----
    st_model = None
    used_st_backend = False

    if compat_info['optimum_intel_available']:
        print("\n1. Loading model with SentenceTransformers OpenVINO backend...")
        try:
            st_model = SentenceTransformer(
                model_name,
                backend="openvino",
                model_kwargs={'attn_implementation': 'eager'}
            )

            # Apply quantization via optimum-intel if requested
            if quantize_int8 or quantize_int4:
                print("\n2. Applying quantization via optimum-intel...")
                try:
                    from optimum.intel import OVWeightQuantizationConfig
                    from sentence_transformers import export_static_quantized_openvino_model

                    if quantize_int8:
                        quant_config = OVWeightQuantizationConfig(
                            bits=8,
                            sym=True,
                            dataset=None,
                        )
                        print("  Applying INT8 weight quantization...")
                    else:
                        quant_config = OVWeightQuantizationConfig(
                            bits=4,
                            sym=True,
                            group_size=128,
                            ratio=0.8,
                            dataset=None,
                        )
                        print("  Applying INT4 weight quantization...")

                    export_static_quantized_openvino_model(
                        st_model,
                        quantization_config=quant_config,
                        model_name_or_path=output_dir,
                    )
                    print("  Quantization applied successfully.")

                except Exception as e:
                    print(f"\n  Warning: optimum-intel quantization failed: {e}")
                    print("  Model will be saved in FP32 format.")

            print(f"\n{'3' if (quantize_int8 or quantize_int4) else '2'}. Saving model to {output_dir}...")
            st_model.save_pretrained(output_dir)
            used_st_backend = True
            print(f"  Model saved to {output_dir}")

            # The ST OpenVINO backend saves the model in its own layout;
            # return the output_dir as the model path.
            return str(output_dir), st_model

        except Exception as e:
            print(f"\n  Warning: SentenceTransformers OpenVINO backend failed: {e}")
            print("  Falling back to manual ONNX export path...")
            st_model = None
    else:
        print("\n  optimum-intel not available; using manual ONNX export path.")

    # ---- Fallback path: manual export via ONNX ----
    print("\n1. Loading model with SentenceTransformers (PyTorch)...")

    try:
        st_model = SentenceTransformer(
            model_name,
            device='cpu',
            model_kwargs={
                'dtype': torch.float32,
                'attn_implementation': 'eager'
            }
        )
    except TypeError:
        st_model = SentenceTransformer(
            model_name,
            device='cpu',
            model_kwargs={
                'torch_dtype': torch.float32,
                'attn_implementation': 'eager'
            }
        )

    st_model.eval()
    tokenizer = st_model.tokenizer

    # Save the SentenceTransformer model (includes tokenizer, pooling config, etc.)
    st_model.save_pretrained(output_dir)
    print(f"  SentenceTransformer model and tokenizer saved to {output_dir}")

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

    # Extract the underlying transformer model
    if hasattr(st_model, '_modules') and '0' in st_model._modules:
        transformer_module = st_model._modules['0']
        if hasattr(transformer_module, 'auto_model'):
            base_model = transformer_module.auto_model
        else:
            base_model = st_model
    else:
        base_model = st_model

    # Convert to OpenVINO format
    print("\n3. Converting to OpenVINO IR format...")
    try:
        input_dict = {
            "input_ids": example_inputs["input_ids"],
            "attention_mask": example_inputs["attention_mask"]
        }

        ov_model = ov.convert_model(
            base_model,
            example_input=input_dict,
            input=[
                ("input_ids", [1, max_length], np.int64),
                ("attention_mask", [1, max_length], np.int64)
            ]
        )

        fp32_model_path = output_path / "granite_embedding_fp32.xml"
        ov.save_model(ov_model, str(fp32_model_path))
        print(f"  FP32 model saved to {fp32_model_path}")

    except Exception as e:
        print(f"  Error during OpenVINO conversion: {e}")
        print("\n  Trying alternative conversion method via ONNX...")

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

        print(f"  ONNX model created at {onnx_path}")

        ov_model = ov.convert_model(str(onnx_path))
        fp32_model_path = output_path / "granite_embedding_fp32.xml"
        ov.save_model(ov_model, str(fp32_model_path))
        print(f"  OpenVINO FP32 model saved to {fp32_model_path}")

        if onnx_path.exists():
            onnx_path.unlink()

    # Apply quantization if requested (fallback NNCF path)
    if quantize_int8:
        quantized_path = _quantize_nncf_int8(
            base_model, tokenizer, ov_model, output_path,
            max_length, calibration_samples, compat_info
        )
        if quantized_path:
            return quantized_path, None

    if quantize_int4:
        quantized_path = _quantize_nncf_int4(ov_model, output_path, compat_info)
        if quantized_path:
            return quantized_path, None

    return str(fp32_model_path), None


def _quantize_nncf_int8(
    model, tokenizer, ov_model, output_path: Path,
    max_length: int, calibration_samples: int,
    compat_info: Dict
) -> Optional[str]:
    """
    Quantize to INT8 using NNCF (fallback path when optimum-intel is unavailable).

    Returns the quantized model path, or None if quantization failed.
    """
    if nncf is None:
        print("\n  Warning: NNCF not available. Skipping INT8 quantization.")
        print("  Install with: pip install nncf")
        return None

    if not compat_info.get('nncf_ov_compatible', True):
        nncf_ver = compat_info['versions'].get('nncf', 'unknown')
        ov_ver = compat_info['versions'].get('openvino', 'unknown')
        print(f"\n  Warning: NNCF {nncf_ver} is not compatible with OpenVINO {ov_ver}.")
        print("  Skipping NNCF quantization. Falling back to FP32.")
        for w in compat_info.get('warnings', []):
            if 'nncf' in w.lower() or 'NNCF' in w:
                print(f"  {w}")
        return None

    print("\n4. Applying INT8 quantization via NNCF...")
    return quantize_model_int8(
        model=model,
        tokenizer=tokenizer,
        ov_model=ov_model,
        output_path=output_path,
        max_length=max_length,
        calibration_samples=calibration_samples
    )


def _quantize_nncf_int4(
    ov_model, output_path: Path, compat_info: Dict
) -> Optional[str]:
    """
    Quantize to INT4 using NNCF (fallback path when optimum-intel is unavailable).

    Returns the quantized model path, or None if quantization failed.
    """
    if nncf is None:
        print("\n  Warning: NNCF not available. Skipping INT4 quantization.")
        print("  Install with: pip install nncf")
        return None

    if not compat_info.get('nncf_ov_compatible', True):
        nncf_ver = compat_info['versions'].get('nncf', 'unknown')
        ov_ver = compat_info['versions'].get('openvino', 'unknown')
        print(f"\n  Warning: NNCF {nncf_ver} is not compatible with OpenVINO {ov_ver}.")
        print("  Skipping NNCF quantization. Falling back to FP32.")
        return None

    print("\n4. Applying INT4 weight quantization via NNCF...")
    return quantize_model_int4(ov_model=ov_model, output_path=output_path)


def quantize_model_int8(
    model,
    tokenizer,
    ov_model,
    output_path: Path,
    max_length: int = 512,
    calibration_samples: int = 300
) -> Optional[str]:
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
        Path to the quantized model, or None if quantization failed
    """
    print("  Creating calibration dataset...")

    calibration_data = []

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

    for i in range(calibration_samples):
        text = sample_texts[i % len(sample_texts)]
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

    def transform_fn(data_item):
        return {
            "input_ids": data_item["input_ids"],
            "attention_mask": data_item["attention_mask"]
        }

    calibration_dataset = nncf.Dataset(calibration_data, transform_fn)

    print("  Quantizing model to INT8...")
    try:
        quantized_model = nncf.quantize(
            ov_model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=min(calibration_samples, 300),
            model_type=nncf.ModelType.TRANSFORMER
        )

        int8_model_path = output_path / "granite_embedding_int8.xml"
        ov.save_model(quantized_model, str(int8_model_path))
        print(f"  INT8 quantized model saved to {int8_model_path}")

        # Report size comparison
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
        nncf_ver = getattr(nncf, '__version__', 'unknown')
        ov_ver = getattr(ov, '__version__', 'unknown')

        print(f"\n  Error during NNCF quantization: {e}")
        print("\n  " + "=" * 70)
        print("  COMPATIBILITY ISSUE DETECTED")
        print("  " + "=" * 70)
        if "openvino.op" in str(e) or "ov.Node" in str(e):
            print(f"  NNCF {nncf_ver} is not compatible with OpenVINO {ov_ver}.")
            print(f"\n  To enable INT8 quantization:")
            print(f"    Option 1: Install optimum-intel (recommended)")
            print(f"      pip install optimum-intel")
            print(f"\n    Option 2: Use compatible NNCF + OpenVINO versions")
            print(f"      pip install 'nncf<3.0' 'openvino<2025.0'  # or")
            print(f"      pip install 'nncf>=3.0' 'openvino>=2025.0'")
        else:
            print(f"  Unexpected error during quantization (NNCF {nncf_ver}, OpenVINO {ov_ver})")
            print(f"  Use --no_quantize flag to skip quantization")
        print("  " + "=" * 70)
        print("\n  Falling back to FP32 model...")
        return None


def quantize_model_int4(
    ov_model,
    output_path: Path
) -> Optional[str]:
    """
    Quantize the OpenVINO model to INT4 using weight compression.

    Args:
        ov_model: OpenVINO model
        output_path: Directory to save the quantized model

    Returns:
        Path to the quantized model, or None if quantization failed
    """
    print("  Applying INT4 weight compression...")

    try:
        compressed_model = nncf.compress_weights(
            ov_model,
            mode=nncf.CompressWeightsMode.INT4_SYM,
        )

        int4_model_path = output_path / "granite_embedding_int4.xml"
        ov.save_model(compressed_model, str(int4_model_path))
        print(f"  INT4 compressed model saved to {int4_model_path}")

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
        print(f"\n  Error during INT4 compression: {e}")
        print("\n  Falling back to FP32 model...")
        return None


def test_openvino_model(model_path: str, tokenizer_path: str, model_type: str = "FP32",
                        st_ov_model: Optional[object] = None) -> None:
    """
    Test the converted OpenVINO model to ensure it works correctly.

    Uses SentenceTransformers for reference embeddings and comparison.

    Args:
        model_path: Path to the OpenVINO model (.xml file) or output directory
          (if model was saved via the SentenceTransformers OpenVINO backend)
        tokenizer_path: Path to the tokenizer directory (contains saved SentenceTransformer)
        model_type: Type of model (FP32, INT8, or INT4)
        st_ov_model: Pre-loaded SentenceTransformer with OpenVINO backend (if available)
    """
    print(f"\n5. Testing converted OpenVINO {model_type} model...")

    # Determine whether to use the ST OpenVINO backend for testing
    use_st_backend = st_ov_model is not None

    if use_st_backend:
        # Test via SentenceTransformer.encode() - cleaner and more representative
        _test_via_st_backend(st_ov_model, tokenizer_path, model_type)
    else:
        # Test via raw OpenVINO inference
        _test_via_raw_openvino(model_path, tokenizer_path, model_type)


def _test_via_st_backend(st_ov_model, tokenizer_path: str, model_type: str) -> None:
    """Test using SentenceTransformer OpenVINO backend (encode-level comparison)."""
    try:
        test_texts = [
            "This is a test sentence for embedding generation.",
            "Another example text to verify the model works correctly.",
            "Machine learning models can be optimized with OpenVINO."
        ]

        # Get OpenVINO embeddings
        print("  Computing embeddings with OpenVINO model...")
        ov_embeddings = st_ov_model.encode(test_texts, convert_to_numpy=True)

        # Load a reference PyTorch model for comparison
        print("  Loading PyTorch reference model for comparison...")
        try:
            ref_model = SentenceTransformer(
                tokenizer_path,
                device='cpu',
                model_kwargs={
                    'dtype': torch.float32,
                    'attn_implementation': 'eager'
                }
            )
        except (TypeError, Exception):
            # If we can't load from the output dir (it may only have OV files),
            # skip the comparison
            print("  Could not load PyTorch reference model; skipping comparison.")
            for i, text in enumerate(test_texts):
                emb = ov_embeddings[i]
                print(f"  Text {i+1}: {text[:50]}...")
                print(f"    Embedding norm: {np.linalg.norm(emb):.4f}")
                print(f"    Embedding dim: {emb.shape[0]}")
            print("  OpenVINO model test completed (no reference comparison).")
            return

        ref_embeddings = ref_model.encode(test_texts, convert_to_numpy=True)

        similarities = []
        for i, text in enumerate(test_texts):
            ov_emb = ov_embeddings[i]
            ref_emb = ref_embeddings[i]
            ov_norm = np.linalg.norm(ov_emb)
            ref_norm = np.linalg.norm(ref_emb)
            cosine_sim = np.dot(ov_emb, ref_emb) / (ov_norm * ref_norm)
            similarities.append(cosine_sim)

            print(f"  Text {i+1}: {text[:50]}...")
            print(f"    OpenVINO embedding norm: {ov_norm:.4f}")
            print(f"    Reference embedding norm: {ref_norm:.4f}")
            print(f"    Cosine similarity: {cosine_sim:.6f}")

        avg_similarity = np.mean(similarities)
        print(f"\n  Average cosine similarity: {avg_similarity:.6f}")
        if avg_similarity > 0.99:
            print("  Excellent match with reference model!")
        elif avg_similarity > 0.95:
            print("  Good match with reference model")
        else:
            print(f"  Warning: Lower similarity ({avg_similarity:.4f}) - may indicate quantization effects")

        print("  OpenVINO model test completed successfully!")

        # Benchmark
        print("\n6. Benchmarking inference speed...")
        import time

        bench_text = test_texts[0]
        # Warm-up
        for _ in range(10):
            st_ov_model.encode([bench_text])

        num_iterations = 100
        start_time = time.time()
        for _ in range(num_iterations):
            st_ov_model.encode([bench_text])
        end_time = time.time()

        avg_time = (end_time - start_time) / num_iterations * 1000
        throughput = 1000 / avg_time

        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  Throughput: {throughput:.2f} samples/sec")

    except Exception as e:
        print(f"Error testing OpenVINO model: {e}")
        import traceback
        traceback.print_exc()


def _test_via_raw_openvino(model_path: str, tokenizer_path: str, model_type: str) -> None:
    """Test using raw OpenVINO Core inference (fallback path)."""
    try:
        core = Core()

        # Load OpenVINO model
        ov_model = core.read_model(model_path)
        compiled_model = core.compile_model(ov_model, "CPU")

        # Load SentenceTransformer for reference and tokenization
        print("  Loading SentenceTransformer for reference...")
        try:
            st_model = SentenceTransformer(
                tokenizer_path,
                device='cpu',
                model_kwargs={
                    'dtype': torch.float32,
                    'attn_implementation': 'eager'
                }
            )
        except TypeError:
            st_model = SentenceTransformer(
                tokenizer_path,
                device='cpu',
                model_kwargs={
                    'torch_dtype': torch.float32,
                    'attn_implementation': 'eager'
                }
            )
        st_model.eval()

        tokenizer = st_model.tokenizer

        test_texts = [
            "This is a test sentence for embedding generation.",
            "Another example text to verify the model works correctly.",
            "Machine learning models can be optimized with OpenVINO."
        ]

        print("Testing with sample texts and comparing with SentenceTransformer...")
        similarities = []

        for i, text in enumerate(test_texts):
            st_embedding = st_model.encode([text], convert_to_numpy=True, normalize_embeddings=False)[0]

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

            result = compiled_model(input_data)

            output_key = list(result.keys())[0]
            embeddings = result[output_key]

            # CLS token pooling
            ov_embedding = embeddings[0, 0, :]

            ov_norm = np.linalg.norm(ov_embedding)
            st_norm = np.linalg.norm(st_embedding)
            cosine_sim = np.dot(ov_embedding, st_embedding) / (ov_norm * st_norm)
            similarities.append(cosine_sim)

            print(f"  Text {i+1}: {text[:50]}...")
            print(f"    OpenVINO embedding norm: {ov_norm:.4f}")
            print(f"    SentenceTransformer norm: {st_norm:.4f}")
            print(f"    Cosine similarity: {cosine_sim:.6f}")

        avg_similarity = np.mean(similarities)
        print(f"\n  Average cosine similarity: {avg_similarity:.6f}")
        if avg_similarity > 0.99:
            print("  Excellent match with SentenceTransformer!")
        elif avg_similarity > 0.95:
            print("  Good match with SentenceTransformer")
        else:
            print(f"  Warning: Lower similarity ({avg_similarity:.4f}) - may indicate quantization effects")

        print("  OpenVINO model test completed successfully!")

        # Benchmark
        print("\n6. Benchmarking inference speed...")
        import time

        for _ in range(10):
            compiled_model(input_data)

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
        model_path: Path to the OpenVINO model (.xml file) or output directory
    """
    print("\n7. Model Information:")
    print("-" * 60)

    model_xml = Path(model_path)

    # If model_path is a directory (ST OpenVINO backend), look for the OV model inside
    if model_xml.is_dir():
        # ST OpenVINO backend saves under openvino/openvino_model.xml
        candidates = [
            model_xml / "openvino" / "openvino_model.xml",
            model_xml / "openvino_model.xml",
        ]
        # Also try any .xml file
        candidates.extend(sorted(model_xml.rglob("*.xml")))

        found = False
        for candidate in candidates:
            if candidate.exists():
                model_xml = candidate
                found = True
                break

        if not found:
            print(f"  No .xml model file found in {model_path}")
            # Still list files in the directory
            print(f"\n  Files in output directory:")
            for file in sorted(Path(model_path).rglob("*")):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024*1024)
                    rel = file.relative_to(model_path)
                    print(f"    - {rel}: {size_mb:.2f} MB")
            return

    try:
        core = Core()
        model = core.read_model(str(model_xml))

        model_bin = model_xml.with_suffix('.bin')

        print(f"Model files:")
        print(f"  XML: {model_xml} ({model_xml.stat().st_size / 1024:.2f} KB)")
        if model_bin.exists():
            print(f"  BIN: {model_bin} ({model_bin.stat().st_size / (1024*1024):.2f} MB)")
            print(f"\nTotal model size: {(model_xml.stat().st_size + model_bin.stat().st_size) / (1024*1024):.2f} MB")

        print(f"\nModel inputs:")
        for input_tensor in model.inputs:
            print(f"  - {input_tensor.any_name}: {input_tensor.shape} ({input_tensor.element_type})")

        print(f"\nModel outputs:")
        for output_tensor in model.outputs:
            print(f"  - {output_tensor.any_name}: {output_tensor.shape} ({output_tensor.element_type})")

        # List files in the top-level output directory
        output_dir = Path(model_path) if Path(model_path).is_dir() else model_xml.parent
        print(f"\nFiles in output directory:")
        for file in sorted(output_dir.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024*1024)
                rel = file.relative_to(output_dir)
                print(f"  - {rel}: {size_mb:.2f} MB")

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

    # Check dependencies and version compatibility
    print("\nChecking dependencies...")
    try:
        import sentence_transformers
        print(f"  sentence-transformers: {sentence_transformers.__version__}")
    except ImportError:
        print("  sentence-transformers: NOT FOUND")
        print("    Install with: pip install sentence-transformers")
        return 1

    print(f"  OpenVINO: {ov.__version__}")

    compat_info = check_version_compatibility()

    if compat_info['optimum_intel_available']:
        print(f"  optimum-intel: {compat_info['versions'].get('optimum-intel', 'unknown')}")
    else:
        print(f"  optimum-intel: NOT FOUND (recommended: pip install optimum-intel)")

    if nncf is not None:
        print(f"  NNCF: {nncf.__version__}")
    else:
        print(f"  NNCF: NOT FOUND")

    # Print compatibility warnings
    if compat_info['warnings']:
        print("\n  Compatibility warnings:")
        for w in compat_info['warnings']:
            for line in w.split('\n'):
                print(f"    {line}")

    # Determine conversion strategy
    if compat_info['optimum_intel_available']:
        print(f"\n  Strategy: SentenceTransformers OpenVINO backend (via optimum-intel)")
    elif nncf is not None and compat_info['nncf_ov_compatible']:
        print(f"\n  Strategy: Manual ONNX export + NNCF quantization (fallback)")
    else:
        quant_note = " (quantization unavailable)" if (args.int8 or args.int4) else ""
        print(f"\n  Strategy: Manual ONNX export + FP32{quant_note}")

    try:
        # Convert the model
        model_path, st_ov_model = convert_to_openvino(
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_length=args.max_length,
            quantize_int8=args.int8,
            quantize_int4=args.int4,
            calibration_samples=args.calibration_samples,
            compat_info=compat_info
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
                model_type=model_type,
                st_ov_model=st_ov_model
            )

        # Display model information
        get_model_info(model_path)

        print("\n" + "=" * 70)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Model saved to: {model_path}")
        print(f"\nUsage example:")
        if st_ov_model is not None:
            print(f'  from sentence_transformers import SentenceTransformer')
            print(f'  model = SentenceTransformer("{args.output_dir}", backend="openvino")')
            print(f'  embeddings = model.encode(["Your text here"])')
        else:
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
