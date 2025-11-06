#!/usr/bin/env python3
"""
Export Granite embedding models to OpenVINO or ONNX format.
This script patches the transformers library at runtime to skip flash_attn imports,
enabling export of ModernBERT-based models.
"""

import sys
import os

# Step 1: Set environment variables BEFORE any imports
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings

# Step 2: Patch transformers import_utils BEFORE importing transformers
def patch_flash_attn_check():
    """Patch the flash_attn availability check to always return False."""
    # Import the module
    from transformers import utils

    # Patch the function
    original_is_flash_attn_2_available = utils.is_flash_attn_2_available

    def patched_is_flash_attn_2_available():
        return False

    utils.is_flash_attn_2_available = patched_is_flash_attn_2_available

    # Also patch in the import_utils module directly
    from transformers.utils import import_utils
    import_utils.is_flash_attn_2_available = patched_is_flash_attn_2_available

    # Patch at module level
    import transformers
    transformers.is_flash_attn_2_available = patched_is_flash_attn_2_available

    print("✓ Patched flash_attn availability check")


# Step 3: Create mock flash_attn modules for fallback imports
class MockFlashAttnModule:
    """Mock module to provide fallbacks for flash_attn imports."""

    def __init__(self, name):
        self.name = name
        # Add __path__ to make it look like a package
        self.__path__ = []
        self.__file__ = f"<mock {name}>"

    def __getattr__(self, item):
        # For specific needed classes, provide mocks
        if item == 'RotaryEmbedding':
            # Return a class that will use torch's rotary instead
            import torch.nn as nn
            return nn.Identity  # Fallback to identity
        elif item == 'apply_rotary':
            # Return a no-op function with proper signature
            def apply_rotary(x, cos, sin, interleaved=False, seqlen_offsets=0):
                return x
            return apply_rotary
        elif item == 'flash_attn_varlen_qkvpacked_func':
            # Return a no-op function with proper signature to avoid inspection issues
            def flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False
            ):
                raise NotImplementedError("Flash attention not available, use eager attention")
            return flash_attn_varlen_qkvpacked_func
        elif item == 'pad_input' or item == 'unpad_input':
            # Mock bert_padding functions with proper signatures
            def pad_input(hidden_states, attention_mask):
                return hidden_states, None, None, None, None, None
            def unpad_input(hidden_states, attention_mask):
                return hidden_states, None, None
            return pad_input if item == 'pad_input' else unpad_input
        elif item == '__path__':
            return []
        elif item == '__file__':
            return f"<mock {self.name}>"
        elif item == '__spec__':
            # Create a minimal spec
            from types import SimpleNamespace
            return SimpleNamespace(
                name=self.name,
                loader=None,
                origin=None,
                submodule_search_locations=[]
            )
        else:
            # For anything else, return another mock
            return MockFlashAttnModule(f"{self.name}.{item}")

    def __call__(self, *args, **kwargs):
        return MockFlashAttnModule(f"{self.name}()")


def install_flash_attn_mocks():
    """Install mock flash_attn modules."""
    mock = MockFlashAttnModule('flash_attn')
    sys.modules['flash_attn'] = mock
    sys.modules['flash_attn.layers'] = MockFlashAttnModule('flash_attn.layers')
    sys.modules['flash_attn.layers.rotary'] = MockFlashAttnModule('flash_attn.layers.rotary')
    sys.modules['flash_attn.ops'] = MockFlashAttnModule('flash_attn.ops')
    sys.modules['flash_attn.ops.triton'] = MockFlashAttnModule('flash_attn.ops.triton')
    sys.modules['flash_attn.ops.triton.rotary'] = MockFlashAttnModule('flash_attn.ops.triton.rotary')
    sys.modules['flash_attn.flash_attn_interface'] = MockFlashAttnModule('flash_attn.flash_attn_interface')
    sys.modules['flash_attn.bert_padding'] = MockFlashAttnModule('flash_attn.bert_padding')
    print("✓ Installed flash_attn mock modules")


# Install patches at module level BEFORE any other imports
print("Installing patches...")
patch_flash_attn_check()
install_flash_attn_mocks()

# NOW import everything else after patches are installed
import argparse
import torch
import numpy as np


def generate_test_sentences(num_sentences=50):
    """Generate diverse test sentences for validation."""
    sentences = [
        # Technical / AI content
        "Machine learning models can process natural language with high accuracy.",
        "Deep neural networks learn hierarchical representations from data.",
        "Transformer architectures revolutionized natural language processing tasks.",
        "Embedding models convert text into dense vector representations.",
        "OpenVINO optimizes inference performance on Intel hardware platforms.",

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
        "In a groundbreaking study published today, researchers have discovered a novel approach to improving the efficiency of renewable energy systems through advanced materials science.",
        "The conference brought together experts from various fields including computer science, biology, and engineering to discuss interdisciplinary approaches to solving global challenges.",
        "Climate change represents one of the most significant challenges facing humanity, requiring coordinated international efforts to reduce greenhouse gas emissions and develop sustainable technologies.",

        # Questions
        "What is the capital of France?",
        "How does machine learning work?",
        "Why is the sky blue?",
        "When was the internet invented?",
        "Where can I find more information?",

        # Domain-specific
        "The convolutional neural network achieved 95% accuracy on the test set.",
        "Quantization reduces model size while maintaining acceptable performance.",
        "Attention mechanisms allow models to focus on relevant input features.",
        "Transfer learning enables models to leverage pre-trained knowledge.",
        "Batch normalization stabilizes training of deep neural networks.",

        # Conversational
        "I really enjoyed the presentation you gave yesterday.",
        "Could you please send me the report by end of day?",
        "Let's schedule a meeting to discuss the project timeline.",
        "I appreciate your help with debugging that issue.",
        "The new feature is working great, thanks for implementing it!",

        # Varied length and complexity
        "AI",
        "Artificial Intelligence",
        "Artificial Intelligence and Machine Learning",
        "Artificial Intelligence, Machine Learning, and Deep Learning are transforming industries.",
        "The rapid advancement of artificial intelligence and machine learning technologies is fundamentally reshaping how we approach complex problems.",

        # Similar semantic content (for testing similarity)
        "Dogs are loyal pets.",
        "Canines make faithful companions.",
        "The weather is beautiful today.",
        "It's a gorgeous day outside.",
        "Programming requires logical thinking.",
        "Coding demands analytical reasoning.",

        # Negations and contrasts
        "I like coffee but not tea.",
        "The model is fast but not very accurate.",
        "This approach works well for small datasets but fails on large ones.",
        "Quantization reduces size but may decrease quality.",
    ]

    # Return exactly num_sentences
    if len(sentences) >= num_sentences:
        return sentences[:num_sentences]
    else:
        # Repeat sentences if needed
        while len(sentences) < num_sentences:
            sentences.append(f"Test sentence number {len(sentences) + 1}.")
        return sentences[:num_sentences]


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_model_similarity(openvino_model, original_model, test_sentences, output_dir):
    """
    Test similarity between OpenVINO and original PyTorch models.

    Args:
        openvino_model: The exported OpenVINO model
        original_model: The original PyTorch model (or None to skip comparison)
        test_sentences: List of test sentences
        output_dir: Output directory path
    """
    print(f"\n{'='*70}")
    print("MODEL VALIDATION AND COMPARISON")
    print(f"{'='*70}\n")

    print(f"Testing with {len(test_sentences)} text segments...")
    print()

    # Get OpenVINO embeddings
    print("1. Computing embeddings with OpenVINO model...")
    ov_embeddings = openvino_model.encode(test_sentences, show_progress_bar=False)
    print(f"   ✓ OpenVINO embeddings shape: {ov_embeddings.shape}")

    if original_model is not None:
        # Get original embeddings
        print("\n2. Computing embeddings with original PyTorch model...")
        orig_embeddings = original_model.encode(test_sentences, show_progress_bar=False)
        print(f"   ✓ Original embeddings shape: {orig_embeddings.shape}")

        # Calculate cosine similarities
        print("\n3. Computing cosine similarities between models...")
        similarities = []
        for i in range(len(test_sentences)):
            sim = cosine_similarity(ov_embeddings[i], orig_embeddings[i])
            similarities.append(sim)

        similarities = np.array(similarities)

        # Report statistics
        print(f"\n{'='*70}")
        print("SIMILARITY STATISTICS (OpenVINO vs Original)")
        print(f"{'='*70}")
        print(f"  Mean cosine similarity:   {similarities.mean():.6f}")
        print(f"  Median cosine similarity: {np.median(similarities):.6f}")
        print(f"  Min cosine similarity:    {similarities.min():.6f}")
        print(f"  Max cosine similarity:    {similarities.max():.6f}")
        print(f"  Std deviation:            {similarities.std():.6f}")
        print()

        # Show distribution
        bins = [0.90, 0.95, 0.97, 0.99, 0.995, 1.0]
        print("Distribution:")
        for i in range(len(bins)-1):
            count = np.sum((similarities >= bins[i]) & (similarities < bins[i+1]))
            pct = 100 * count / len(similarities)
            print(f"  [{bins[i]:.3f} - {bins[i+1]:.3f}): {count:3d} samples ({pct:5.1f}%)")

        # Show worst matches
        print(f"\n{'='*70}")
        print("LOWEST SIMILARITY SAMPLES")
        print(f"{'='*70}")
        worst_indices = np.argsort(similarities)[:5]
        for idx in worst_indices:
            print(f"\nSimilarity: {similarities[idx]:.6f}")
            print(f"Text: \"{test_sentences[idx][:80]}{'...' if len(test_sentences[idx]) > 80 else ''}\"")
    else:
        print("\n   (Skipping comparison with original model)")

    # Self-similarity test for OpenVINO model
    print(f"\n{'='*70}")
    print("SELF-SIMILARITY TEST (OpenVINO model)")
    print(f"{'='*70}")
    print("\nComputing pairwise similarities for similar sentence pairs...")

    # Define similar pairs (if they exist in test set)
    similar_pairs = [
        ("Dogs are loyal pets.", "Canines make faithful companions."),
        ("The weather is beautiful today.", "It's a gorgeous day outside."),
        ("Programming requires logical thinking.", "Coding demands analytical reasoning."),
    ]

    for text1, text2 in similar_pairs:
        try:
            idx1 = test_sentences.index(text1)
            idx2 = test_sentences.index(text2)
            sim = cosine_similarity(ov_embeddings[idx1], ov_embeddings[idx2])
            print(f"\n  Similarity: {sim:.4f}")
            print(f"  Text 1: \"{text1}\"")
            print(f"  Text 2: \"{text2}\"")
        except ValueError:
            # Sentence not in test set
            pass


def main():
    parser = argparse.ArgumentParser(description='Export Granite model to OpenVINO or ONNX format')
    parser.add_argument('--model', type=str, default="ibm-granite/granite-embedding-small-english-r2",
                        help='Model name or path')
    parser.add_argument('--output', type=str, default="granite-small-exported",
                        help='Output directory for the exported model')
    parser.add_argument('--format', choices=['openvino', 'onnx'], default='openvino',
                        help='Export format (openvino or onnx)')
    parser.add_argument('--quantization', choices=['none', 'int8', 'int4'], default='none',
                        help='Quantization mode (none, int8, or int4). For OpenVINO: int8/int4. For ONNX: int8 only (dynamic quantization)')
    parser.add_argument('--compare-with-original', action='store_true',
                        help='Compare exported model with original PyTorch model (slower)')
    parser.add_argument("--onnx_config", type=str,
                        choices=["arm64", "avx2", "avx512", "avx512_vnni"],
                        default="avx2",
                        help="ONNX quantization config",)
    args = parser.parse_args()

    print("=" * 70)
    print(f"GRANITE MODEL EXPORT TO {args.format.upper()}")
    print("=" * 70)
    print()

    # Import SentenceTransformer (after patches were already installed at module level)
    from sentence_transformers import SentenceTransformer

    model_name = args.model
    output_dir = args.output

    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Format: {args.format}")
    if args.quantization != 'none':
        print(f"Quantization: {args.quantization}")
    print()

    try:
        if args.format == 'openvino':
            # Import OpenVINO-specific modules only when needed
            from optimum.intel import OVWeightQuantizationConfig
            from sentence_transformers import export_static_quantized_openvino_model

            print("Step 1: Loading model with OpenVINO backend (will trigger export)...")
            print("This may take several minutes...")
            print()

            # Load with OpenVINO backend - this will export the model
            quantization_config = None
            if args.quantization != 'none':
                if args.quantization == 'int8':
                    # INT8: per-channel quantization, ratio=1.0 and group_size=-1 are required
                    quantization_config = OVWeightQuantizationConfig(
                        bits=8,
                        sym=True,
                    )
                else:  # int4
                    # INT4: can use group-wise quantization for better quality
                    quantization_config = OVWeightQuantizationConfig(
                        bits=4,
                        sym=True,
                        group_size=128,  # Smaller groups = better quality
                        ratio=0.8,  # Quantize 80% of layers
                    )

            model = SentenceTransformer(
                model_name,
                backend="openvino",
                model_kwargs={
                    'attn_implementation': 'eager',
                }
            )

            if quantization_config is not None:
                export_static_quantized_openvino_model(
                    model,
                    quantization_config=quantization_config,
                    model_name_or_path=output_dir,
                )

            print(f"Step 2: Saving to {output_dir}...")
            model.save_pretrained(output_dir)
            print("✓ Model saved!")
            print("✓ Model loaded and exported successfully!")
            print()

        elif args.format == 'onnx':
            print("Step 1: Loading model with ONNX backend (will trigger export)...")
            print("This may take a few minutes...")
            print()

            # Load with ONNX backend - this will automatically export to ONNX
            model = SentenceTransformer(model_name, backend="onnx")
            print("✓ Model loaded and exported to ONNX successfully!")
            print()

            if args.quantization == 'int8':
                from sentence_transformers import export_dynamic_quantized_onnx_model
                print(f"Step 2: Quantizing ({args.quantization}) to {output_dir}...")

                export_dynamic_quantized_onnx_model(
                    model=model,
                    quantization_config=args.onnx_config,
                    model_name_or_path=output_dir,
                    push_to_hub=False,
                    create_pr=False,
                )
            else:
                print(f"Step 2: Saving to {output_dir}...")
                model.save_pretrained(output_dir)
                print("✓ Model saved!")
                print()

            # Apply quantization if requested
            # if args.quantization == 'int8':
            #     from onnxruntime.quantization import quantize_dynamic, QuantType
            #     from pathlib import Path
            #
            #     print("Step 3: Applying INT8 dynamic quantization...")
            #
            #     # Find the ONNX model file
            #     onnx_dir = Path(output_dir) / "onnx"
            #     onnx_model_path = onnx_dir / "model.onnx"
            #     quantized_model_path = onnx_dir / "model_quantized.onnx"
            #
            #     if not onnx_model_path.exists():
            #         raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")
            #
            #     # Apply dynamic quantization
            #     quantize_dynamic(
            #         str(onnx_model_path),
            #         str(quantized_model_path),
            #         weight_type=QuantType.QInt8,
            #         per_channel=True,
            #         reduce_range=False,
            #     )
            #
            #     # Replace original with quantized model
            #     onnx_model_path.unlink()
            #     quantized_model_path.rename(onnx_model_path)
            #
            #     print("✓ INT8 quantization applied!")
            #
            #     # Get model sizes
            #     model_size = onnx_model_path.stat().st_size / (1024 * 1024)
            #     print(f"  Quantized model size: {model_size:.1f} MB")
            #     print()
            #
            #     # Reload the quantized model for testing
            #     model = SentenceTransformer(output_dir, backend="onnx")
            #     print("✓ Quantized model reloaded successfully!")
            #     print()
            # elif args.quantization == 'int4':
            #     print("\n⚠ Warning: INT4 quantization is not supported for ONNX. Only INT8 is available.")
            #     print("   Skipping quantization...")
            #     print()

        # Test the model with comprehensive validation
        step_num = 3 if args.format == 'openvino' else (4 if args.quantization == 'int8' else 3)
        print(f"Step {step_num}: Validating the exported model...")
        test_sentences = generate_test_sentences(50)

        # Load original model for comparison if requested
        original_model = None
        if args.compare_with_original:
            print("\nLoading original PyTorch model for comparison...")
            original_model = SentenceTransformer(model_name)
            print("✓ Original model loaded")

        # Run comprehensive testing
        test_model_similarity(model, original_model, test_sentences, output_dir)

        print(f"\n{'='*70}")
        print("✓ EXPORT COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print()
        print(f"Model saved to: {output_dir}")
        print(f"\nYou can now use the model with:")
        if args.format == 'openvino':
            print(f'  model = SentenceTransformer("{output_dir}", backend="openvino")')
        else:
            print(f'  model = SentenceTransformer("{output_dir}", backend="onnx")')

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        print()
        print("="*70)
        print("❌ EXPORT FAILED")
        print("="*70)

        return 1


if __name__ == "__main__":
    sys.exit(main())
