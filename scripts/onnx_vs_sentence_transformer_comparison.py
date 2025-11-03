import argparse
import json
import statistics

import numpy as np
from openvino.runtime import Core
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import time
import torch
from tqdm import tqdm
import os

from docuverse.utils import open_stream
from optimum.onnxruntime import ORTModelForFeatureExtraction
import onnxruntime as ort
from transformers import AutoTokenizer


def load_onnx_model(onnx_model_path: str, device: str = 'cuda', num_threads: int = None):
    """Load an ONNX model.

    Args:
        onnx_model_path: Path to ONNX model directory
        device: Device to run on ('cuda' or 'cpu')
        num_threads: Number of CPU threads (None = all available)

    Returns:
        Tuple of (session, tokenizer)
    """
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model file not found: {onnx_model_path}")

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)

        # Load ONNX model
        print("Loading ONNX model...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']

        # Set up session options
        sess_options = ort.SessionOptions()
        if num_threads is not None:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads

        session = ort.InferenceSession(
            os.path.join(onnx_model_path, "model.onnx"),
            sess_options=sess_options,
            providers=providers
        )

        # ort_model = ORTModelForFeatureExtraction.from_pretrained(onnx_model_path)
        # ort_model.to(device)
        # return ort_model
        return session, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading ONNX model: {e}")


def load_sentence_transformer(model_name: str, device='cuda'):
    """Load a SentenceTransformer model."""
    try:
        # Force eager attention on CPU to avoid Flash Attention issues
        model_kwargs = {}
        if device == 'cpu':
            model_kwargs['model_kwargs'] = {'attn_implementation': 'eager'}

        model = SentenceTransformer(model_name, device=device, **model_kwargs)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading SentenceTransformer model: {e}")


def read_sentences(file_path: str, max_text_length: int = None) -> List[str]:
    """Read sentences from a file, one per line.

    Args:
        file_path: Path to input file
        max_text_length: Maximum allowed text length (optional)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open_stream(file_path) as f:
        lines = f.readlines()

    # Try to parse as JSONL first
    try:
        sentences = [json.loads(s)['text'] for s in lines if s.strip()]
    except (json.JSONDecodeError, KeyError):
        # Fall back to plain text (one line per sentence)
        sentences = [line.strip() for line in lines if line.strip()]

    if max_text_length:
        sentences = [s[:max_text_length] if len(s)>max_text_length else s for s in sentences]

    return sentences


def get_onnx_embeddings(onnx_model, tokenizer,
                        batch: List[str]) -> np.ndarray:
    """Generate embeddings using the ONNX model for a single batch."""
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="np")

    # Prepare inputs for ONNX Runtime
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    # Run inference
    outputs = onnx_model.run(None, onnx_inputs)

    # Extract and normalize CLS embeddings (first token)
    embeddings = []
    for cls in outputs[0][:, 0]:
        cls = cls / np.linalg.norm(cls)
        embeddings.append(cls)

    return np.array(embeddings)


def get_sentence_transformer_embeddings(model, batch: List[str]) -> np.ndarray:
    """Generate embeddings using the SentenceTransformer model for a single batch."""
    with torch.no_grad():
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True,
                                  normalize_embeddings=True)
    return embeddings


def load_openvino_model(model_path: str, tokenizer_path: str = None, precision: str = None, device: str = 'CPU', num_threads: int = None):
    """Load an OpenVINO model and tokenizer.

    Args:
        model_path: Path to the OpenVINO model (.xml file)
        tokenizer_path: Path to tokenizer directory (default: same as model directory)
        precision: Precision hint ('int8', 'int4', or None for default)
        device: Device to run on (CPU, GPU, etc.)
        num_threads: Number of CPU threads (None = all available)

    Returns:
        Tuple of (compiled_model, tokenizer)
    """
    try:
        core = Core()

        # Load model
        model = core.read_model(model_path)

        # Build config
        config = {}
        if precision:
            config["INFERENCE_PRECISION_HINT"] = precision.lower()
        if num_threads is not None:
            # OpenVINO uses INFERENCE_NUM_THREADS for thread control
            config["INFERENCE_NUM_THREADS"] = str(num_threads)

        # Compile model
        if config:
            compiled_model = core.compile_model(model, device, config)
        else:
            compiled_model = core.compile_model(model, device)

        # Load tokenizer from the same directory as model or specified path
        if tokenizer_path is None:
            tokenizer_path = os.path.dirname(model_path)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        return compiled_model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading OpenVINO model: {e}")


def get_openvino_embeddings(model, tokenizer, batch: List[str], normalize: bool = True, use_cls_pooling: bool = True) -> np.ndarray:
    """Generate embeddings using the OpenVINO model for a single batch.

    Args:
        model: Compiled OpenVINO model
        tokenizer: Tokenizer for the model
        batch: List of input texts
        normalize: Whether to normalize embeddings
        use_cls_pooling: Whether to use CLS token pooling (True) or mean pooling (False)

    Returns:
        Array of embeddings
    """
    # Get model input shape
    input_shape = None
    for input_info in model.inputs:
        if 'input_ids' in input_info.any_name or input_info.any_name == 'input_ids':
            input_shape = input_info.shape
            break

    # Check if model has fixed batch size of 1
    fixed_batch_size_1 = input_shape is not None and input_shape[0] == 1

    if fixed_batch_size_1:
        # Process one at a time if model has batch size 1
        all_embeddings = []
        for text in batch:
            inputs = tokenizer(
                [text],
                padding="max_length",
                truncation=True,
                return_tensors="np",
                max_length=512
            )

            # Prepare inputs for OpenVINO
            input_data = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }

            # Run inference
            result = model(input_data)

            # Get output
            output_key = list(result.keys())[0]
            embeddings = result[output_key]

            # Apply pooling
            if use_cls_pooling:
                # CLS token pooling - use first token
                pooled = embeddings[:, 0, :]
            else:
                # Mean pooling
                attention_mask = inputs["attention_mask"]
                masked_embeddings = embeddings * attention_mask[:, :, np.newaxis]
                sum_embeddings = masked_embeddings.sum(axis=1)
                sum_mask = attention_mask.sum(axis=1, keepdims=True)
                pooled = sum_embeddings / np.maximum(sum_mask, 1e-9)

            all_embeddings.append(pooled[0])

        pooled_embeddings = np.array(all_embeddings)
    else:
        # Process batch as normal
        inputs = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            return_tensors="np",
            max_length=512
        )

        # Prepare inputs for OpenVINO
        input_data = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        # Run inference
        result = model(input_data)

        # Get output
        output_key = list(result.keys())[0]
        embeddings = result[output_key]

        # Apply pooling
        if use_cls_pooling:
            # CLS token pooling - use first token
            pooled_embeddings = embeddings[:, 0, :]
        else:
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            masked_embeddings = embeddings * attention_mask[:, :, np.newaxis]
            sum_embeddings = masked_embeddings.sum(axis=1)
            sum_mask = attention_mask.sum(axis=1, keepdims=True)
            pooled_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)

    # Normalize if requested
    if normalize:
        norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
        pooled_embeddings = pooled_embeddings / np.maximum(norms, 1e-9)

    return pooled_embeddings


def compute_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> Tuple[float, float, float]:
    """Compute similarity metrics between two sets of embeddings."""
    # Normalize embeddings
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    embeddings1_normalized = embeddings1 / norm1
    embeddings2_normalized = embeddings2 / norm2
    
    # Compute cosine similarities between corresponding embeddings
    cosine_similarities = np.sum(embeddings1_normalized * embeddings2_normalized, axis=1)
    
    # Compute MSE
    mse = np.mean(np.square(embeddings1 - embeddings2))
    
    # Compute Manhattan distance
    manhattan = np.mean(np.abs(embeddings1 - embeddings2))
    
    return np.mean(cosine_similarities), mse, manhattan


def main():
    parser = argparse.ArgumentParser(
        description="Compare ONNX and/or OpenVINO model with SentenceTransformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare ONNX with HuggingFace
  python script.py --onnx model.onnx --transformer model-name --input data.txt

  # Compare OpenVINO with HuggingFace
  python script.py --openvino model.xml --transformer model-name --input data.txt

  # Compare both ONNX and OpenVINO with HuggingFace
  python script.py --onnx model.onnx --openvino model.xml --transformer model-name --input data.txt
        """
    )
    parser.add_argument("--onnx", help="Path to the ONNX model directory")
    parser.add_argument("--transformer", required=True, help="SentenceTransformer model name")
    parser.add_argument("--input", required=True, help="Path to the input text file with sentences")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use (default: all)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing (default: 128)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run models on (default: cuda)")
    parser.add_argument("--max_text_length", type=int, default=None, help="Maximum text length (default: no limit)")
    parser.add_argument("--openvino", type=str, help="Path to OpenVINO IR model (.xml file)")
    parser.add_argument("--openvino_tokenizer", type=str, help="Path to OpenVINO tokenizer (default: model directory)")
    parser.add_argument("--openvino_precision", type=str, choices=["int4", "int8", "fp16", "fp32"],
                        help="OpenVINO precision hint (int4, int8, fp16, fp32)")
    parser.add_argument("--int8", action="store_true", help="Use INT8 precision hint for OpenVINO (deprecated, use --openvino_precision)")
    parser.add_argument("--openvino_device", type=str, default="CPU",
                        help="OpenVINO device (CPU, GPU, etc.) (default: CPU)")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"],
                        help="Pooling method for OpenVINO embeddings (default: cls)")
    parser.add_argument("--num_threads", type=int, default=None,
                        help="Number of CPU threads for PyTorch (default: all available)")
    parser.add_argument("--openvino_threads", type=int, default=None,
                        help="Number of CPU threads for OpenVINO (default: all available)")

    args = parser.parse_args()

    # Set PyTorch thread count if specified
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        print(f"Set PyTorch CPU threads to: {args.num_threads}")
    else:
        print(f"Using default PyTorch CPU threads: {torch.get_num_threads()}")

    # Validate that at least one model format is provided
    if not args.onnx and not args.openvino:
        parser.error("At least one of --onnx or --openvino must be specified")

    # Load models
    onnx_model = None
    onnx_tokenizer = None
    openvino_model = None
    openvino_tokenizer = None

    if args.onnx:
        print(f"Loading ONNX model from {args.onnx}")
        # Only set threads for CPU device
        onnx_threads = args.num_threads if args.device == 'cpu' else None
        if onnx_threads:
            print(f"Setting ONNX Runtime CPU threads to: {onnx_threads}")
        onnx_model, onnx_tokenizer = load_onnx_model(
            args.onnx,
            device=args.device,
            num_threads=onnx_threads
        )

    if args.openvino:
        print(f"Loading OpenVINO model from {args.openvino}")

        # Determine precision hint (backward compatibility with --int8)
        precision = args.openvino_precision
        if args.int8 and not precision:
            precision = "int8"
            print("Note: --int8 is deprecated, use --openvino_precision int8")

        if precision:
            print(f"Using OpenVINO precision hint: {precision}")
        if args.openvino_threads:
            print(f"Setting OpenVINO CPU threads to: {args.openvino_threads}")

        openvino_model, openvino_tokenizer = load_openvino_model(
            args.openvino,
            tokenizer_path=args.openvino_tokenizer,
            precision=precision,
            device=args.openvino_device,
            num_threads=args.openvino_threads
        )

    print(f"Loading SentenceTransformer model: {args.transformer}")
    sentence_transformer = load_sentence_transformer(args.transformer, device=args.device)

    # Read sentences
    sentences = read_sentences(args.input, args.max_text_length)
    if args.num_samples and args.num_samples < len(sentences):
        print(f"Using {args.num_samples} out of {len(sentences)} sentences")
        sentences = sentences[:args.num_samples]
    else:
        print(f"Using all {len(sentences)} sentences")

    # Calculate and display sentence length statistics
    lengths = [len(s) for s in sentences]
    print("\nSentence length statistics:")
    print(f"Mean length: {statistics.mean(lengths):.1f} chars")
    print(f"Median length: {statistics.median(lengths):.1f} chars")
    print(f"Min length: {min(lengths)} chars")
    print(f"Max length: {max(lengths)} chars")
    print(f"Std dev: {statistics.stdev(lengths):.1f} chars")

    # Process in batches
    batches = [sentences[i:i + args.batch_size] for i in range(0, len(sentences), args.batch_size)]

    # Initialize tracking variables
    onnx_cosine_sim = 0
    onnx_mse = 0
    onnx_manhattan = 0
    onnx_time = 0

    openvino_cosine_sim = 0
    openvino_mse = 0
    openvino_manhattan = 0
    openvino_time = 0

    onnx_openvino_cosine_sim = 0
    onnx_openvino_mse = 0
    onnx_openvino_manhattan = 0

    transformer_time = 0

    print("\nProcessing batches...")
    for i, batch in enumerate(tqdm(batches)):
        # Generate HuggingFace embeddings (baseline)
        start_time = time.time()
        transformer_batch_embeddings = get_sentence_transformer_embeddings(sentence_transformer, batch)
        transformer_time += time.time() - start_time

        onnx_batch_embeddings = None
        openvino_batch_embeddings = None

        # Generate ONNX embeddings if available
        if onnx_model is not None:
            start_time = time.time()
            onnx_batch_embeddings = get_onnx_embeddings(onnx_model, onnx_tokenizer, batch)
            onnx_time += time.time() - start_time

            # Check and adjust embedding shapes if needed
            if onnx_batch_embeddings.shape != transformer_batch_embeddings.shape:
                min_dim = min(onnx_batch_embeddings.shape[1], transformer_batch_embeddings.shape[1])
                onnx_batch_embeddings = onnx_batch_embeddings[:, :min_dim]
                transformer_batch_trimmed = transformer_batch_embeddings[:, :min_dim]
            else:
                transformer_batch_trimmed = transformer_batch_embeddings

            # Compute similarities with HuggingFace
            batch_cosine, batch_mse, batch_manhattan = compute_similarity(
                onnx_batch_embeddings, transformer_batch_trimmed)

            # Update running averages
            weight = len(batch) / ((i + 1) * args.batch_size)
            onnx_cosine_sim = (onnx_cosine_sim * (1 - weight)) + (batch_cosine * weight)
            onnx_mse = (onnx_mse * (1 - weight)) + (batch_mse * weight)
            onnx_manhattan = (onnx_manhattan * (1 - weight)) + (batch_manhattan * weight)

        # Generate OpenVINO embeddings if available
        if openvino_model is not None:
            start_time = time.time()
            use_cls = args.pooling == "cls"
            openvino_batch_embeddings = get_openvino_embeddings(
                openvino_model, openvino_tokenizer, batch, use_cls_pooling=use_cls
            )
            openvino_time += time.time() - start_time

            # Check and adjust embedding shapes if needed
            if openvino_batch_embeddings.shape != transformer_batch_embeddings.shape:
                min_dim = min(openvino_batch_embeddings.shape[1], transformer_batch_embeddings.shape[1])
                openvino_batch_embeddings = openvino_batch_embeddings[:, :min_dim]
                transformer_batch_trimmed = transformer_batch_embeddings[:, :min_dim]
            else:
                transformer_batch_trimmed = transformer_batch_embeddings

            # Compute similarities with HuggingFace
            batch_cosine, batch_mse, batch_manhattan = compute_similarity(
                openvino_batch_embeddings, transformer_batch_trimmed)

            # Update running averages
            weight = len(batch) / ((i + 1) * args.batch_size)
            openvino_cosine_sim = (openvino_cosine_sim * (1 - weight)) + (batch_cosine * weight)
            openvino_mse = (openvino_mse * (1 - weight)) + (batch_mse * weight)
            openvino_manhattan = (openvino_manhattan * (1 - weight)) + (batch_manhattan * weight)

        # Compare ONNX and OpenVINO if both are available
        if onnx_batch_embeddings is not None and openvino_batch_embeddings is not None:
            # Ensure same shape
            min_dim = min(onnx_batch_embeddings.shape[1], openvino_batch_embeddings.shape[1])
            onnx_trimmed = onnx_batch_embeddings[:, :min_dim]
            openvino_trimmed = openvino_batch_embeddings[:, :min_dim]

            batch_cosine, batch_mse, batch_manhattan = compute_similarity(
                onnx_trimmed, openvino_trimmed)

            weight = len(batch) / ((i + 1) * args.batch_size)
            onnx_openvino_cosine_sim = (onnx_openvino_cosine_sim * (1 - weight)) + (batch_cosine * weight)
            onnx_openvino_mse = (onnx_openvino_mse * (1 - weight)) + (batch_mse * weight)
            onnx_openvino_manhattan = (onnx_openvino_manhattan * (1 - weight)) + (batch_manhattan * weight)

        # Clean up
        del transformer_batch_embeddings
        if onnx_batch_embeddings is not None:
            del onnx_batch_embeddings
        if openvino_batch_embeddings is not None:
            del openvino_batch_embeddings

    # Report results
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"HuggingFace SentenceTransformer: {transformer_time:.2f}s")
    if onnx_model is not None:
        print(f"ONNX Runtime:                    {onnx_time:.2f}s (speedup: {transformer_time / onnx_time:.2f}x)")
    if openvino_model is not None:
        print(f"OpenVINO:                        {openvino_time:.2f}s (speedup: {transformer_time / openvino_time:.2f}x)")

    if onnx_model is not None and openvino_model is not None:
        print(f"\nOpenVINO vs ONNX:                {openvino_time / onnx_time:.2f}x {'faster' if openvino_time < onnx_time else 'slower'}")

    # Report embedding comparisons
    print("\n" + "=" * 70)
    print("EMBEDDING SIMILARITY COMPARISON")
    print("=" * 70)

    if onnx_model is not None:
        print("\nONNX vs HuggingFace:")
        print(f"  Mean Cosine Similarity:  {onnx_cosine_sim:.6f} (1.0 = identical)")
        print(f"  Mean Squared Error:      {onnx_mse:.6f} (0.0 = identical)")
        print(f"  Mean Manhattan Distance: {onnx_manhattan:.6f} (0.0 = identical)")
        _analyze_results(onnx_cosine_sim, "ONNX")

    if openvino_model is not None:
        print("\nOpenVINO vs HuggingFace:")
        print(f"  Mean Cosine Similarity:  {openvino_cosine_sim:.6f} (1.0 = identical)")
        print(f"  Mean Squared Error:      {openvino_mse:.6f} (0.0 = identical)")
        print(f"  Mean Manhattan Distance: {openvino_manhattan:.6f} (0.0 = identical)")
        _analyze_results(openvino_cosine_sim, "OpenVINO")

    if onnx_model is not None and openvino_model is not None:
        print("\nONNX vs OpenVINO:")
        print(f"  Mean Cosine Similarity:  {onnx_openvino_cosine_sim:.6f} (1.0 = identical)")
        print(f"  Mean Squared Error:      {onnx_openvino_mse:.6f} (0.0 = identical)")
        print(f"  Mean Manhattan Distance: {onnx_openvino_manhattan:.6f} (0.0 = identical)")
        _analyze_results(onnx_openvino_cosine_sim, "ONNX/OpenVINO")


def _analyze_results(cosine_sim: float, model_name: str):
    """Analyze and print interpretation of similarity results."""
    if cosine_sim > 0.99:
        print(f"  → {model_name} produces very similar embeddings! ✓")
    elif cosine_sim > 0.95:
        print(f"  → {model_name} produces similar embeddings with minor differences.")
    else:
        print(f"  → {model_name} has significant differences.")
        print("  Potential issues:")
        print("    - Conversion might not be exact")
        print("    - Models might use different preprocessing")
        print("    - Check tokenization consistency")


if __name__ == "__main__":
    main()
