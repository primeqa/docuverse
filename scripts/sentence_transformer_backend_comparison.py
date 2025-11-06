import argparse
import json
import statistics
import gc
import sys

import psutil

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, LiteralString
import time
import torch
from tqdm import tqdm
import os

from docuverse.utils import open_stream


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def load_sentence_transformer_with_backend(model_name: str, backend: str = 'pytorch', 
                                         device: str = 'cuda', model_path: str = None,
                                         **backend_kwargs):
    """Load a SentenceTransformer model with specified backend.
    
    Args:
        model_name: SentenceTransformer model name or path
        backend: Backend to use ('pytorch', 'onnx', 'openvino')
        device: Device to run on ('cuda', 'cpu')
        model_path: Path to converted model (for onnx/openvino)
        **backend_kwargs: Additional backend-specific arguments
    """
    try:
        if backend == 'pytorch':
            # Standard SentenceTransformer loading
            model_kwargs = {}
            if device == 'cpu':
                model_kwargs['model_kwargs'] = {'attn_implementation': 'eager'}
            
            model = SentenceTransformer(model_name, device=device, **model_kwargs)
            
        elif backend == 'onnx':
            if model_path is None:
                raise ValueError("model_path is required for ONNX backend")
            onnx_kwargs = {
                    'provider': 'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'
                }
            if model_path.endswith('.onnx'):
                model_path = _get_model_and_file_names(model_path, onnx_kwargs)
            # Use ONNX backend with SentenceTransformers
            model = SentenceTransformer(
                model_name if os.path.exists(model_name) else model_path,
                device=device,
                backend='onnx',
                model_kwargs=onnx_kwargs,
                **backend_kwargs
            )
            
        elif backend == 'openvino':
            if model_path is None:
                raise ValueError("model_path is required for OpenVINO backend")

            # Use OpenVINO backend with SentenceTransformers
            openvino_kwargs = {
                'device': device.upper(),  # OpenVINO uses uppercase device names
                **backend_kwargs
            }

            # Check if model_path points to an XML file
            if model_path and model_path.endswith('.xml'):
                model_path = _get_model_and_file_names(model_path, openvino_kwargs)

            model = SentenceTransformer(
                model_name if os.path.exists(model_name) else model_path,
                backend='openvino',
                model_kwargs=openvino_kwargs
            )
            
        else:
            raise ValueError(f"Unsupported backend: {backend}")
            
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading SentenceTransformer with {backend} backend: {e}")


def _get_model_and_file_names(model_path: str, kwargs: dict[str, str]) -> LiteralString | str | bytes:
    # Split path into all components
    path_components = []
    current_path = model_path
    path_components = os.path.normpath(current_path).split(os.sep)
    if path_components[0] == '':
        path_components[0] = os.sep

    # Use filename and immediate parent dir for OpenVINO model path
    kwargs['file_name'] = os.path.join(path_components[-2], path_components[-1])
    # Join all path components except last two
    model_path = os.path.join(*path_components[:-2]) if len(path_components) > 2 else ''
    return model_path


def get_embeddings(model: SentenceTransformer, batch: List[str], convert_to_numpy: bool=True) -> np.ndarray:
    """Generate embeddings using SentenceTransformer (works with all backends)."""
    with torch.no_grad():
        embeddings = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=True
        )

    # Ensure embeddings are on CPU and in numpy format for comparison
    if isinstance(embeddings, torch.Tensor):
        if embeddings.is_cuda:
            embeddings = embeddings.cpu()
        embeddings = embeddings.numpy()
    elif not isinstance(embeddings, np.ndarray):
        # Convert other types to numpy, handling nested tensors
        if hasattr(embeddings, '__iter__') and len(embeddings) > 0 and isinstance(embeddings[0], torch.Tensor):
            embeddings = torch.stack([e.cpu() if e.is_cuda else e for e in embeddings]).numpy()
        else:
            embeddings = np.array(embeddings)

    return embeddings


def read_sentences(file_path: str, max_text_length: int = None) -> List[str]:
    """Read sentences from a file, one per line.

    Args:
        file_path: Path to input file
        max_text_length: Maximum allowed text length (optional)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    sentences = []
    with open_stream(file_path) as f:
        for line in f:  # Read line by line instead of readlines()
            line = line.strip()
            if not line:
                continue
            
            # Try to parse as JSONL first
            try:
                text = json.loads(line)['text']
            except (json.JSONDecodeError, KeyError):
                # Fall back to plain text
                text = line
            
            if max_text_length and len(text) > max_text_length:
                text = text[:max_text_length]
            
            sentences.append(text)

    return sentences


def compute_similarity(embeddings1: np.ndarray|list, embeddings2: np.ndarray|list, return_mean: bool=True) -> Tuple[float, float, float]:
    """Compute similarity metrics between two sets of embeddings."""
    # Normalize embeddings
    if isinstance(embeddings1, list):
        embeddings1 = np.array(embeddings1)
    if isinstance(embeddings2, list):
        embeddings2 = np.array(embeddings2)

    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

    embeddings1_normalized = embeddings1 / np.maximum(norm1, 1e-9)
    embeddings2_normalized = embeddings2 / np.maximum(norm2, 1e-9)

    # Compute cosine similarities between corresponding embeddings
    cosine_similarities = np.sum(embeddings1_normalized * embeddings2_normalized, axis=1)

    if return_mean:
        function = np.mean
    else:
        function = np.sum

        # Compute MSE
    mse = function(np.square(embeddings1 - embeddings2))

    # Compute Manhattan distance
    manhattan = function(np.abs(embeddings1 - embeddings2))

    return np.array([function(cosine_similarities), mse, manhattan])


def main():
    # Get the script name from sys.argv[0]
    script_name = os.path.basename(sys.argv[0])
    
    parser = argparse.ArgumentParser(
        description="Compare different SentenceTransformer backends (PyTorch, ONNX, OpenVINO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Compare PyTorch with ONNX
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch onnx --onnx-path ./model.onnx --input data.txt

  # Compare PyTorch with OpenVINO
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch openvino --openvino-path ./model.xml --input data.txt

  # Compare all three backends
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch onnx openvino --onnx-path ./model.onnx --openvino-path ./model.xml --input data.txt
        """
    )
    # ... rest of the argument definitions ...
    parser.add_argument("--model", required=True, help="SentenceTransformer model name or path")
    parser.add_argument("--backends", nargs='+', choices=["pytorch", "onnx", "openvino"], 
                       default=["pytorch"], help="Backends to compare")
    parser.add_argument("--input", required=True, help="Path to the input text file with sentences")
    
    # Accept both - and _ variants for multi-word arguments
    parser.add_argument("--onnx-path", "--onnx_path", dest="onnx_path", 
                       help="Path to ONNX model directory or file")
    parser.add_argument("--openvino-path", "--openvino_path", dest="openvino_path",
                       help="Path to OpenVINO model directory or .xml file")
    parser.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=None, 
                       help="Number of samples to use (default: all)")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128, 
                       help="Batch size for processing (default: 128)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to run models on (default: cuda)")
    parser.add_argument("--max-text-length", "--max_text_length", dest="max_text_length", 
                       type=int, default=None, help="Maximum text length (default: no limit)")
    parser.add_argument("--num-threads", "--num_threads", dest="num_threads", type=int, default=None,
                       help="Number of CPU threads for PyTorch (default: all available)")
    parser.add_argument("--openvino-device", "--openvino_device", dest="openvino_device", 
                       type=str, default="CPU", help="OpenVINO device (CPU, GPU, etc.) (default: CPU)")

    args = parser.parse_args()

    # Set PyTorch thread count if specified
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        print(f"Set PyTorch CPU threads to: {args.num_threads}")

    # Validate backend-specific paths
    onnx_model = args.onnx_path
    openvino_model = args.openvino_path
    if 'onnx' in args.backends and not args.onnx_path:
        print("No path provided for onnx model, will use the model name")
        onnx_model = args.model
        # parser.error("--onnx-path (or --onnx_path) is required when using ONNX backend")
    if 'openvino' in args.backends and not args.openvino_path:
        print("No path provided for openvino model, will use the model name")
        openvino_model = args.model
        # parser.error("--openvino-path (or --openvino_path) is required when using OpenVINO backend")

    # Load models for each backend
    models = {}
    for backend in args.backends:
        print(f"Loading {backend.upper()} model...")
        
        if backend == 'pytorch':
            models[backend] = load_sentence_transformer_with_backend(
                args.model, backend='pytorch', device=args.device
            )
        elif backend == 'onnx':
            models[backend] = load_sentence_transformer_with_backend(
                args.model, backend='onnx', device=args.device,
                model_path=onnx_model
            )
        elif backend == 'openvino':
            models[backend] = load_sentence_transformer_with_backend(
                args.model, backend='openvino', device=args.openvino_device,
                model_path=openvino_model
            )

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

    # Warmup phase - run a few batches to initialize models
    print("\nWarming up models...")
    warmup_batches = min(3, len(batches))  # Use up to 3 batches for warmup
    for backend in args.backends:
        print(f"  Warming up {backend.upper()}...")
        for i in range(warmup_batches):
            _ = get_embeddings(models[backend], batches[i], convert_to_numpy=False)
    print("✓ Warmup completed")

    # Initialize tracking variables
    backend_times = {backend: 0 for backend in args.backends}
    backend_embeddings = {backend: [] for backend in args.backends}

    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.1f} MB")

    print("\nProcessing batches...")
    scores = {}
    for i, batch in enumerate(tqdm(batches)):
        batch_embeddings = [[] for _ in args.backends]
        
        # Generate embeddings for each backend
        for i, backend in enumerate(args.backends):
            start_time = time.time()
            embeddings = get_embeddings(models[backend], batch, convert_to_numpy=False)
            backend_times[backend] += time.time() - start_time
            batch_embeddings[i] = embeddings

        for i in range(len(args.backends)):
            for j in range(i+1, len(args.backends)):
                key = f"{args.backends[j]} {args.backends[i]}"
                if key not in scores:
                    scores[key] = np.zeros(3)
                scores[key] += compute_similarity(batch_embeddings[i], batch_embeddings[j], return_mean=False)


        # Clean up batch embeddings
        del batch_embeddings
        
        # Memory management
        if i % 50 == 0 and i > 0:
            current_memory = get_memory_usage()
            if current_memory > initial_memory * 1.5:
                gc.collect()
                print(f"GC triggered at batch {i}, memory: {current_memory:.1f}MB")

    # Normalize the results:
    for k in scores.keys():
        scores[k] /= len(sentences)

    # Report performance results
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    num_docs = len(sentences)
    baseline_backend = args.backends[0]  # Use first backend as baseline
    baseline_time = backend_times[baseline_backend]

    # Print header
    print(f"\n{'Backend':<15} {'Time (s)':<12} {'Throughput':<18} {'Speedup':<10}")
    print("-" * 70)

    for backend in args.backends:
        time_val = backend_times[backend]
        throughput = num_docs / time_val if time_val > 0 else float('inf')

        if backend == baseline_backend:
            print(f"{backend.upper():<15} {time_val:>8.2f}s    {throughput:>8.1f} docs/s    {'(baseline)':<10}")
        else:
            speedup = baseline_time / time_val if time_val > 0 else float('inf')
            print(f"{backend.upper():<15} {time_val:>8.2f}s    {throughput:>8.1f} docs/s    {speedup:>6.2f}x")

    # Report embedding similarity comparisons
    if len(args.backends) > 1:
        print("\n" + "=" * 70)
        print("EMBEDDING SIMILARITY COMPARISON")
        print("=" * 70)
        
        # Compare each backend with the baseline (first backend)
        baseline_embeddings = backend_embeddings[baseline_backend]
        
        for backend in args.backends[1:]:
            cosine_sim, mse, manhattan = scores[f"{backend} {baseline_backend}"]
            print(f"\n{baseline_backend.upper()} vs {backend.upper()}:")
            print(f"  Mean Cosine Similarity:  {cosine_sim:.6f} (1.0 = identical)")
            print(f"  Mean Squared Error:      {mse:.6f} (0.0 = identical)")
            print(f"  Mean Manhattan Distance: {manhattan:.6f} (0.0 = identical)")
            _analyze_results(cosine_sim, f"{baseline_backend.upper()}/{backend.upper()}")

        # Compare all pairs if more than 2 backends
        if len(args.backends) > 2:
            print(f"\nAdditional pairwise comparisons:")
            for i, backend1 in enumerate(args.backends[1:], 1):
                for backend2 in args.backends[i+1:]:
                    # emb1 = backend_embeddings[backend1]
                    # emb2 = backend_embeddings[backend2]
                    #
                    # min_samples = min(emb1.shape[0], emb2.shape[0])
                    # min_dims = min(emb1.shape[1], emb2.shape[1])
                    #
                    # emb1_trimmed = emb1[:min_samples, :min_dims]
                    # emb2_trimmed = emb2[:min_samples, :min_dims]
                    #
                    # cosine_sim, mse, manhattan = compute_similarity(emb1_trimmed, emb2_trimmed)
                    cosine_sim, mse, manhattan = scores[f"{backend2} {backend1}"]
                    
                    print(f"\n{backend1.upper()} vs {backend2.upper()}:")
                    print(f"  Cosine Similarity: {cosine_sim:.6f}")


def _analyze_results(cosine_sim: float, model_name: str):
    """Analyze and print interpretation of similarity results."""
    if cosine_sim > 0.99:
        print(f"  → {model_name} produces very similar embeddings! ✓")
    elif cosine_sim > 0.95:
        print(f"  → {model_name} produces similar embeddings with minor differences.")
    else:
        print(f"  → {model_name} has significant differences.")
        print("  Potential issues:")
        print("    - Backend conversion might not be exact")
        print("    - Different preprocessing or tokenization")
        print("    - Check model compatibility")


if __name__ == "__main__":
    main()
