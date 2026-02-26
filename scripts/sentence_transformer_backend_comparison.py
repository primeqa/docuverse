import argparse
import json
import statistics
import gc
import sys

import psutil

from docuverse.utils.embeddings.ollama_embedding_function import OllamaSentenceTransformer

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, LiteralString, Any
import time
import torch
from tqdm import tqdm
import os

from docuverse.utils import open_stream
from docuverse.utils.jsonl_utils import read_jsonl_file

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024



def load_sentence_transformer_with_backend(model_name: str,
                                           backend: str = 'pytorch',
                                           device: str = 'cuda',
                                           model_path: str = None,
                                           precision: str = 'bf16',
                                           **backend_kwargs):
    """Load a SentenceTransformer model with a specified backend.

    Args:
        model_name: SentenceTransformer model name or path
        backend: Backend to use ('pytorch', 'onnx', 'openvino')
        device: Device to run on ('cuda', 'cpu')
        model_path: Path to a converted model (for onnx/openvino)
        precision: Model precision for PyTorch ('fp32', 'fp16', 'bf16')
        **backend_kwargs: Additional backend-specific arguments
    """
    model = None
    try:
        if backend == 'pytorch':
            # Standard SentenceTransformer loading with specified precision
            model_kwargs = {'model_kwargs': {}}

            # Set precision
            if precision == 'bf16':
                model_kwargs['model_kwargs']['dtype'] = torch.bfloat16
            elif precision == 'fp16':
                model_kwargs['model_kwargs']['dtype'] = torch.float16
            elif precision == 'fp32':
                model_kwargs['model_kwargs']['dtype'] = torch.float32

            if device == 'cpu':
                model_kwargs['model_kwargs']['attn_implementation'] = 'eager'

            model = SentenceTransformer(model_name, device=device, **model_kwargs)
            print(f"  Loaded model in {precision.upper()} precision")

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
        elif backend == "ollama":
            print(f"Loading model with Ollama backend...")
            try:

                # Get Ollama base URL from backend_kwargs if provided
                base_url = backend_kwargs.get('base_url', 'http://localhost:11434')

                model = OllamaSentenceTransformer(model_name, base_url=base_url)
                print(f"✓ Model loaded successfully with Ollama backend")
                print(f"  Model: {model_name}")
                print(f"  Ollama URL: {base_url}")
                print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

            except ConnectionError as e:
                print(f"❌ Ollama connection error: {e}")
                return None
            except Exception as e:
                print(f"❌ Error loading model with Ollama backend: {e}")
                return None

        elif backend == "vllm":
            print(f"Loading model with vLLM backend...")
            import requests
            try:
                # from vllm import LLM, SamplingParams

                # For vLLM, we need to handle embedding models differently
                # vLLM primarily supports generative models, but can be used for embeddings
                # with some models that support pooling

                # Initialize vLLM model
                # vllm_model = LLM(
                #     model=model_name,
                #     trust_remote_code=True,
                #     enforce_eager=True,  # Disable CUDA graphs for compatibility
                #     **backend_kwargs
                # )

                # Create a wrapper class to make it compatible with SentenceTransformer interface
                class VLLMSentenceTransformer:
                    _dim = None
                    def __init__(self, vllm_model, original_model_name: str=None):
                        self.vllm_model = vllm_model
                        self.model_name = original_model_name if original_model_name else vllm_model

                    def encode(self, sentences, batch:int=32, **kwargs):
                        """Encode sentences using vLLM model."""
                        if isinstance(sentences, str):
                            sentences = [sentences]

                        embeddings = []

                        for ex in range(0, len(sentences), batch):
                            response = requests.post(
                                "http://localhost:8000/v1/embeddings",
                                json={
                                    "model": self.vllm_model,
                                    "input": sentences[ex:ex+batch],
                                    }
                            )

                            embeddings.extend([e['embedding'] for e in response.json()["data"]])

                        return np.array(embeddings)

                    def get_sentence_embedding_dimension(self):
                        if self._dim is None:
                            enc = self.encode(["Test"])
                            _dim = enc[0].size()
                        return self._dim
                        # return self._tokenizer_model.get_sentence_embedding_dimension()

                model = VLLMSentenceTransformer(model_name)
                print(f"✓ Model loaded successfully with vLLM backend")
            except ImportError as e:
                print(f"❌ vLLM not available: {e}")
                print("Install with: pip install vllm")
                return None
            except Exception as e:
                print(f"❌ Error loading model with vLLM backend: {e}")
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    except Exception as e:
        raise RuntimeError(f"Error loading SentenceTransformer with {backend} backend: {e}")

    return model


def compute_stats(vector: np.ndarray | list) -> Tuple[float, float, float, float, int]:
    """Compute statistics from a vector.

    Args:
        vector: Input vector (numpy array or list)

    Returns:
        Tuple containing (mean, median, std, min, arg_min)
    """
    if isinstance(vector, list):
        vector = np.array(vector)

    if len(vector) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    mean = float(np.mean(vector))
    median = float(np.median(vector))
    std = float(np.std(vector))
    min_val = float(np.min(vector))
    arg_min = int(np.argmin(vector))

    return mean, median, std, min_val, arg_min


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


def get_embeddings(model: SentenceTransformer, batch: List[str], convert_to_numpy: bool=True, batch_no=-1) -> np.ndarray:
    """Generate embeddings using SentenceTransformer (works with all backends)."""
    embeddings = None
    with torch.no_grad():
        try:
            embeddings = model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=True
            )
        except Exception as e:
            print(f"Embedding computation failed on batch {batch_no}: {e}. The text: {[t[:100]+" ..." for t in batch]}")
            raise e

    # Ensure embeddings are on CPU and in numpy format for comparison
    if isinstance(embeddings, torch.Tensor):
        if embeddings.is_cuda:
            embeddings = embeddings.cpu()
        # Convert BFloat16 to Float32 before numpy conversion (numpy doesn't support bfloat16)
        if embeddings.dtype == torch.bfloat16:
            embeddings = embeddings.float()
        embeddings = embeddings.numpy()
    elif not isinstance(embeddings, np.ndarray):
        # Convert other types to numpy, handling nested tensors
        if hasattr(embeddings, '__iter__') and len(embeddings) > 0 and isinstance(embeddings[0], torch.Tensor):
            stacked = torch.stack([e.cpu() if e.is_cuda else e for e in embeddings])
            # Convert BFloat16 to Float32 before numpy conversion
            if stacked.dtype == torch.bfloat16:
                stacked = stacked.float()
            embeddings = stacked.numpy()
        else:
            embeddings = np.array(embeddings)

    return embeddings


def read_sentences(file_path_or_string: str, max_text_length: int = None, field_path: str = None) -> List[str]:
    """Read sentences from a file or direct string input.

    Supports:
    - Direct string input (if not a file path)
    - Plain text files (one sentence per line)
    - JSONL files with field path extraction
    - JSONL.bz2 compressed files

    Args:
        file_path_or_string: Path to input file (.txt, .jsonl, .jsonl.bz2) or direct string
        max_text_length: Maximum allowed text length (optional)
        field_path: Dot-separated path to text field in JSONL (e.g., 'document.text', 'documents[*].text')
                   If None and file is JSONL, will try common field names: 'text', 'content', 'question'
    """
    # Check if input is a file path or direct string
    if os.path.exists(file_path_or_string):
        # It's a file path
        file_path = file_path_or_string

        # Check if file is JSONL format
        is_jsonl = file_path.endswith('.jsonl') or file_path.endswith('.jsonl.bz2')

        if is_jsonl:
            # Use jsonl_utils for JSONL files
            print(f"Reading JSONL file: {file_path}")
            if field_path:
                print(f"  Using field path: {field_path}")
            sentences = read_jsonl_file(file_path, field_path=field_path, verbose=True)
            print(f"✓ Loaded {len(sentences)} texts from JSONL")
        else:
            # Original logic for plain text files
            sentences = []
            with open_stream(file_path) as f:
                for line in f:  # Read line by line instead of readlines()
                    line = line.strip()
                    if not line:
                        continue

                    # Try to parse as JSONL first (for backward compatibility)
                    try:
                        text = json.loads(line)['text']
                    except (json.JSONDecodeError, KeyError):
                        # Fall back to plain text
                        text = line

                    sentences.append(text)
    else:
        # It's a direct string input
        print(f"Using direct string input")
        sentences = [file_path_or_string]

    # Apply max_text_length truncation if specified
    if max_text_length:
        sentences = [text[:max_text_length] if len(text) > max_text_length else text
                    for text in sentences]

    return sentences


def compute_similarity(embeddings1: np.ndarray | list, embeddings2: np.ndarray | list, return_mean: bool = True) -> \
List[float]:
    """Compute similarity metrics between two sets of embeddings.

    Returns:
        List containing [cosine_similarity, mse, manhattan_distance]
    """
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
    mse = function(np.square(embeddings1 - embeddings2), axis=1)

    # Compute Manhattan distance
    manhattan = function(np.abs(embeddings1 - embeddings2), axis=1)

    return [cosine_similarities, mse, manhattan]


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

  # Use FP32 precision instead of default BF16
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch --input data.txt --precision fp32

  # Compare PyTorch with Ollama (same model name)
  python {script_name} --model nomic-embed-text --backends pytorch ollama --input data.txt

  # Compare with Ollama using different model name
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch ollama --ollama-model all-minilm --input data.txt

  # Compare Ollama with custom URL and model name
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends ollama --ollama-url http://localhost:11434 --ollama-model nomic-embed-text --input data.txt

  # Compare all backends
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch onnx openvino ollama --onnx-path ./model.onnx --openvino-path ./model.xml --ollama-model all-minilm --input data.txt

  # Direct string input (single sentence)
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch --input "What is machine learning?"

  # JSONL file with auto-detected text field
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch --input data.jsonl

  # JSONL file with custom field path
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch --input data.jsonl --field-path document.text

  # JSONL with array wildcard (all items)
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch --input data.jsonl --field-path documents[*].text

  # Compressed JSONL file
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch ollama --input data.jsonl.bz2 --field-path question

  # Save results to JSON file
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch onnx --input data.txt --output-file results.json

  # JSONL input with JSON output
  python {script_name} --model nomic-embed-text --backends pytorch ollama --input data.jsonl --field-path text --output-file benchmark_results.json

  # Save results including embeddings (warning: creates large files)
  python {script_name} --model sentence-transformers/all-MiniLM-L6-v2 --backends pytorch onnx --input data.txt --output-file results_with_embeddings.json --save-embeddings
        """
    )
    # ... rest of the argument definitions ...
    parser.add_argument("--model", required=True, help="SentenceTransformer model name or path")
    parser.add_argument('--backends', nargs='+',
                        choices=['pytorch', 'onnx', 'openvino', 'tensorrt', 'vllm', 'ollama'],
                        default=['pytorch', 'onnx'],
                        help='Backends to compare')
    parser.add_argument("--input", required=True, help="Path to input file (text, JSONL, JSONL.bz2) or direct string to encode")
    parser.add_argument("--field-path", "--field_path", dest="field_path",
                       type=str, default=None,
                       help="Dot-separated path to text field in JSONL (e.g., 'document.text', 'documents[*].text'). "
                            "Supports array indexing: 'documents[0].text' (first item), 'documents[*].text' (all items). "
                            "If not specified for JSONL files, will try common fields: 'text', 'content', 'question'")

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
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"],
                       help="Model precision for PyTorch backend (default: bf16)")
    parser.add_argument("--max-text-length", "--max_text_length", dest="max_text_length",
                       type=int, default=None, help="Maximum text length (default: no limit)")
    parser.add_argument("--num-threads", "--num_threads", dest="num_threads", type=int, default=None,
                       help="Number of CPU threads for PyTorch (default: all available)")
    parser.add_argument("--openvino-device", "--openvino_device", dest="openvino_device",
                       type=str, default="CPU", help="OpenVINO device (CPU, GPU, etc.) (default: CPU)")
    parser.add_argument("--ollama-url", "--ollama_url", dest="ollama_url",
                       type=str, default="http://localhost:11434",
                       help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--ollama-model", "--ollama_model", dest="ollama_model",
                       type=str, default=None,
                       help="Ollama model name (if different from --model). E.g., 'nomic-embed-text' for Ollama vs HF model name")
    parser.add_argument("--output-file", "--output_file", dest="output_file",
                       type=str, default=None,
                       help="Path to save results (JSON format). If not specified, only prints to stdout")
    parser.add_argument("--output-format", "--output_format", dest="output_format",
                       type=str, default="json", choices=["json"],
                       help="Output file format (default: json)")
    parser.add_argument("--save-embeddings", "--save_embeddings", dest="save_embeddings",
                       action="store_true",
                       help="Include embeddings in the JSON output (warning: can create large files)")

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
                args.model, backend='pytorch', device=args.device, precision=args.precision
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
        elif backend == "ollama":
            # Add Ollama-specific parameters
            # Use ollama_model if specified, otherwise use the HF model name
            ollama_model_name = args.ollama_model if args.ollama_model else args.model
            model_kwargs = {
                'base_url': args.ollama_url,
            }
            models[backend] = load_sentence_transformer_with_backend(
                ollama_model_name, backend='ollama', device=args.device,
                **model_kwargs
            )
        elif backend == "vllm":
            # Add any vLLM-specific parameters
            model_kwargs = {
                'max_model_len': 512,  # Adjust based on your needs
                'gpu_memory_utilization': 0.8,
            }
            models[backend] = load_sentence_transformer_with_backend(args.model, backend, model_kwargs)

    # Read sentences
    sentences = read_sentences(args.input, args.max_text_length, args.field_path)
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
    # Standard deviation requires at least 2 samples
    if len(lengths) > 1:
        print(f"Std dev: {statistics.stdev(lengths):.1f} chars")
    else:
        print(f"Std dev: N/A (only 1 sample)")

    # Process in batches
    num_sentences = len(sentences)
    batches = [sentences[i:i + args.batch_size] for i in range(0, num_sentences, args.batch_size)]

    # Warmup phase - run a few batches to initialize models
    if num_sentences > 100:
        print("\nWarming up models...")
        warmup_examples = min(5, num_sentences)  # Use up to 5 examples for warmup
        warmup_sentences = sentences[:warmup_examples]
        for backend in args.backends:
            print(f"  Warming up {backend.upper()}...")
            _ = get_embeddings(models[backend], warmup_sentences, convert_to_numpy=False)
        print("✓ Warmup completed")

    # Initialize tracking variables
    backend_times = {backend: 0 for backend in args.backends}
    backend_embeddings = {backend: [] for backend in args.backends}

    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.1f} MB")

    print("\nProcessing batches...")
    scores = {}
    min_vals = {}
    arg_min_vals = {}
    for i, batch in enumerate(tqdm(batches)):
        batch_embeddings = [[] for _ in args.backends]

        # Generate embeddings for each backend
        for i1, backend in enumerate(args.backends):
            start_time = time.time()
            embeddings = get_embeddings(models[backend], batch, convert_to_numpy=False, batch_no=i)
            backend_times[backend] += time.time() - start_time
            batch_embeddings[i1] = embeddings

            # Store embeddings if requested
            if args.save_embeddings:
                # Convert to numpy if needed
                if isinstance(embeddings, list):
                    embeddings_np = np.array(embeddings)
                elif hasattr(embeddings, 'cpu'):  # torch tensor
                    embeddings_np = embeddings.cpu().numpy()
                else:
                    embeddings_np = np.array(embeddings)
                backend_embeddings[backend].append(embeddings_np)

        for i1 in range(len(args.backends)):
            for j in range(i1+1, len(args.backends)):
                key = f"{args.backends[j]} {args.backends[i1]}"
                if key not in scores:
                    scores[key] = [[],[],[]]

                res = compute_similarity(batch_embeddings[i1], batch_embeddings[j], return_mean=False)
                for v in range(len(scores[key])):
                    scores[key][v].extend(res[v])

        # Clean up batch embeddings
        del batch_embeddings

        # Memory management
        if i % 50 == 0 and i > 0:
            current_memory = get_memory_usage()
            if current_memory > initial_memory * 1.5:
                gc.collect()
                print(f"GC triggered at batch {i}, memory: {current_memory:.1f}MB")

    # Normalize the results:
    # for k in scores.keys():
    #     scores[k] /= num_sentences

    # Concatenate embeddings from all batches if they were collected
    if args.save_embeddings:
        print("\nConcatenating embeddings from all batches...")
        for backend in args.backends:
            if backend_embeddings[backend]:
                backend_embeddings[backend] = np.vstack(backend_embeddings[backend])
                print(f"  {backend}: shape {backend_embeddings[backend].shape}")

    # Initialize results dictionary for JSON output
    results_dict = {
        'config': {
            'model': args.model,
            'backends': args.backends,
            'num_sentences': num_sentences,
            'batch_size': args.batch_size,
            'device': args.device,
            'precision': args.precision,
            'input_file': args.input,
            'field_path': args.field_path if hasattr(args, 'field_path') else None,
            'save_embeddings': args.save_embeddings,
        },
        'performance': {},
        'similarity': {}
    }

    # Add embeddings to results if requested
    if args.save_embeddings:
        results_dict['embeddings'] = {}
        for backend in args.backends:
            if isinstance(backend_embeddings[backend], np.ndarray):
                # Convert numpy array to list for JSON serialization
                results_dict['embeddings'][backend] = backend_embeddings[backend].tolist()
                print(f"  Added {backend} embeddings to results (shape: {backend_embeddings[backend].shape})")

    # Report performance results
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    num_docs = num_sentences
    baseline_backend = args.backends[0]  # Use first backend as baseline
    baseline_time = backend_times[baseline_backend]

    # Print header
    print(f"\n{'Backend':<15} {'Time (s)':<12} {'Throughput':<18} {'Speedup':<10}")
    print("-" * 70)

    for backend in args.backends:
        time_val = backend_times[backend]
        throughput = num_docs / time_val if time_val > 0 else float('inf')
        speedup = baseline_time / time_val if time_val > 0 and backend != baseline_backend else 1.0

        # Store in results dictionary
        results_dict['performance'][backend] = {
            'time_seconds': float(time_val),
            'throughput_docs_per_sec': float(throughput),
            'speedup': float(speedup) if backend != baseline_backend else None,
            'is_baseline': backend == baseline_backend
        }

        if backend == baseline_backend:
            print(f"{backend.upper():<15} {time_val:>8.2f}s    {throughput:>8.1f} docs/s    {'(baseline)':<10}")
        else:
            print(f"{backend.upper():<15} {time_val:>8.2f}s    {throughput:>8.1f} docs/s    {speedup:>6.2f}x")

    # Report embedding similarity comparisons
    if len(args.backends) > 1:
        print("\n" + "=" * 70)
        print("EMBEDDING SIMILARITY COMPARISON")
        print("=" * 70)

        # Compare each backend with the baseline (first backend)
        baseline_embeddings = backend_embeddings[baseline_backend]
        arg_min = 0

        for backend in args.backends[1:]:
            _, _, _, arg_min = _compute_and_print_similarity(baseline_backend, backend,
                                                            scores, results_dict, verbose=True)

        # Compare all pairs if more than 2 backends
        if len(args.backends) > 2:
            print(f"\nAdditional pairwise comparisons:")
            for i, backend1 in enumerate(args.backends[1:], 1):
                for backend2 in args.backends[i + 1:]:
                    cosine_sim, _, _ = _compute_and_print_similarity(backend1, backend2, scores, results_dict,
                                                                     verbose=False)
                    print(f"\n{backend1.upper()} vs {backend2.upper()}:")
                    print(f"  Cosine Similarity: {cosine_sim:.6f}")
        print("\n"+ "=" * 70)
        print(f"The min cosine is realized for \n {sentences[arg_min][:300]} ...")

    # Save results to JSON if output file specified
    if args.output_file:
        save_results_json(args.output_file, results_dict)


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


def _compute_and_print_similarity(backend1: str, backend2: str, scores: dict, results_dict: dict, verbose: bool = True):
    """Compute and optionally print similarity statistics between two backends.

    Args:
        backend1: First backend name
        backend2: Second backend name
        scores: Dictionary containing similarity scores
        results_dict: Dictionary to store results
        verbose: Whether to print detailed statistics

    Returns:
        Tuple of (cosine_sim, mse_mean, manhattan)
    """
    key = f"{backend2} {backend1}"
    cosine_sim, cosine_median, cosine_std, cosine_min, cosine_arg_min = compute_stats(scores[key][0])
    mse_mean, mse_std, _, _, _ = compute_stats(scores[key][1])
    manhattan, manhattan_std, _, _, _ = compute_stats(scores[key][2])


    # Compute histogram of cosine similarity scores
    cosine_scores = np.array(scores[key][0])
    total_count = len(cosine_scores)

    # Define ranges and compute counts
    range_099_1 = np.sum((cosine_scores >= 0.99) & (cosine_scores <= 1.0))
    range_098_099 = np.sum((cosine_scores >= 0.98) & (cosine_scores < 0.99))
    range_095_098 = np.sum((cosine_scores >= 0.95) & (cosine_scores < 0.98))
    range_0_095 = np.sum((cosine_scores >= 0.0) & (cosine_scores < 0.95))

    # Calculate percentages
    pct_099_1 = (range_099_1 / total_count * 100) if total_count > 0 else 0
    pct_098_099 = (range_098_099 / total_count * 100) if total_count > 0 else 0
    pct_095_098 = (range_095_098 / total_count * 100) if total_count > 0 else 0
    pct_0_095 = (range_0_095 / total_count * 100) if total_count > 0 else 0

    # Store in results dictionary
    comparison_key = f"{backend1}_vs_{backend2}"
    results_dict['similarity'][comparison_key] = {
        'backend1': backend1,
        'backend2': backend2,
        'mean_cosine_similarity': float(cosine_sim),
        'mean_squared_error': float(mse_mean),
        'mean_manhattan_distance': float(manhattan),
        'std_cosine_similarity': float(cosine_std),
        'min_cosine_similarity': float(cosine_min),
        'arg_min_cosine_similarity': int(cosine_arg_min),
        'histogram': {
            'range_0.99_1.0': {'count': int(range_099_1), 'percentage': float(pct_099_1)},
            'range_0.98_0.99': {'count': int(range_098_099), 'percentage': float(pct_098_099)},
            'range_0.95_0.98': {'count': int(range_095_098), 'percentage': float(pct_095_098)},
            'range_0.0_0.95': {'count': int(range_0_095), 'percentage': float(pct_0_095)},
        }
    }

    if verbose:
        print(f"\n{backend1.upper()} vs {backend2.upper()}:")
        print(f"  Mean Cosine Similarity:    {cosine_sim:.6f} (1.0 = identical)")
        print(f"  Median Cosine Similarity:  {cosine_median:.6f}")
        print(f"  Std Dev Cosine Similarity: {cosine_std:.6f}")
        print(f"  Min Cosine Similarity:     {cosine_min:.6f}")
        print(f"  Arg Min Cosine Similarity: {int(cosine_arg_min)}")
        print(f"  Mean Squared Error:        {mse_mean:.6f} ± {mse_std:.2f} (0.0 = identical)")
        print(f"  Mean Manhattan Distance:   {manhattan:.6f} ± {manhattan_std:.2f} (0.0 = identical)")

        # Display histogram
        print(f"\n  Cosine Similarity Distribution:")
        print(f"    [0.99, 1.00]: {range_099_1:6d} ({pct_099_1:5.1f}%)")
        print(f"    [0.98, 0.99): {range_098_099:6d} ({pct_098_099:5.1f}%)")
        print(f"    [0.95, 0.98): {range_095_098:6d} ({pct_095_098:5.1f}%)")
        print(f"    [0.00, 0.95): {range_0_095:6d} ({pct_0_095:5.1f}%)")

        _analyze_results(cosine_sim, f"{backend1.upper()}/{backend2.upper()}")

    return cosine_sim, mse_mean, manhattan, cosine_arg_min


def save_results_json(output_file: str, results: dict):
    """Save benchmark results to JSON file.

    Args:
        output_file: Path to output JSON file
        results: Dictionary containing benchmark results
    """
    import json
    from datetime import datetime

    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'script_version': '1.0'
    }

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
