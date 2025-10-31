import argparse
import json
import statistics

import numpy as np
# import onnxruntime as ort
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


def load_onnx_model(onnx_model_path: str, device: str = 'cuda'):
    """Load an ONNX model."""
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model file not found: {onnx_model_path}")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)

        # Load ONNX model
        print("Loading ONNX model...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(os.path.join(onnx_model_path, "model.onnx"), providers=providers)

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
    parser = argparse.ArgumentParser(description="Compare ONNX model with SentenceTransformer")
    parser.add_argument("--onnx", required=True, help="Path to the ONNX model file")
    parser.add_argument("--transformer", required=True, help="SentenceTransformer model name")
    parser.add_argument("--input", required=True, help="Path to the input text file with sentences")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use (default: all)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing (default: 128)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run models on (default: cuda)")
    parser.add_argument("--max_text_length", type=int, default=None, help="Maximum text length (default: no limit)")

    args = parser.parse_args()
    
    # Load models
    print(f"Loading ONNX model from {args.onnx}")
    onnx_model, tokenizer = load_onnx_model(args.onnx, device=args.device)

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

    total_cosine_sim = 0
    total_mse = 0
    total_manhattan = 0
    onnx_time = 0
    transformer_time = 0

    print("\nProcessing batches...")
    for i, batch in enumerate(tqdm(batches)):
        # Generate embeddings for current batch
        start_time = time.time()
        onnx_batch_embeddings = get_onnx_embeddings(onnx_model, tokenizer, batch)
        onnx_time += time.time() - start_time

        start_time = time.time()
        transformer_batch_embeddings = get_sentence_transformer_embeddings(sentence_transformer, batch)
        transformer_time += time.time() - start_time

        # Check and adjust embedding shapes if needed
        if onnx_batch_embeddings.shape != transformer_batch_embeddings.shape:
            min_dim = min(onnx_batch_embeddings.shape[1], transformer_batch_embeddings.shape[1])
            onnx_batch_embeddings = onnx_batch_embeddings[:, :min_dim]
            transformer_batch_embeddings = transformer_batch_embeddings[:, :min_dim]

        # Compute similarities for current batch
        batch_cosine, batch_mse, batch_manhattan = compute_similarity(
            onnx_batch_embeddings, transformer_batch_embeddings)

        # Update running averages
        weight = len(batch) / ((i + 1) * args.batch_size)
        total_cosine_sim = (total_cosine_sim * (1 - weight)) + (batch_cosine * weight)
        total_mse = (total_mse * (1 - weight)) + (batch_mse * weight)
        total_manhattan = (total_manhattan * (1 - weight)) + (batch_manhattan * weight)
        del onnx_batch_embeddings
        del transformer_batch_embeddings

    # Report results
    print(f"\nONNX processing time: {onnx_time:.2f} seconds")
    print(f"SentenceTransformer processing time: {transformer_time:.2f} seconds")
    print(f"Speed improvement: {transformer_time / onnx_time:.2f}x")

    print("\nEmbedding Comparison:")
    print(f"Mean Cosine Similarity: {total_cosine_sim:.6f} (1.0 is identical)")
    print(f"Mean Squared Error: {total_mse:.6f} (0.0 is identical)")
    print(f"Mean Manhattan Distance: {total_manhattan:.6f} (0.0 is identical)")

    # Analyze results
    if total_cosine_sim > 0.99:
        print("\nResult: The models produce very similar embeddings!")
    elif total_cosine_sim > 0.95:
        print("\nResult: The models produce similar embeddings with minor differences.")
    else:
        print("\nResult: There are significant differences between the model outputs.")

    # Report potential issues
    if total_cosine_sim < 0.9:
        print("\nPotential issues:")
        print("- ONNX conversion might not be exact")
        print("- Models might be using different preprocessing steps")
        print("- Check if tokenization is consistent between models")


if __name__ == "__main__":
    main()
