import argparse
import json

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



def load_onnx_model(onnx_model_path: str, device: str = 'cuda'):
    """Load an ONNX model."""
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model file not found: {onnx_model_path}")
    
    try:
        ort_model = ORTModelForFeatureExtraction.from_pretrained(onnx_model_path)
        ort_model.to(device)
        return ort_model
    except Exception as e:
        raise RuntimeError(f"Error loading ONNX model: {e}")


def load_sentence_transformer(model_name: str, device='cuda'):
    """Load a SentenceTransformer model."""
    try:
        model = SentenceTransformer(model_name, device=device)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading SentenceTransformer model: {e}")


def read_sentences(file_path: str) -> List[str]:
    """Read sentences from a file, one per line."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # with open(file_path, "r", encoding="utf-8") as f:
    #     sentences = [line.strip() for line in f if line.strip()]
    with open_stream(file_path) as f:
        sentences = [json.loads(s)['text'] for s in f.readlines()]

    return sentences


def get_onnx_embeddings(onnx_model, tokenizer,
                        sentences: List[str], batch_size: int = 128) -> Tuple[np.ndarray, float]:
    """Generate embeddings using the ONNX model."""
    start_time = time.time()
    
    # Process and get embeddings
    all_embeddings = []
    batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

    for batch in tqdm(batches, desc="ONNX embeddings"):
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="np")
        outputs = onnx_model(**inputs)
        for cls in outputs[0][:, 0]:
            # Normalize cls vector to have norm2 = 1  
            cls = cls / np.linalg.norm(cls)
            all_embeddings.append(cls)
    
    duration = time.time() - start_time
    return np.array(all_embeddings), duration


def get_sentence_transformer_embeddings(model, sentences: List[str],
                                        batch_size: int=128) -> Tuple[np.ndarray, float]:
    """Generate embeddings using the SentenceTransformer model."""
    start_time = time.time()
    
    with torch.no_grad():
        embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True,
                                  batch_size=batch_size)
    
    duration = time.time() - start_time
    return embeddings, duration


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

    args = parser.parse_args()
    
    # Load models
    print(f"Loading ONNX model from {args.onnx}")
    onnx_model = load_onnx_model(args.onnx)
    
    print(f"Loading SentenceTransformer model: {args.transformer}")
    sentence_transformer = load_sentence_transformer(args.transformer)
    
    # Read sentences
    sentences = read_sentences(args.input)
    if args.num_samples and args.num_samples < len(sentences):
        print(f"Using {args.num_samples} out of {len(sentences)} sentences")
        sentences = sentences[:args.num_samples]
    else:
        print(f"Using all {len(sentences)} sentences")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    onnx_embeddings, onnx_time = get_onnx_embeddings(onnx_model, sentence_transformer.tokenizer, sentences, args.batch_size)
    transformer_embeddings, transformer_time = get_sentence_transformer_embeddings(sentence_transformer, sentences, args.batch_size)
    
    # Check embedding shapes
    print(f"\nONNX embedding shape: {onnx_embeddings.shape}")
    print(f"SentenceTransformer embedding shape: {transformer_embeddings.shape}")
    
    if onnx_embeddings.shape != transformer_embeddings.shape:
        print("WARNING: Embedding shapes don't match!")
        min_dim = min(onnx_embeddings.shape[1], transformer_embeddings.shape[1])
        onnx_embeddings = onnx_embeddings[:, :min_dim]
        transformer_embeddings = transformer_embeddings[:, :min_dim]
        print(f"Truncated to shape: ({onnx_embeddings.shape[0]}, {min_dim})")
    
    # Compare performance
    print(f"\nONNX processing time: {onnx_time:.2f} seconds")
    print(f"SentenceTransformer processing time: {transformer_time:.2f} seconds")
    print(f"Speed improvement: {transformer_time/onnx_time:.2f}x")
    
    # Compare embeddings
    cosine_sim, mse, manhattan = compute_similarity(onnx_embeddings, transformer_embeddings)
    print("\nEmbedding Comparison:")
    print(f"Mean Cosine Similarity: {cosine_sim:.6f} (1.0 is identical)")
    print(f"Mean Squared Error: {mse:.6f} (0.0 is identical)")
    print(f"Mean Manhattan Distance: {manhattan:.6f} (0.0 is identical)")
    
    # Analyze results
    if cosine_sim > 0.99:
        print("\nResult: The models produce very similar embeddings!")
    elif cosine_sim > 0.95:
        print("\nResult: The models produce similar embeddings with minor differences.")
    else:
        print("\nResult: There are significant differences between the model outputs.")
    
    # Report potential issues
    if cosine_sim < 0.9:
        print("\nPotential issues:")
        print("- ONNX conversion might not be exact")
        print("- Models might be using different preprocessing steps")
        print("- Check if tokenization is consistent between models")


if __name__ == "__main__":
    main()
