#!/usr/bin/env python3
"""
Compute embeddings from JSONL files with text tiling support.

This script reads text inputs and document/query IDs from a JSONL file,
optionally splits long texts using TextTiler, computes embeddings with
a specified model, and saves the results as a pickle file.

Usage examples:
    # Basic usage with simple field paths
    python compute_embeddings_from_jsonl.py data.jsonl output.pkl \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --text_field text --id_field id

    # With nested field paths (jq-like syntax)
    python compute_embeddings_from_jsonl.py data.jsonl output.pkl \\
        --model intfloat/e5-base-v2 \\
        --text_field document.content --id_field document.doc_id

    # With text tiling enabled
    python compute_embeddings_from_jsonl.py data.jsonl output.pkl \\
        --model BAAI/bge-base-en-v1.5 \\
        --text_field text --id_field id \\
        --tile --max_length 512 --stride 128 \\
        --sentence_aligned

    # With truncation instead of tiling
    python compute_embeddings_from_jsonl.py data.jsonl output.pkl \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --text_field text --id_field id \\
        --truncate --max_length 512
"""

import argparse
import json
import pickle
import sys
import time
from functools import partial
from pathlib import Path
from typing import List, Tuple, Optional

from tqdm import tqdm
import numpy as np
from collections import Counter

# Import TextTiler and embedding utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from docuverse.utils.text_tiler import TextTiler
from docuverse.utils import open_stream, parallel_process
from docuverse.utils.jsonl_utils import get_nested_field
from docuverse.utils.timer import timer

# Milvus imports (optional)
try:
    from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


def print_histogram_stats(values: List[int], name: str = "values", num_bins: int = 10):
    """
    Print histogram statistics for a list of values.

    Args:
        values: List of numeric values
        name: Name of the values being displayed
        num_bins: Number of bins for histogram
    """
    if not values:
        return

    arr = np.array(values)
    print(f"\n{name} statistics:")
    print(f"  Count: {len(arr)}")
    print(f"  Min: {arr.min()}")
    print(f"  Max: {arr.max()}")
    print(f"  Mean: {arr.mean():.1f}")
    print(f"  Median: {np.median(arr):.1f}")
    print(f"  Std: {arr.std():.1f}")

    # Create histogram
    hist, bin_edges = np.histogram(arr, bins=num_bins)
    print(f"\n  Histogram ({num_bins} bins):")
    for i in range(len(hist)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        count = hist[i]
        pct = 100 * count / len(arr)
        bar = '█' * int(pct / 2)  # Scale bar to 50 chars max
        print(f"    [{bin_start:8.1f} - {bin_end:8.1f}): {count:6d} ({pct:5.1f}%) {bar}")


def load_jsonl_data(
    file_path: str,
    text_field: str,
    id_field: str,
    title_field: Optional[str] = None,
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load texts, IDs, and optionally titles from a JSONL file.

    Args:
        file_path: Path to JSONL or JSONL.bz2 file
        text_field: Dot-separated path to text field (e.g., "document.text")
        id_field: Dot-separated path to ID field (e.g., "id" or "doc.id")
        title_field: Optional dot-separated path to title field
        max_samples: Maximum number of samples to read
        verbose: Print warnings for skipped lines

    Returns:
        Tuple of (texts, ids, titles) where titles may be empty strings
    """

    texts = []
    ids = []
    titles = []

    # is_compressed = file_path.endswith('.bz2')
    #
    # if is_compressed:
    #     file_handle = bz2.open(file_path, 'rt', encoding='utf-8')
    # else:
    #     file_handle = open(file_path, 'r', encoding='utf-8')
    file_handle = open_stream(file_path)
    try:
        for i, line in enumerate(file_handle):
            if max_samples is not None and i >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Skipping invalid JSON at line {i+1}: {e}")
                continue

            try:
                # Extract text
                text = get_nested_field(data, text_field)
                if not isinstance(text, str):
                    text = str(text)

                # Extract ID
                doc_id = get_nested_field(data, id_field)
                if not isinstance(doc_id, str):
                    doc_id = str(doc_id)

                # Extract title if specified
                title = ""
                if title_field:
                    try:
                        title = get_nested_field(data, title_field)
                        if not isinstance(title, str):
                            title = str(title)
                    except (KeyError, IndexError):
                        if verbose:
                            print(f"Warning: Title field '{title_field}' not found at line {i+1}")

                texts.append(text)
                ids.append(doc_id)
                titles.append(title)

            except (KeyError, IndexError) as e:
                if verbose:
                    print(f"Warning: {e} at line {i+1}")
                continue

    finally:
        file_handle.close()

    return texts, ids, titles


def _tile_document_worker(data_item, tiler):
    """
    Module-level worker function for parallel tiling.
    Creates a tiler instance from parameters packaged with the data item.

    Args:
        data_item: Tuple of (text, doc_id, title, tiler_kwargs, title_handling)
    """
    text, doc_id, title, title_handling = data_item

    try:
        # Create a tiler instance in this worker process
        # tiler = TextTiler(**tiler_kwargs)

        tiles = tiler.create_tiles(
            id_=doc_id,
            text=text,
            title=title,
            title_handling=title_handling
        )
        return tiles
    except Exception as e:
        print(f"Error in worker for doc {doc_id}: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_with_tiling(
    texts: List[str],
    ids: List[str],
    titles: List[str],
    tiler: TextTiler,
    title_handling: str = "none",
    num_workers: int = 1,
    verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Process texts with TextTiler to split long documents.

    Args:
        texts: List of input texts
        ids: List of document IDs
        titles: List of document titles
        tiler: TextTiler instance
        title_handling: How to handle titles ('all', 'first', 'none')
        num_workers: Number of parallel workers (1 = sequential)
        verbose: Show progress bar

    Returns:
        Tuple of (tiled_texts, tiled_ids)
    """
    # Prepare tiler parameters for workers
    # Get tokenizer name/path for pickling (can't pickle tokenizer object directly)
    tokenizer_param = tiler.tokenizer
    if hasattr(tiler.tokenizer, 'name_or_path'):
        tokenizer_param = tiler.tokenizer.name_or_path
    elif hasattr(tiler.tokenizer, 'model_name'):
        tokenizer_param = tiler.tokenizer.model_name

    tiler_kwargs = {
        'max_doc_length': tiler.max_doc_size,
        'stride': tiler.stride,
        'tokenizer': tokenizer_param,
        'aligned_on_sentences': tiler.aligned_on_sentences,
        'count_type': 'token' if tiler.count_type == TextTiler.COUNT_TYPE_TOKEN else 'char',
        'trim_text_to': tiler.text_trim_to,
        'trim_text_count_type': tiler.text_trim_to_type
    }
    tiler = TextTiler(**tiler_kwargs)

    # Package data with tiler parameters: (text, id, title, tiler_kwargs, title_handling)
    data = [(text, doc_id, title, title_handling)
            for text, doc_id, title in zip(texts, ids, titles)]

    tiler_func = partial(_tile_document_worker, tiler=tiler)

    if num_workers <= 1:
        # Sequential processing
        all_tiles = []
        iterator = data
        if verbose:
            iterator = tqdm(data, desc="Tiling texts", leave=True)

        for data_item in iterator:
            tiles = tiler_func(data_item=data_item)
            all_tiles.extend(tiles)
    else:
        # Parallel processing
        if verbose:
            print(f"Tiling texts with {num_workers} workers...")

        results = parallel_process(
            process_func=tiler_func,
            data=data,
            num_threads=num_workers,
            msg="Tiling texts"
        )

        # Flatten results
        all_tiles = []
        for tiles in results:
            all_tiles.extend(tiles)

    # Extract texts and IDs from tiles
    tiled_texts = [tile['text'] for tile in all_tiles]
    tiled_ids = [tile['id'] for tile in all_tiles]

    return tiled_texts, tiled_ids


def truncate_texts(
    texts: List[str],
    tokenizer,
    max_length: int,
    verbose: bool = True
) -> List[str]:
    """
    Truncate texts to a maximum token length.

    Args:
        texts: List of input texts
        tokenizer: Tokenizer to use for counting tokens
        max_length: Maximum number of tokens
        verbose: Show progress bar

    Returns:
        List of truncated texts
    """
    truncated = []

    iterator = texts
    if verbose:
        iterator = tqdm(texts, desc="Truncating texts")

    for text in iterator:
        tokens = tokenizer(text, truncation=True, max_length=max_length)
        truncated_text = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
        truncated.append(truncated_text)

    return truncated


def compute_embeddings(
    texts: List[str],
    model,
    batch_size: int = 128,
    normalize: bool = True,
    prompt_name: Optional[str] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.

    Args:
        texts: List of input texts
        model: SentenceTransformer model
        batch_size: Batch size for encoding
        normalize: Whether to normalize embeddings
        prompt_name: Optional prompt name for models that support it
        verbose: Show progress bar

    Returns:
        Numpy array of embeddings with shape (num_texts, embedding_dim)
    """
    print(f"Computing embeddings for {len(texts)} texts...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=verbose,
        normalize_embeddings=normalize,
        prompt_name=prompt_name
    )

    return embeddings


def save_embeddings_pickle(
    embeddings: np.ndarray,
    ids: List[str],
    texts: List[str],
    output_path: str
):
    """
    Save embeddings and IDs to a pickle file.

    Args:
        embeddings: Numpy array of embeddings
        ids: List of document IDs
        texts: List of texts (not used for pickle, kept for API compatibility)
        output_path: Path to output pickle file
    """
    data = {
        'embeddings': embeddings,
        'ids': ids
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved {len(ids)} embeddings to {output_path}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")


def save_embeddings_milvus(
    embeddings: np.ndarray,
    ids: List[str],
    texts: List[str],
    output_path: str,
    embedding_dim: int,
    verbose: bool = True
):
    """
    Save embeddings, IDs, and texts to a Milvus file database.

    Args:
        embeddings: Numpy array of embeddings
        ids: List of document IDs
        texts: List of texts
        output_path: Path to Milvus database file
        embedding_dim: Dimension of embeddings
        verbose: Show progress
    """
    if not MILVUS_AVAILABLE:
        raise ImportError(
            "pymilvus is not installed. Install it with: pip install pymilvus"
        )

    # Extract collection name from output path (remove .db extension)
    collection_name = Path(output_path).stem
    from docuverse.utils.milvus import sanitize_milvus_collection_name
    collection_name = sanitize_milvus_collection_name(collection_name)

    if verbose:
        print(f"Creating Milvus database: {output_path}")
        print(f"  Collection name: {collection_name}")

    # Create Milvus client with file-based storage
    client = MilvusClient(uri=output_path)

    # Check if collection exists and drop it
    if client.has_collection(collection_name):
        if verbose:
            print(f"  Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name)

    # Define schema
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        ],
        description=f"Embeddings collection from JSONL"
    )

    # Prepare index params
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embeddings",
        index_type="FLAT",  # Use FLAT for simplicity with file-based storage
        metric_type="IP",    # Inner Product (for normalized embeddings)
    )

    # Create collection
    if verbose:
        print(f"  Creating collection with schema...")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    # Prepare data for insertion
    if verbose:
        print(f"  Preparing data for insertion...")

    data_to_insert = []
    for doc_id, text, embedding in zip(ids, texts, embeddings):
        # Truncate text if too long
        if len(text) > 65000:
            text = text[:65000] + "..."

        data_to_insert.append({
            "id": doc_id,
            "text": text,
            "embeddings": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        })

    # Insert data in batches
    batch_size = 100
    total_inserted = 0

    if verbose:
        iterator = tqdm(range(0, len(data_to_insert), batch_size), desc="Inserting data")
    else:
        iterator = range(0, len(data_to_insert), batch_size)

    for i in iterator:
        batch = data_to_insert[i:i + batch_size]
        client.insert(collection_name=collection_name, data=batch)
        total_inserted += len(batch)

    if verbose:
        print(f"Saved {total_inserted} embeddings to Milvus database: {output_path}")
        print(f"  Collection name: {collection_name}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")

    # Close client
    client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute embeddings from JSONL files with text tiling support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python compute_embeddings_from_jsonl.py data.jsonl output.pkl \\
      --model sentence-transformers/all-MiniLM-L6-v2 \\
      --text_field text --id_field id

  # With text tiling (sentence-aligned, 512 tokens, 128 overlap)
  python compute_embeddings_from_jsonl.py data.jsonl output.pkl \\
      --model BAAI/bge-base-en-v1.5 \\
      --text_field document.content --id_field document.id \\
      --tile --max_length 512 --stride 128 --sentence_aligned

  # With truncation (no tiling)
  python compute_embeddings_from_jsonl.py data.jsonl output.pkl \\
      --model intfloat/e5-base-v2 \\
      --text_field text --id_field id \\
      --truncate --max_length 512

  # With nested fields and titles
  python compute_embeddings_from_jsonl.py data.jsonl output.pkl \\
      --model sentence-transformers/all-mpnet-base-v2 \\
      --text_field document.text --id_field document.id \\
      --title_field document.title --title_handling first \\
      --tile --max_length 384

  # Save to Milvus database
  python compute_embeddings_from_jsonl.py data.jsonl embeddings.db \\
      --model BAAI/bge-base-en-v1.5 \\
      --text_field text --id_field id \\
      --output_format milvus
        """
    )

    # Input/output arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL or JSONL.bz2 file'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output file (.pkl for pickle, .db for Milvus)'
    )
    parser.add_argument(
        '--output_format',
        type=str,
        default='pickle',
        choices=['pickle', 'milvus'],
        help='Output format: pickle (default) or milvus database'
    )

    # Field specification
    parser.add_argument(
        '--text_field',
        type=str,
        required=True,
        help='Dot-separated path to text field (e.g., "text", "document.content")'
    )
    parser.add_argument(
        '--id_field',
        type=str,
        required=True,
        help='Dot-separated path to ID field (e.g., "id", "document.doc_id")'
    )
    parser.add_argument(
        '--title_field',
        type=str,
        default=None,
        help='Optional dot-separated path to title field'
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='SentenceTransformer model name or path'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to run on (default: cuda)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for encoding (default: 128)'
    )
    parser.add_argument(
        '--prompt_name',
        type=str,
        default=None,
        help='Prompt name for models that support it (e.g., "query", "passage")'
    )
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Disable embedding normalization'
    )

    # Text processing mode
    processing_group = parser.add_mutually_exclusive_group()
    processing_group.add_argument(
        '--tile',
        action='store_true',
        help='Enable text tiling for long documents'
    )
    processing_group.add_argument(
        '--truncate',
        action='store_true',
        help='Truncate texts to max_length (no tiling)'
    )

    # TextTiler arguments
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum document length in tokens (default: 512)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=128,
        help='Overlap/stride for tiling in tokens (default: 128)'
    )
    parser.add_argument(
        '--sentence_aligned',
        action='store_true',
        help='Align tiles on sentence boundaries (requires pyizumo)'
    )
    parser.add_argument(
        '--count_type',
        type=str,
        default='token',
        choices=['token', 'char'],
        help='Count type for max_length and stride (default: token)'
    )
    parser.add_argument(
        '--title_handling',
        type=str,
        default='none',
        choices=['all', 'first', 'none'],
        help='How to add titles to tiles: all=every tile, first=first tile only, none=no titles (default: none)'
    )
    parser.add_argument(
        '--trim_text_to',
        type=int,
        default=None,
        help='Trim text to this many tokens before tiling (default: no trimming)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers for tiling (default: 1, sequential). Use 4-8 for faster processing.'
    )

    # Other arguments
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process from input file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress bars and warnings'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable detailed profiling and timing information'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Initialize profiling timer
    tm = timer("compute_embeddings", disable=not args.profile)
    tm.mark()

    # Validate inputs
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    if args.output_format == 'milvus' and not MILVUS_AVAILABLE:
        print("Error: --output_format milvus requires pymilvus package")
        print("Install it with: pip install pymilvus")
        return 1

    if args.tile and args.sentence_aligned:
        try:
            import pyizumo
        except ImportError:
            print("Error: --sentence_aligned requires pyizumo package")
            print("Install it with: pip install pyizumo")
            return 1

    # Load SentenceTransformer model
    if verbose:
        print(f"Loading model: {args.model}")

    try:
        from sentence_transformers import SentenceTransformer
        import torch
        from docuverse.utils import detect_device

        # Check device availability
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, detecting device ..", end='')
            args.device = detect_device()
            print(f" using {args.device}")
        # elif args.device == 'mps' and not torch.backends.mps.is_available():
        #     print("Warning: MPS not available, falling back to CPU")
        #     args.device = 'cpu'

        model = SentenceTransformer(args.model, device=args.device, trust_remote_code=True)
        tm.add_timing("model_loading")

        if verbose:
            print(f"  Device: {args.device}")
            print(f"  Max sequence length: {model.max_seq_length}")

    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Load data from JSONL
    if verbose:
        print(f"\nLoading data from: {args.input_file}")

    texts, ids, titles = load_jsonl_data(
        file_path=args.input_file,
        text_field=args.text_field,
        id_field=args.id_field,
        title_field=args.title_field,
        max_samples=args.max_samples,
        verbose=verbose
    )
    tm.add_timing("data_loading")

    if len(texts) == 0:
        print("Error: No texts loaded from input file")
        return 1

    if verbose:
        print(f"  Loaded {len(texts)} texts")
        print("\n".join([
            f"  First 5 texts:"]+
                        texts[:5]))
        avg_length = sum(len(t.split()) for t in texts) / len(texts)
        print(f"  Average text length: {avg_length:.1f} words")

        # Display histogram of text lengths (in words)
        text_lengths = [len(t.split()) for t in texts]
        print_histogram_stats(text_lengths, name="Text lengths (words)", num_bins=10)

    # Process texts based on mode
    if args.tile:
        if verbose:
            print(f"\nTiling texts with TextTiler:")
            print(f"  Max length: {args.max_length} {args.count_type}s")
            print(f"  Stride: {args.stride} {args.count_type}s")
            print(f"  Sentence aligned: {args.sentence_aligned}")
            print(f"  Title handling: {args.title_handling}")
            print(f"  Number of workers: {args.num_workers}")

        tiler = TextTiler(
            max_doc_length=args.max_length,
            stride=args.stride,
            tokenizer=model.tokenizer,
            aligned_on_sentences=args.sentence_aligned,
            count_type=args.count_type,
            trim_text_to=args.trim_text_to,
            trim_text_count_type=args.count_type
        )
        tm.add_timing("tiler_initialization")

        texts, ids = process_with_tiling(
            texts=texts,
            ids=ids,
            titles=titles,
            tiler=tiler,
            title_handling=args.title_handling,
            num_workers=args.num_workers,
            verbose=verbose
        )
        tm.add_timing("text_tiling")

        if verbose:
            # Display histogram of tile lengths (in words)
            tile_lengths = [len(t.split()) for t in texts]
            print_histogram_stats(tile_lengths, name="Tile lengths (tokens)", num_bins=10)

            # Display histogram of tiles per document
            id_counts = list(Counter(ids).values())
            print_histogram_stats(id_counts, name="Tiles per document", num_bins=min(10, max(id_counts)))
            print(f"  Created {len(texts)} tiles from {len(set(ids))} unique documents")
    elif args.truncate:
        if verbose:
            print(f"\nTruncating texts to {args.max_length} tokens")

        texts = truncate_texts(
            texts=texts,
            tokenizer=model.tokenizer,
            max_length=args.max_length,
            verbose=verbose
        )
        tm.add_timing("text_truncation")

    # Compute embeddings
    if verbose:
        print(f"\nComputing embeddings:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Normalize: {not args.no_normalize}")
        if args.prompt_name:
            print(f"  Prompt name: {args.prompt_name}")

    embeddings = compute_embeddings(
        texts=texts,
        model=model,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        prompt_name=args.prompt_name,
        verbose=verbose
    )
    tm.add_timing("embedding_computation")

    # Save results
    if verbose:
        print(f"\nSaving results to: {args.output_file}")
        print(f"  Output format: {args.output_format}")

    if args.output_format == 'pickle':
        save_embeddings_pickle(
            embeddings=embeddings,
            ids=ids,
            texts=texts,
            output_path=args.output_file
        )
    elif args.output_format == 'milvus':
        save_embeddings_milvus(
            embeddings=embeddings,
            ids=ids,
            texts=texts,
            output_path=args.output_file,
            embedding_dim=embeddings.shape[1],
            verbose=verbose
        )
    tm.add_timing("save_results")

    # Display profiling information if enabled
    if args.profile:
        total_time = tm.milliseconds_since_beginning()
        # Compute statistics
        num_docs = len(set(ids)) if args.tile or args.truncate else len(ids)
        num_tiles = len(ids)
        total_chars = sum(len(t) for t in texts)
        total_words = sum(len(t.split()) for t in texts)

        stats = {
            'documents': num_docs,
            'tiles': num_tiles,
            'chars': total_chars,
            'words': total_words
        }

        print("\n")
        timer.display_timing(totalms=total_time, keys=stats, output_stream=sys.stdout)

    if verbose:
        print("\n✓ Done!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
