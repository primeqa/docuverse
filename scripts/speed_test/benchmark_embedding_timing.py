#!/usr/bin/env python3
"""
Benchmark script to compare timing of OpenAI API-based embeddings vs GPU-based embeddings.

This script compares:
1. OpenAI-compatible API (RITS) - network-based inference
2. Local GPU with transformers - on-device inference

Usage:
    # With generated sample texts:
    python benchmark_embedding_timing.py --model_name ibm/slate-125m-english-rtrvr-v2 \
        --endpoint https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/slate-125m-english-rtrvr-v2 \
        --local_model_name ibm-granite/granite-embedding-125m-english \
        --num_samples 100 --batch_sizes 1,8,16,32

    # With JSONL file (auto-detect text field):
    python benchmark_embedding_timing.py --input_file data.jsonl --batch_sizes 1,8,16

    # With JSONL.bz2 file and custom field path:
    python benchmark_embedding_timing.py --input_file data.jsonl.bz2 \
        --field_path document.text --max_samples 500

    # With nested field path:
    python benchmark_embedding_timing.py --input_file nq-dev.jsonl \
        --field_path question --batch_sizes 1,16,32

    # Random sampling from large file:
    python benchmark_embedding_timing.py --input_file large.jsonl \
        --field_path text --num_samples 200 --random_seed 42

    # With plots:
    python benchmark_embedding_timing.py --input_file data.jsonl \
        --batch_sizes 1,8,16,32 --plot --plot_format pdf
"""

import argparse
import time
import os
import json
import bz2
import random
import numpy as np
from typing import List, Any, Dict
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from tqdm import tqdm


class APIEmbedder:
    """OpenAI API-based embedding generator."""

    def __init__(self, model_name: str, endpoint: str):
        api_key = os.environ.get("RITS_API_KEY")
        if api_key is None:
            raise ValueError("RITS_API_KEY environment variable is not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint + '/v1',
            default_headers={'RITS_API_KEY': api_key}
        )
        self.model = model_name
        print(f"✓ Initialized API embedder: {model_name}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using API."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)


class GPUEmbedder:
    """Local GPU-based embedding generator using transformers."""

    def __init__(self, model_name: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"Loading model on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print(f"✓ Initialized GPU embedder: {model_name} on {device}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local GPU."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_nested_field(obj: Dict[str, Any], path: str) -> Any:
    """
    Extract a nested field from a dictionary using dot notation with array support.

    Args:
        obj: Dictionary to extract from
        path: Dot-separated path with optional array indexing
              Examples:
              - "document.text" - simple nested field
              - "documents[*].text" - all items in array
              - "documents[0].text" - first item in array
              - "documents[].text" - same as [*], all items

    Returns:
        The value at the specified path. If array wildcard is used, returns list of values.

    Raises:
        KeyError: If path doesn't exist in the object
        IndexError: If array index is out of bounds
    """
    import re

    # Split path into segments, handling array notation
    # e.g., "documents[*].text" -> ["documents[*]", "text"]
    segments = path.split('.')
    current = obj

    for segment in segments:
        # Check if segment contains array indexing
        array_match = re.match(r'^([^\[]+)\[([^\]]*)\]$', segment)

        if array_match:
            # Handle array indexing: field[index] or field[*] or field[]
            field_name = array_match.group(1)
            index_str = array_match.group(2)

            # Navigate to the field
            if isinstance(current, dict):
                if field_name not in current:
                    raise KeyError(f"Key '{field_name}' not found in path '{path}'")
                current = current[field_name]
            else:
                raise KeyError(f"Cannot navigate to '{field_name}' in path '{path}' - current value is not a dict")

            # Handle the array indexing
            if not isinstance(current, list):
                raise KeyError(f"Field '{field_name}' in path '{path}' is not an array (got {type(current).__name__})")

            if index_str == '*' or index_str == '':
                # Wildcard: collect all items
                # Continue processing remaining path for each item
                remaining_path = '.'.join(segments[segments.index(segment) + 1:])
                if remaining_path:
                    # Recursively get nested field from each item
                    results = []
                    for item in current:
                        try:
                            results.append(get_nested_field({'_': item}, '_.' + remaining_path))
                        except (KeyError, IndexError):
                            continue  # Skip items that don't have the field
                    return results
                else:
                    # No more path, return all items
                    return current
            else:
                # Specific index
                try:
                    index = int(index_str)
                    current = current[index]
                except ValueError:
                    raise KeyError(f"Invalid array index '{index_str}' in path '{path}'")
                except IndexError:
                    raise IndexError(f"Array index {index} out of bounds in path '{path}'")
        else:
            # Regular field access
            if isinstance(current, dict):
                if segment not in current:
                    raise KeyError(f"Key '{segment}' not found in path '{path}'")
                current = current[segment]
            elif isinstance(current, list):
                # Implicit array wildcard - apply to all items
                remaining_path = '.'.join(segments[segments.index(segment):])
                results = []
                for item in current:
                    try:
                        results.append(get_nested_field({'_': item}, '_.' + remaining_path))
                    except (KeyError, IndexError):
                        continue
                return results
            else:
                raise KeyError(f"Cannot navigate to '{segment}' in path '{path}' - current value is not a dict or list")

    return current


def read_jsonl_file(file_path: str, field_path: str = None, max_samples: int = None) -> List[str]:
    """
    Read texts from a JSONL or JSONL.bz2 file.

    Args:
        file_path: Path to JSONL or JSONL.bz2 file
        field_path: Dot-separated path to text field (e.g., "document.text")
                   If None, assumes each line is a string or has a "text" field
        max_samples: Maximum number of samples to read (None = read all)

    Returns:
        List of text strings
    """
    texts = []

    # Determine if file is compressed
    is_compressed = file_path.endswith('.bz2')

    # Open file with appropriate handler
    if is_compressed:
        file_handle = bz2.open(file_path, 'rt', encoding='utf-8')
    else:
        file_handle = open(file_path, 'r', encoding='utf-8')

    try:
        for i, line in enumerate(file_handle):
            # Check max_samples limit
            if max_samples is not None and i >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {i+1}: {e}")
                continue

            # Extract text based on field_path
            try:
                if field_path:
                    result = get_nested_field(data, field_path)
                elif isinstance(data, str):
                    result = data
                elif isinstance(data, dict):
                    # Try common field names
                    if 'text' in data:
                        result = data['text']
                    elif 'content' in data:
                        result = data['content']
                    elif 'question' in data:
                        result = data['question']
                    else:
                        print(f"Warning: No obvious text field found at line {i+1}. Available fields: {list(data.keys())}")
                        continue
                else:
                    print(f"Warning: Unexpected data type at line {i+1}: {type(data)}")
                    continue

                # Handle result - could be a string or list (from array wildcard)
                if isinstance(result, list):
                    # Array wildcard was used - add all items
                    for item in result:
                        if isinstance(item, str):
                            texts.append(item)
                        else:
                            texts.append(str(item))
                elif isinstance(result, str):
                    texts.append(result)
                else:
                    texts.append(str(result))

            except KeyError as e:
                print(f"Warning: {e} at line {i+1}")
                continue
            except IndexError as e:
                print(f"Warning: {e} at line {i+1}")
                continue

    finally:
        file_handle.close()

    return texts


def generate_sample_texts(num_samples: int) -> List[str]:
    """Generate sample texts for benchmarking."""
    templates = [
        "What is the capital of {}?",
        "How does {} work in practice?",
        "Explain the concept of {} in simple terms.",
        "What are the benefits of using {}?",
        "Compare and contrast {} with alternatives.",
        "What is the history of {}?",
        "How can I implement {} effectively?",
        "What are common mistakes when using {}?",
    ]

    topics = [
        "machine learning", "artificial intelligence", "data science",
        "neural networks", "deep learning", "natural language processing",
        "computer vision", "reinforcement learning", "transformer models",
        "attention mechanisms", "gradient descent", "backpropagation",
        "convolutional networks", "recurrent networks", "embeddings",
        "tokenization", "fine-tuning", "transfer learning",
    ]

    texts = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        texts.append(template.format(topic))

    return texts


def benchmark_embedder(embedder, texts: List[str], batch_size: int, name: str) -> dict:
    """Benchmark an embedder with given batch size."""
    num_batches = (len(texts) + batch_size - 1) // batch_size
    timings = []

    print(f"\n  Testing {name} with batch_size={batch_size} ({num_batches} batches)...")

    # Warmup
    warmup_texts = texts[:min(batch_size, len(texts))]
    _ = embedder.embed(warmup_texts)

    # Benchmark with progress bar
    batch_ranges = list(range(0, len(texts), batch_size))
    pbar = tqdm(batch_ranges, desc=f"  {name}", unit="batch", leave=False)

    for i in pbar:
        batch = texts[i:i + batch_size]

        start_time = time.time()
        embeddings = embedder.embed(batch)
        elapsed = time.time() - start_time

        timings.append(elapsed)

        # Update progress bar with current throughput
        if len(timings) > 0:
            current_throughput = len(batch) / elapsed
            pbar.set_postfix({"throughput": f"{current_throughput:.1f} q/s"})

    total_time = sum(timings)
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    throughput = len(texts) / total_time
    avg_latency_per_item = total_time / len(texts) * 1000  # ms

    return {
        "name": name,
        "batch_size": batch_size,
        "num_samples": len(texts),
        "total_time": total_time,
        "avg_batch_time": avg_time,
        "std_batch_time": std_time,
        "throughput": throughput,
        "avg_latency_ms": avg_latency_per_item,
        "embedding_dim": embeddings.shape[-1]
    }


def print_results(results: List[dict]):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)

    # Group by batch size
    batch_sizes = sorted(set(r["batch_size"] for r in results))

    for batch_size in batch_sizes:
        print(f"\n{'Batch Size: ' + str(batch_size):^100}")
        print("-"*100)
        print(f"{'Method':<20} {'Total Time (s)':<18} {'Throughput (q/s)':<20} {'Avg Latency (ms)':<20} {'Speedup':<15}")
        print("-"*100)

        batch_results = [r for r in results if r["batch_size"] == batch_size]
        api_result = next((r for r in batch_results if "API" in r["name"]), None)

        for result in batch_results:
            speedup = ""
            if api_result and result["name"] != api_result["name"]:
                speedup = f"{api_result['total_time'] / result['total_time']:.2f}x"
            elif api_result and result["name"] == api_result["name"]:
                speedup = "1.00x (baseline)"

            print(f"{result['name']:<20} "
                  f"{result['total_time']:<18.3f} "
                  f"{result['throughput']:<20.2f} "
                  f"{result['avg_latency_ms']:<20.2f} "
                  f"{speedup:<15}")

    print("\n" + "="*100)

    # Summary comparison
    print("\nSUMMARY")
    print("-"*100)
    api_results = [r for r in results if "API" in r["name"]]
    gpu_results = [r for r in results if "GPU" in r["name"]]

    if api_results and gpu_results:
        avg_api_throughput = np.mean([r["throughput"] for r in api_results])
        avg_gpu_throughput = np.mean([r["throughput"] for r in gpu_results])

        print(f"Average API throughput:  {avg_api_throughput:.2f} queries/sec")
        print(f"Average GPU throughput:  {avg_gpu_throughput:.2f} queries/sec")
        print(f"GPU speedup over API:   {avg_gpu_throughput / avg_api_throughput:.2f}x")

    print("="*100 + "\n")


def create_plots(results: List[dict], output_dir: str, plot_format: str = 'png'):
    """Create visualization plots from benchmark results."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique batch sizes and method names
    batch_sizes = sorted(set(r["batch_size"] for r in results))
    methods = sorted(set(r["name"] for r in results))

    # Define colors for consistency
    colors = {'API': '#E74C3C', 'GPU': '#3498DB'}

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    # 1. Throughput Comparison (Bar Chart)
    print("\n  Creating throughput comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(batch_sizes))
    width = 0.35

    for i, method in enumerate(methods):
        method_results = [r for r in results if r["name"] == method]
        throughputs = [next((r["throughput"] for r in method_results if r["batch_size"] == bs), 0)
                      for bs in batch_sizes]
        offset = width * (i - len(methods)/2 + 0.5)
        ax.bar(x + offset, throughputs, width, label=method,
               color=colors.get(method, f'C{i}'), alpha=0.8)

    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (queries/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Embedding Generation Throughput Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    throughput_file = os.path.join(output_dir, f'throughput_comparison.{plot_format}')
    plt.savefig(throughput_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {throughput_file}")

    # 2. Latency Comparison (Bar Chart)
    print("  Creating latency comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        method_results = [r for r in results if r["name"] == method]
        latencies = [next((r["avg_latency_ms"] for r in method_results if r["batch_size"] == bs), 0)
                    for bs in batch_sizes]
        offset = width * (i - len(methods)/2 + 0.5)
        ax.bar(x + offset, latencies, width, label=method,
               color=colors.get(method, f'C{i}'), alpha=0.8)

    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Embedding Generation Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    latency_file = os.path.join(output_dir, f'latency_comparison.{plot_format}')
    plt.savefig(latency_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {latency_file}")

    # 3. Speedup Factors (Bar Chart)
    if len(methods) > 1:
        print("  Creating speedup comparison plot...")
        api_results = [r for r in results if "API" in r["name"]]
        gpu_results = [r for r in results if "GPU" in r["name"]]

        if api_results and gpu_results:
            fig, ax = plt.subplots(figsize=(10, 6))

            speedups = []
            for bs in batch_sizes:
                api_time = next((r["total_time"] for r in api_results if r["batch_size"] == bs), None)
                gpu_time = next((r["total_time"] for r in gpu_results if r["batch_size"] == bs), None)
                if api_time and gpu_time:
                    speedups.append(api_time / gpu_time)
                else:
                    speedups.append(0)

            bars = ax.bar(x, speedups, width*2, color='#27AE60', alpha=0.8)

            # Add value labels on bars
            for i, (bar, speedup) in enumerate(zip(bars, speedups)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{speedup:.2f}x', ha='center', va='bottom', fontweight='bold')

            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
            ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
            ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
            ax.set_title('GPU Speedup over API', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(batch_sizes)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            speedup_file = os.path.join(output_dir, f'speedup_comparison.{plot_format}')
            plt.savefig(speedup_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {speedup_file}")

    # 4. Scaling Plot (Line Chart)
    print("  Creating batch size scaling plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Throughput scaling
    for method in methods:
        method_results = sorted([r for r in results if r["name"] == method],
                               key=lambda x: x["batch_size"])
        bs = [r["batch_size"] for r in method_results]
        throughputs = [r["throughput"] for r in method_results]
        ax1.plot(bs, throughputs, marker='o', linewidth=2, markersize=8,
                label=method, color=colors.get(method, None))

    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (queries/sec)', fontsize=12, fontweight='bold')
    ax1.set_title('Throughput Scaling with Batch Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)

    # Latency scaling
    for method in methods:
        method_results = sorted([r for r in results if r["name"] == method],
                               key=lambda x: x["batch_size"])
        bs = [r["batch_size"] for r in method_results]
        latencies = [r["avg_latency_ms"] for r in method_results]
        ax2.plot(bs, latencies, marker='s', linewidth=2, markersize=8,
                label=method, color=colors.get(method, None))

    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Latency Scaling with Batch Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    scaling_file = os.path.join(output_dir, f'batch_scaling.{plot_format}')
    plt.savefig(scaling_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {scaling_file}")

    # 5. Summary Dashboard (Combined)
    print("  Creating summary dashboard...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Throughput
    ax1 = fig.add_subplot(gs[0, 0])
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["name"] == method]
        throughputs = [next((r["throughput"] for r in method_results if r["batch_size"] == bs), 0)
                      for bs in batch_sizes]
        offset = width * (i - len(methods)/2 + 0.5)
        ax1.bar(np.arange(len(batch_sizes)) + offset, throughputs, width,
               label=method, color=colors.get(method, f'C{i}'), alpha=0.8)
    ax1.set_xlabel('Batch Size', fontweight='bold')
    ax1.set_ylabel('Throughput (q/s)', fontweight='bold')
    ax1.set_title('Throughput Comparison', fontweight='bold')
    ax1.set_xticks(np.arange(len(batch_sizes)))
    ax1.set_xticklabels(batch_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Latency
    ax2 = fig.add_subplot(gs[0, 1])
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["name"] == method]
        latencies = [next((r["avg_latency_ms"] for r in method_results if r["batch_size"] == bs), 0)
                    for bs in batch_sizes]
        offset = width * (i - len(methods)/2 + 0.5)
        ax2.bar(np.arange(len(batch_sizes)) + offset, latencies, width,
               label=method, color=colors.get(method, f'C{i}'), alpha=0.8)
    ax2.set_xlabel('Batch Size', fontweight='bold')
    ax2.set_ylabel('Latency (ms)', fontweight='bold')
    ax2.set_title('Latency Comparison', fontweight='bold')
    ax2.set_xticks(np.arange(len(batch_sizes)))
    ax2.set_xticklabels(batch_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Throughput scaling
    ax3 = fig.add_subplot(gs[1, 0])
    for method in methods:
        method_results = sorted([r for r in results if r["name"] == method],
                               key=lambda x: x["batch_size"])
        bs = [r["batch_size"] for r in method_results]
        throughputs = [r["throughput"] for r in method_results]
        ax3.plot(bs, throughputs, marker='o', linewidth=2, markersize=6,
                label=method, color=colors.get(method, None))
    ax3.set_xlabel('Batch Size', fontweight='bold')
    ax3.set_ylabel('Throughput (q/s)', fontweight='bold')
    ax3.set_title('Throughput Scaling', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)

    # Speedup (if applicable)
    ax4 = fig.add_subplot(gs[1, 1])
    if len(methods) > 1 and api_results and gpu_results:
        speedups = []
        for bs in batch_sizes:
            api_time = next((r["total_time"] for r in api_results if r["batch_size"] == bs), None)
            gpu_time = next((r["total_time"] for r in gpu_results if r["batch_size"] == bs), None)
            if api_time and gpu_time:
                speedups.append(api_time / gpu_time)
            else:
                speedups.append(0)

        bars = ax4.bar(np.arange(len(batch_sizes)), speedups, width*2,
                      color='#27AE60', alpha=0.8)
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Batch Size', fontweight='bold')
        ax4.set_ylabel('Speedup Factor', fontweight='bold')
        ax4.set_title('GPU Speedup over API', fontweight='bold')
        ax4.set_xticks(np.arange(len(batch_sizes)))
        ax4.set_xticklabels(batch_sizes)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        # Summary statistics
        summary_text = "Benchmark Summary\n" + "="*30 + "\n\n"
        for method in methods:
            method_results = [r for r in results if r["name"] == method]
            avg_throughput = np.mean([r["throughput"] for r in method_results])
            avg_latency = np.mean([r["avg_latency_ms"] for r in method_results])
            summary_text += f"{method}:\n"
            summary_text += f"  Avg Throughput: {avg_throughput:.2f} q/s\n"
            summary_text += f"  Avg Latency: {avg_latency:.2f} ms\n\n"

        ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')

    fig.suptitle('Embedding Benchmark Results - Summary Dashboard',
                fontsize=16, fontweight='bold', y=0.995)

    dashboard_file = os.path.join(output_dir, f'summary_dashboard.{plot_format}')
    plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {dashboard_file}")

    print(f"\n✓ All plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenAI API vs GPU embeddings")
    parser.add_argument("--model_name", type=str,
                        default="ibm/slate-125m-english-rtrvr-v2",
                        help="API model name")
    parser.add_argument("--endpoint", type=str,
                        default="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/slate-125m-english-rtrvr-v2",
                        help="API endpoint URL")
    parser.add_argument("--local_model_name", type=str,
                        default="ibm-granite/granite-embedding-125m-english",
                        help="Local HuggingFace model name for GPU")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of texts to use for benchmarking. "
                             "If --input_file is provided, randomly samples this many texts from the file. "
                             "If no --input_file, generates this many sample texts. "
                             "Default: 100 for generated texts, all texts for input files.")
    parser.add_argument("--batch_sizes", type=str, default="1,8,16,32",
                        help="Comma-separated list of batch sizes to test")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to JSONL or JSONL.bz2 file containing texts to embed")
    parser.add_argument("--field_path", type=str, default=None,
                        help="Dot-separated path to text field in JSONL with array support. "
                             "Examples: 'document.text', 'documents[*].text', 'documents[0].text', 'documents[].text'. "
                             "If not specified, will try common fields: 'text', 'content', 'question'")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to read from input file (default: read all)")
    parser.add_argument("--skip_api", action="store_true",
                        help="Skip API benchmarking (useful if API is slow/unavailable)")
    parser.add_argument("--skip_gpu", action="store_true",
                        help="Skip GPU benchmarking")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for GPU model (cuda/cpu, auto-detected if not specified)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for sampling (default: 42, set to different value for different samples)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots from benchmark results")
    parser.add_argument("--plot_dir", type=str, default="benchmark_plots",
                        help="Directory to save plots (default: benchmark_plots)")
    parser.add_argument("--plot_format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Plot file format (default: png)")

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print("="*100)
    print("EMBEDDING TIMING BENCHMARK")
    print("="*100)
    print(f"Batch sizes: {batch_sizes}")
    print("="*100)

    # Load or generate sample texts
    if args.input_file:
        print(f"\nLoading texts from file: {args.input_file}")
        if args.field_path:
            print(f"  Using field path: {args.field_path}")
        if args.max_samples:
            print(f"  Max samples to read: {args.max_samples}")

        try:
            texts = read_jsonl_file(args.input_file, args.field_path, args.max_samples)
            print(f"✓ Loaded {len(texts)} texts from file")

            if not texts:
                print("✗ No texts loaded from file!")
                return

            # Show preview of first text
            preview = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
            print(f"  Preview: {preview}")

        except FileNotFoundError:
            print(f"✗ Error: File not found: {args.input_file}")
            return
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return
    else:
        # Generate sample texts
        num_to_generate = args.num_samples if args.num_samples is not None else 100
        print(f"\nGenerating {num_to_generate} sample texts...")
        texts = generate_sample_texts(num_to_generate)
        print(f"✓ Generated {len(texts)} sample texts")

    # Random sampling if num_samples is specified and we have more texts than requested
    if args.num_samples is not None and len(texts) > args.num_samples:
        print(f"\nRandomly sampling {args.num_samples} texts from {len(texts)} available texts")
        print(f"  Random seed: {args.random_seed}")
        random.seed(args.random_seed)
        texts = random.sample(texts, args.num_samples)
        print(f"✓ Sampled {len(texts)} texts")

    print(f"\nFinal dataset size: {len(texts)} texts")

    # Initialize embedders
    embedders = []

    if not args.skip_api:
        try:
            api_embedder = APIEmbedder(args.model_name, args.endpoint)
            embedders.append(("API", api_embedder))
        except Exception as e:
            print(f"✗ Failed to initialize API embedder: {e}")
            print("  Skipping API benchmarking")

    if not args.skip_gpu:
        try:
            gpu_embedder = GPUEmbedder(args.local_model_name, args.device)
            embedders.append(("GPU", gpu_embedder))
        except Exception as e:
            print(f"✗ Failed to initialize GPU embedder: {e}")
            print("  Skipping GPU benchmarking")

    if not embedders:
        print("\n✗ No embedders available for benchmarking!")
        return

    # Run benchmarks
    results = []

    for batch_size in batch_sizes:
        print(f"\n{'='*100}")
        print(f"BENCHMARKING WITH BATCH SIZE: {batch_size}")
        print(f"{'='*100}")

        for name, embedder in embedders:
            try:
                result = benchmark_embedder(embedder, texts, batch_size, name)
                results.append(result)
            except Exception as e:
                print(f"  ✗ Error benchmarking {name} with batch_size={batch_size}: {e}")

    # Print results
    if results:
        print_results(results)

        # Generate plots if requested
        if args.plot:
            print("\n" + "="*100)
            print("GENERATING PLOTS")
            print("="*100)
            create_plots(results, args.plot_dir, args.plot_format)
    else:
        print("\n✗ No benchmark results available!")


if __name__ == "__main__":
    main()
