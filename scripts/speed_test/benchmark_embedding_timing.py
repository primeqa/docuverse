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

    # Multiple input files (per-file results + aggregate statistics):
    python benchmark_embedding_timing.py \
        --input_file queries.jsonl documents.jsonl.bz2 \
        --field_path text --batch_sizes 1,16,32
"""

import argparse
import time
import os
import sys
import random
import numpy as np
from typing import List
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from tqdm import tqdm

# Add parent directory to path to import jsonl_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from docuverse.utils.jsonl_utils import read_jsonl_file


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

    DTYPE_MAP = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    def __init__(self, model_name: str, device: str = None, use_torch_compile: bool = False,
                 dtype: str = "bf16"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"Loading model on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        torch_dtype = self.DTYPE_MAP.get(dtype, torch.bfloat16)
        print(f"  Using dtype: {dtype}")
        model_kwargs = {"torch_dtype": torch_dtype}
        try:
            self.model = AutoModel.from_pretrained(
                model_name, attn_implementation="flash_attention_2", **model_kwargs
            ).to(device)
            print(f"  Using flash_attention_2 with bf16")
        except (ValueError, ImportError):
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)
            print(f"  Flash attention not supported, using default attention with bf16")

        self.model.eval()
        if use_torch_compile:
            self.model = torch.compile(self.model)
            print(f"  Using torch.compile")
        print(f"✓ Initialized GPU embedder: {model_name} on {device}")

    def tokenize(self, texts: List[str]):
        """Tokenize texts and move to device."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

    def encode(self, inputs) -> np.ndarray:
        """Run transformer inference on pre-tokenized inputs."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local GPU (tokenize + encode)."""
        inputs = self.tokenize(texts)
        return self.encode(inputs)

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# The read_jsonl_file and get_nested_field functions have been moved to jsonl_utils.py
# They are imported at the top of this file


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


def warmup_embedder(embedder, texts: List[str], batch_size: int, name: str,
                    warmup_batches: int = 1):
    """Run warmup batches for an embedder to stabilize GPU/torch.compile performance."""
    if warmup_batches <= 0:
        return
    print(f"  Warming up {name} with {warmup_batches} batch(es) (batch_size={batch_size})...")
    for wi in range(warmup_batches):
        warmup_start = wi * batch_size
        warmup_texts = texts[warmup_start % len(texts):(warmup_start % len(texts)) + batch_size]
        if not warmup_texts:
            warmup_texts = texts[:min(batch_size, len(texts))]
        _ = embedder.embed(warmup_texts)
    print(f"  Warmup complete for {name}.")


def benchmark_embedder(embedder, texts: List[str], batch_size: int, name: str) -> dict:
    """Benchmark an embedder with given batch size."""
    num_batches = (len(texts) + batch_size - 1) // batch_size
    timings = []
    tokenize_timings = []
    encode_timings = []
    has_split = hasattr(embedder, 'tokenize') and hasattr(embedder, 'encode')

    print(f"\n  Testing {name} with batch_size={batch_size} ({num_batches} batches)...")

    # Benchmark with progress bar
    batch_ranges = list(range(0, len(texts), batch_size))
    pbar = tqdm(batch_ranges, desc=f"  {name}", unit="batch", leave=False)

    for i in pbar:
        batch = texts[i:i + batch_size]

        if has_split:
            tok_start = time.time()
            inputs = embedder.tokenize(batch)
            tok_elapsed = time.time() - tok_start

            enc_start = time.time()
            embeddings = embedder.encode(inputs)
            enc_elapsed = time.time() - enc_start

            elapsed = tok_elapsed + enc_elapsed
            tokenize_timings.append(tok_elapsed)
            encode_timings.append(enc_elapsed)
        else:
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

    result = {
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

    if has_split and tokenize_timings:
        total_tokenize = sum(tokenize_timings)
        total_encode = sum(encode_timings)
        result.update({
            "tokenize_time": total_tokenize,
            "encode_time": total_encode,
            "tokenize_pct": total_tokenize / total_time * 100,
            "encode_pct": total_encode / total_time * 100,
            "avg_tokenize_batch": np.mean(tokenize_timings),
            "avg_encode_batch": np.mean(encode_timings),
        })

    return result


def print_results(all_file_results: dict):
    """Print all benchmark results in a unified table, grouped by batch size, method, and file."""
    all_results = []
    for results in all_file_results.values():
        all_results.extend(results)

    if not all_results:
        print("\nNo benchmark results available!")
        return

    batch_sizes = sorted(set(r["batch_size"] for r in all_results))
    file_labels = list(all_file_results.keys())
    methods = sorted(set(r["name"] for r in all_results))
    multiple_files = len(file_labels) > 1

    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)

    for batch_size in batch_sizes:
        print(f"\n{'Batch Size: ' + str(batch_size):^120}")
        print("-" * 120)

        if multiple_files:
            print(f"{'File':<30} {'Method':<10} {'Samples':<10} {'Total (s)':<12} "
                  f"{'Throughput (q/s)':<20} {'Avg Latency (ms)':<20} {'Std Batch (s)':<15}")
        else:
            print(f"{'Method':<15} {'Samples':<10} {'Total (s)':<12} "
                  f"{'Throughput (q/s)':<20} {'Avg Latency (ms)':<20} {'Std Batch (s)':<15}")
        print("-" * 120)

        for file_label in file_labels:
            file_results = [r for r in all_file_results[file_label]
                            if r["batch_size"] == batch_size]
            for result in file_results:
                if multiple_files:
                    print(f"{file_label:<30} {result['name']:<10} "
                          f"{result['num_samples']:<10} {result['total_time']:<12.3f} "
                          f"{result['throughput']:<20.2f} {result['avg_latency_ms']:<20.2f} "
                          f"{result['std_batch_time']:<15.4f}")
                else:
                    print(f"{result['name']:<15} "
                          f"{result['num_samples']:<10} {result['total_time']:<12.3f} "
                          f"{result['throughput']:<20.2f} {result['avg_latency_ms']:<20.2f} "
                          f"{result['std_batch_time']:<15.4f}")

    # Aggregate across files (shown only when multiple files)
    if multiple_files:
        print("\n" + "=" * 120)
        print("AGGREGATE ACROSS FILES")
        print("=" * 120)

        for batch_size in batch_sizes:
            print(f"\n{'Batch Size: ' + str(batch_size):^120}")
            print("-" * 120)
            print(f"{'Method':<15} {'Files':<8} {'Avg Throughput (q/s)':<22} "
                  f"{'Std Throughput':<18} {'Avg Latency (ms)':<20} {'Std Latency (ms)':<18}")
            print("-" * 120)

            for method in methods:
                entries = [r for r in all_results
                           if r["batch_size"] == batch_size and r["name"] == method]
                if not entries:
                    continue
                throughputs = [e["throughput"] for e in entries]
                latencies = [e["avg_latency_ms"] for e in entries]
                print(f"{method:<15} {len(entries):<8} "
                      f"{np.mean(throughputs):<22.2f} {np.std(throughputs):<18.2f} "
                      f"{np.mean(latencies):<20.2f} {np.std(latencies):<18.2f}")

    # Timing breakdown (tokenization vs inference)
    has_breakdown = any("tokenize_time" in r for r in all_results)
    if has_breakdown:
        print("\n" + "=" * 120)
        print("TIMING BREAKDOWN (Tokenization vs Inference)")
        print("=" * 120)

        for batch_size in batch_sizes:
            print(f"\n{'Batch Size: ' + str(batch_size):^120}")
            print("-" * 120)

            if multiple_files:
                print(f"{'File':<30} {'Method':<10} {'Tokenize (s)':<15} {'Inference (s)':<15} "
                      f"{'Tok %':<10} {'Inf %':<10} {'Avg Tok/bat (ms)':<20} {'Avg Inf/bat (ms)':<20}")
            else:
                print(f"{'Method':<15} {'Tokenize (s)':<15} {'Inference (s)':<15} "
                      f"{'Tok %':<10} {'Inf %':<10} {'Avg Tok/bat (ms)':<20} {'Avg Inf/bat (ms)':<20}")
            print("-" * 120)

            for file_label in file_labels:
                file_results = [r for r in all_file_results[file_label]
                                if r["batch_size"] == batch_size and "tokenize_time" in r]
                for result in file_results:
                    if multiple_files:
                        print(f"{file_label:<30} {result['name']:<10} "
                              f"{result['tokenize_time']:<15.3f} {result['encode_time']:<15.3f} "
                              f"{result['tokenize_pct']:<10.1f} {result['encode_pct']:<10.1f} "
                              f"{result['avg_tokenize_batch']*1000:<20.2f} {result['avg_encode_batch']*1000:<20.2f}")
                    else:
                        print(f"{result['name']:<15} "
                              f"{result['tokenize_time']:<15.3f} {result['encode_time']:<15.3f} "
                              f"{result['tokenize_pct']:<10.1f} {result['encode_pct']:<10.1f} "
                              f"{result['avg_tokenize_batch']*1000:<20.2f} {result['avg_encode_batch']*1000:<20.2f}")

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("-" * 120)
    for method in methods:
        method_results = [r for r in all_results if r["name"] == method]
        avg_throughput = np.mean([r["throughput"] for r in method_results])
        avg_latency = np.mean([r["avg_latency_ms"] for r in method_results])
        print(f"  {method}: Avg throughput = {avg_throughput:.2f} q/s, "
              f"Avg latency = {avg_latency:.2f} ms")

    api_results = [r for r in all_results if "API" in r["name"]]
    gpu_results = [r for r in all_results if "GPU" in r["name"]]
    if api_results and gpu_results:
        avg_api = np.mean([r["throughput"] for r in api_results])
        avg_gpu = np.mean([r["throughput"] for r in gpu_results])
        print(f"  GPU speedup over API: {avg_gpu / avg_api:.2f}x")

    print("=" * 120 + "\n")


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


def compute_short_labels(file_paths: List[str]) -> List[str]:
    """Compute short labels by removing common prefix and suffix from file paths.

    For inputs like:
        benchmark/miracl/ar/train.jsonl.bz2
        benchmark/miracl/bn/train.jsonl.bz2
    Returns: ['ar', 'bn']
    """
    if len(file_paths) <= 1:
        return [os.path.basename(f) for f in file_paths]

    # Strip common directory prefix
    common_dir = os.path.commonpath(file_paths)
    if os.path.isfile(common_dir):
        common_dir = os.path.dirname(common_dir)
    stripped = [os.path.relpath(f, common_dir) for f in file_paths]

    # Strip common suffix (e.g. /train.jsonl.bz2)
    reversed_strs = [s[::-1] for s in stripped]
    common_suffix = os.path.commonprefix(reversed_strs)[::-1]
    if common_suffix:
        suffix_len = len(common_suffix)
        labels = [s[:-suffix_len].strip(os.sep).strip('/') for s in stripped]
        if all(labels):
            return labels

    return stripped


def load_texts_from_file(input_file: str, field_path: str = None,
                         max_samples: int = None) -> List[str]:
    """Load texts from a JSONL or JSONL.bz2 file.

    Returns a list of text strings, or raises on error.
    """
    print(f"\nLoading texts from file: {input_file}")
    if field_path:
        print(f"  Using field path: {field_path}")
    if max_samples:
        print(f"  Max samples to read: {max_samples}")

    texts = read_jsonl_file(
        input_file,
        field_path=field_path,
        max_samples=max_samples,
        verbose=True
    )
    print(f"  Loaded {len(texts)} texts from file")

    if texts:
        preview = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
        print(f"  Preview: {preview}")

    return texts


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
    parser.add_argument("--input_file", type=str, nargs="+", default=None,
                        help="Path(s) to JSONL or JSONL.bz2 file(s) containing texts to embed. "
                             "When multiple files are given, benchmarks run per-file with aggregate stats.")
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
    parser.add_argument("--warmup_batches", type=int, default=1,
                        help="Number of warmup batches to run before timing (default: 1)")
    parser.add_argument("--max_text_size", type=int, default=1536,
                        help="The maximum size for the text to encode")
    parser.add_argument("--torch_compile", action="store_true",
                        help="Enable torch.compile optimization for GPU model (default: disabled)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"],
                        help="Model weight dtype for GPU embedder (default: bf16)")

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print("="*100)
    print("EMBEDDING TIMING BENCHMARK")
    print("="*100)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Warmup batches: {args.warmup_batches}")
    print("="*100)

    # Build list of input sources: each is (label, file_path_or_None)
    input_sources = []
    if args.input_file:
        labels = compute_short_labels(args.input_file)
        for label, f in zip(labels, args.input_file):
            input_sources.append((label, f))
    else:
        input_sources.append(("generated", None))

    multiple_files = len(input_sources) > 1

    # Initialize embedders (once, shared across all files)
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
            gpu_embedder = GPUEmbedder(args.local_model_name, args.device, args.torch_compile, args.dtype)
            embedders.append(("GPU", gpu_embedder))
        except Exception as e:
            print(f"✗ Failed to initialize GPU embedder: {e}")
            print("  Skipping GPU benchmarking")

    if not embedders:
        print("\n✗ No embedders available for benchmarking!")
        return

    gpu_embedder_obj = next((emb for name, emb in embedders if name == "GPU"), None)

    # One-time warmup for all embedders
    if args.warmup_batches > 0:
        print(f"\n{'='*100}")
        print("WARMUP PHASE")
        print(f"{'='*100}")
        warmup_batch_size = max(batch_sizes)
        warmup_count = warmup_batch_size * args.warmup_batches

        if args.input_file:
            # Sample warmup texts from the first input file
            try:
                warmup_texts = load_texts_from_file(
                    args.input_file[0],
                    field_path=args.field_path,
                    max_samples=warmup_count
                )
                if not warmup_texts:
                    print("  No texts loaded from first file for warmup, using generated texts.")
                    warmup_texts = generate_sample_texts(warmup_count)
                elif len(warmup_texts) > warmup_count:
                    random.seed(args.random_seed)
                    warmup_texts = random.sample(warmup_texts, warmup_count)
            except Exception as e:
                print(f"  Failed to load warmup texts from first file: {e}")
                warmup_texts = generate_sample_texts(warmup_count)
        else:
            warmup_texts = generate_sample_texts(warmup_count)

        for name, embedder in embedders:
            warmup_embedder(embedder, warmup_texts, warmup_batch_size, name,
                            warmup_batches=args.warmup_batches)

    # Iterate over input files
    all_file_results = {}  # file_label -> list of result dicts

    for file_idx, (file_label, file_path) in enumerate(input_sources):
        if multiple_files:
            print(f"\n{'#'*100}")
            print(f"FILE {file_idx + 1}/{len(input_sources)}: {file_label}")
            print(f"{'#'*100}")

        # Load or generate texts
        if file_path is not None:
            try:
                texts = load_texts_from_file(
                    file_path,
                    field_path=args.field_path,
                    max_samples=args.max_samples
                )
                if not texts:
                    print(f"✗ No texts loaded from {file_label}, skipping.")
                    continue
            except FileNotFoundError:
                print(f"✗ Error: File not found: {file_path}, skipping.")
                continue
            except Exception as e:
                print(f"✗ Error loading file {file_path}: {e}, skipping.")
                continue
        else:
            num_to_generate = args.num_samples if args.num_samples is not None else 100
            print(f"\nGenerating {num_to_generate} sample texts...")
            texts = generate_sample_texts(num_to_generate)
            print(f"  Generated {len(texts)} sample texts")

        # Random sampling if num_samples is specified and we have more texts than requested
        if args.num_samples is not None and len(texts) > args.num_samples:
            print(f"\nRandomly sampling {args.num_samples} texts from {len(texts)} available texts")
            print(f"  Random seed: {args.random_seed}")
            random.seed(args.random_seed)
            texts = random.sample(texts, args.num_samples)
            print(f"  Sampled {len(texts)} texts")

        # Prune texts that exceed max_text_size (requires GPU tokenizer)
        if gpu_embedder_obj is not None:
            new_t = []
            eliminated = 0
            for t in texts:
                toks = gpu_embedder_obj.tokenizer.tokenize(t)
                if len(toks) <= args.max_text_size:
                    new_t.append(t)
                else:
                    eliminated += 1
            texts = new_t
            if eliminated > 0:
                print(f"  Eliminated {eliminated} texts exceeding {args.max_text_size} tokens, "
                      f"{len(texts)} remaining.")

        print(f"\nFinal dataset size: {len(texts)} texts")

        if not texts:
            print(f"✗ No texts remaining for {file_label} after filtering, skipping.")
            continue

        # Run benchmarks for this file
        file_results = []

        for batch_size in batch_sizes:
            for name, embedder in embedders:
                try:
                    result = benchmark_embedder(embedder, texts, batch_size, name)
                    result["source_file"] = file_label
                    file_results.append(result)
                except Exception as e:
                    print(f"  ✗ Error benchmarking {name} with batch_size={batch_size}: {e}")

        if file_results:
            all_file_results[file_label] = file_results
        else:
            print(f"\n✗ No benchmark results for {file_label}!")

    # Print all results at the end
    if all_file_results:
        print_results(all_file_results)

        # Generate plots
        if args.plot:
            print("\n" + "="*100)
            print("GENERATING PLOTS")
            print("="*100)
            for file_label, file_results in all_file_results.items():
                plot_dir = args.plot_dir
                if multiple_files:
                    plot_dir = os.path.join(args.plot_dir, file_label.replace('.', '_'))
                create_plots(file_results, plot_dir, args.plot_format)
    else:
        print("\n✗ No benchmark results available!")


if __name__ == "__main__":
    main()
