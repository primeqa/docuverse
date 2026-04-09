#!/usr/bin/env python3
"""
Compute vocabulary token usage statistics for a SentenceTransformer model
over a directory of gzipped JSONL files.

The 'text' field is read from each JSON record. Statistics include:
  - Per-document token count distribution (min, max, mean, median, percentiles)
  - Token frequency across the corpus
  - Vocabulary coverage (% of model vocab seen)
  - Most / least frequent tokens
  - Truncation rate (documents exceeding the model's max_seq_length)

Usage:
    python token_stats_from_jsonl.py \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --data_dir /path/to/jsonl_gz_files \\
        [--text_field text] \\
        [--workers 8] \\
        [--top_k 50] \\
        [--output stats.json]
"""

import argparse
import gzip
import multiprocessing
import os

import orjson
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from docuverse.utils.jsonl_utils import get_nested_field


# ---------------------------------------------------------------------------
# Per-process worker (module-level so it can be pickled)
# ---------------------------------------------------------------------------

_tokenizer = None
_worker_pos = None


def _init_worker(model_name_or_path: str, pos_counter: multiprocessing.Value):
    """Initialise the tokenizer once per worker process and claim a tqdm slot."""
    global _tokenizer, _worker_pos
    from transformers import AutoTokenizer
    with pos_counter.get_lock():
        _worker_pos = pos_counter.value
        pos_counter.value += 1
    _tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fix_mistral_regex=True)
    _tokenizer.model_max_length = int(1e9)  # suppress "sequence longer than max_length" warnings


def _process_file(args: Tuple[str, List[str]]) -> Tuple[List[int], Counter]:
    """
    Read one gzipped JSONL file, tokenize every 'text' field and return:
      - list of token counts per document
      - Counter of token ids
    """
    filepath, text_fields = args
    token_counts: List[int] = []
    freq: Counter = Counter()

    open_fn = gzip.open if filepath.endswith(".gz") else open
    desc = Path(filepath).name[:40]

    with open_fn(filepath, "rt", encoding="utf-8", errors="replace") as fh:
        with tqdm(fh, desc=desc, position=_worker_pos, leave=False,
                  unit="line", dynamic_ncols=True) as pbar:
            for line in pbar:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                for text_field in text_fields:
                    try:
                        value = get_nested_field(record, text_field)
                    except (KeyError, IndexError):
                        continue
                    texts = value if isinstance(value, list) else [value]
                    for text in texts:
                        if not text:
                            continue
                        ids = _tokenizer.encode(str(text), add_special_tokens=True)
                        token_counts.append(len(ids))
                        freq.update(ids)

    return token_counts, freq


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def percentile_stats(counts: np.ndarray) -> dict:
    return {
        "min": int(counts.min()),
        "p05": float(np.percentile(counts, 5)),
        "p25": float(np.percentile(counts, 25)),
        "median": float(np.median(counts)),
        "mean": float(counts.mean()),
        "p75": float(np.percentile(counts, 75)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(counts.max()),
        "std": float(counts.std()),
    }


def token_id_to_str(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.convert_ids_to_tokens([token_id])[0]
    except Exception:
        return f"<id:{token_id}>"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="SentenceTransformer model name or local path")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing *.jsonl.gz (or *.jsonl) files")
    parser.add_argument("--text_field", default=["text"], nargs="+",
                        help="One or more field paths to extract text from (default: text). "
                             "Supports dot notation and array wildcards per get_nested_field "
                             "(e.g. --text_field positives[*] negatives[*])")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="Number of parallel worker processes (default: min(8, cpu_count))")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Number of most/least-frequent tokens to report; -1 = full histogram (default: -1)")
    parser.add_argument("--output", default="token_stats.json",
                        help="Path to write JSON results to (default: token_stats.json)")
    parser.add_argument("--glob", default="**/*.jsonl*",
                        help="Glob pattern for finding files (default: **/*.jsonl*)")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Max sequence length for truncation-rate calculation (default: 8192)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"ERROR: {data_dir} is not a directory")

    files = sorted(data_dir.glob(args.glob))
    files = [str(f) for f in files if f.suffix in {".gz", ".jsonl"}
             or str(f).endswith(".jsonl.gz")]
    if not files:
        sys.exit(f"ERROR: no .jsonl or .jsonl.gz files found under {data_dir}")

    print(f"Found {len(files)} file(s) under {data_dir}")
    print(f"Model : {args.model}")
    print(f"Workers: {args.workers}")

    # ------------------------------------------------------------------
    # Load tokenizer in the main process to get vocab metadata
    # (no full model load = no CUDA init, safe to fork)
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer
    print("Loading tokenizer in main process for metadata…")
    main_tokenizer = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=True)
    vocab_size = main_tokenizer.vocab_size
    max_seq_length = args.max_seq_length
    print(f"  max_seq_length : {max_seq_length}")
    print(f"  vocab_size     : {vocab_size}")

    # ------------------------------------------------------------------
    # Parallel tokenisation
    # ------------------------------------------------------------------
    all_counts: List[int] = []
    global_freq: Counter = Counter()

    work_items = [(f, args.text_field) for f in files]

    n_workers = min(args.workers, len(files))
    pos_counter = multiprocessing.Value('i', 0)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(args.model, pos_counter),
    ) as pool:
        futures = {pool.submit(_process_file, item): item[0] for item in work_items}
        with tqdm(total=len(futures), desc="Files done", unit="file",
                  position=n_workers, dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                filepath = futures[future]
                try:
                    counts, freq = future.result()
                    all_counts.extend(counts)
                    global_freq.update(freq)
                except Exception as exc:
                    tqdm.write(f"WARNING: {filepath} raised {exc}", file=sys.stderr)
                finally:
                    pbar.update(1)

    if not all_counts:
        sys.exit("ERROR: no documents were processed — check --text_field and file contents")

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    counts_arr = np.array(all_counts, dtype=np.int64)
    doc_stats = percentile_stats(counts_arr)

    n_docs = len(all_counts)
    n_truncated = int((counts_arr > max_seq_length).sum())
    total_tokens = int(counts_arr.sum())

    unique_tokens_seen = len(global_freq)
    vocab_coverage_pct = 100.0 * unique_tokens_seen / vocab_size if vocab_size > 0 else 0.0

    # Top-k most and least frequent (excluding special tokens if possible)
    special_ids = set(
        v for v in vars(main_tokenizer).get("all_special_ids", [])
    )
    filtered_freq = {tid: cnt for tid, cnt in global_freq.items()
                     if tid not in special_ids}

    full_histogram = args.top_k == -1
    effective_k = None if full_histogram else args.top_k

    freq_counter = Counter(filtered_freq)
    top_k_most = freq_counter.most_common(effective_k)
    top_k_least = [] if full_histogram else freq_counter.most_common()[:-effective_k - 1:-1]

    def freq_table(items):
        return [
            {"token": token_id_to_str(main_tokenizer, tid),
             "token_id": tid,
             "count": cnt}
            for tid, cnt in items
        ]

    results = {
        "model": args.model,
        "max_seq_length": max_seq_length,
        "vocab_size": vocab_size,
        "data_dir": str(data_dir),
        "files_processed": len(files),
        "documents_processed": n_docs,
        "total_tokens": total_tokens,
        "unique_tokens_seen": unique_tokens_seen,
        "vocab_coverage_pct": round(vocab_coverage_pct, 4),
        "truncated_documents": n_truncated,
        "truncation_rate_pct": round(100.0 * n_truncated / n_docs, 4),
        "token_count_per_doc": doc_stats,
    }
    if full_histogram:
        results["histogram"] = freq_table(top_k_most)
    else:
        results[f"top_{args.top_k}_most_frequent"] = freq_table(top_k_most)
        results[f"top_{args.top_k}_least_frequent"] = freq_table(top_k_least)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n=== Token Statistics ===")
    print(f"Documents processed  : {n_docs:,}")
    print(f"Total tokens         : {total_tokens:,}")
    print(f"Unique tokens seen   : {unique_tokens_seen:,}  /  {vocab_size:,}  ({vocab_coverage_pct:.2f}% vocab coverage)")
    print(f"Truncated docs       : {n_truncated:,}  ({results['truncation_rate_pct']:.2f}%)")
    print("\nTokens per document:")
    for k, v in doc_stats.items():
        print(f"  {k:>8}: {v:.1f}" if isinstance(v, float) else f"  {k:>8}: {v}")
    freq_label = "histogram" if full_histogram else f"top_{args.top_k}_most_frequent"
    preview = results[freq_label][:10]
    print(f"\n{'Full histogram' if full_histogram else f'Top {args.top_k}'} most frequent tokens (showing first 10):")
    for entry in preview:
        print(f"  {entry['token']:20s}  {entry['count']:>12,}")
    remaining = len(results[freq_label]) - 10
    if remaining > 0:
        print(f"  … ({remaining:,} more in output)")

    # ------------------------------------------------------------------
    # Optional JSON output
    # ------------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()