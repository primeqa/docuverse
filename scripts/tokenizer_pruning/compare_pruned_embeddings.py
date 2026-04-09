#!/usr/bin/env python3
"""
Compare embeddings between an original and a pruned SentenceTransformer model.

Reads JSONL files in streaming chunks, encodes each chunk with both models,
computes per-document cosine similarity, and reports aggregate statistics.
Only one chunk's embeddings are ever in memory, so this scales to arbitrarily
large corpora.

Usage:
    python compare_pruned_embeddings.py \\
        --original /path/to/original-model \\
        --pruned   /path/to/pruned-model \\
        --data_dir /path/to/jsonl_files \\
        [--text_field text] \\
        [--glob "**/*.jsonl*"] \\
        [--batch_size 64] \\
        [--chunk_size 10000] \\
        [--max_docs 0] \\
        [--dtype bfloat16] \\
        [--attn_implementation flash_attention_2] \\
        [--torch_compile] \\
        [--output comparison.json]
"""

import argparse
import gzip
import sys
import time
from pathlib import Path
from typing import Iterator, List

import numpy as np
import orjson
import torch
from tqdm import tqdm

from docuverse.utils.jsonl_utils import get_nested_field


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device: str,
    dtype: torch.dtype | None,
    attn_implementation: str | None,
    compile_model: bool,
):
    """Load a SentenceTransformer with the requested configuration."""
    from sentence_transformers import SentenceTransformer

    model_kwargs = {}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = SentenceTransformer(
        model_path, device=device, model_kwargs=model_kwargs,
        **({"dtype": dtype} if dtype is not None else {}),
    )

    if compile_model:
        model[0].auto_model = torch.compile(model[0].auto_model)

    return model


# ---------------------------------------------------------------------------
# Streaming text chunks
# ---------------------------------------------------------------------------

def find_files(data_dir: Path, glob_pattern: str) -> List[Path]:
    """Return sorted list of JSONL / .jsonl.gz files matching *glob_pattern*."""
    files = sorted(data_dir.glob(glob_pattern))
    files = [
        f for f in files
        if f.suffix in {".gz", ".jsonl"} or str(f).endswith(".jsonl.gz")
    ]
    return files


def iter_text_chunks(
    data_dir: Path,
    glob_pattern: str,
    text_fields: List[str],
    max_docs: int,
    chunk_size: int,
) -> Iterator[List[str]]:
    """Yield lists of up to *chunk_size* texts from JSONL files.

    Respects *max_docs* (0 = unlimited).
    """
    files = find_files(data_dir, glob_pattern)
    if not files:
        sys.exit(f"ERROR: no matching files under {data_dir}")
    print(f"  Found {len(files)} file(s)")

    chunk: List[str] = []
    total = 0

    for filepath in files:
        open_fn = gzip.open if str(filepath).endswith(".gz") else open
        with open_fn(filepath, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
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
                    items = value if isinstance(value, list) else [value]
                    for text in items:
                        if not text:
                            continue
                        chunk.append(str(text))
                        total += 1
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                        if 0 < max_docs <= total:
                            if chunk:
                                yield chunk
                            return
    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cosine_similarity_paired(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two L2-normalised (N, D) matrices."""
    return np.sum(a * b, axis=1)


def percentile_stats(values: np.ndarray) -> dict:
    return {
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "std": float(values.std()),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(similarities: np.ndarray):
    stats = percentile_stats(similarities)

    print("\n=== Cosine Similarity (original vs pruned) ===")
    print(f"  Documents:  {len(similarities):,}")
    for key, val in stats.items():
        print(f"  {key:>8}: {val:.6f}")

    thresholds = [0.99, 0.95, 0.90, 0.80]
    print("\n  Distribution:")
    prev = 1.01  # above 1.0 so first bucket is ">= threshold"
    for t in thresholds:
        count = int(((similarities >= t) & (similarities < prev)).sum())
        pct = 100.0 * count / len(similarities)
        label = f">= {t}" if prev > 1.0 else f"[{t}, {prev})"
        print(f"    {label:>12}: {count:>7,}  ({pct:5.1f}%)")
        prev = t
    count_low = int((similarities < thresholds[-1]).sum())
    pct_low = 100.0 * count_low / len(similarities)
    print(f"    {'< ' + str(thresholds[-1]):>12}: {count_low:>7,}  ({pct_low:5.1f}%)")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Models
    parser.add_argument(
        "--original", required=True,
        help="Path to the original (unpruned) model",
    )
    parser.add_argument(
        "--pruned", required=True,
        help="Path to the pruned model",
    )

    # Data
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing JSONL (or .jsonl.gz) files",
    )
    parser.add_argument(
        "--text_field", default=["text"], nargs="+",
        help="JSON field paths to extract text from (default: text). "
             "Supports dot notation and array wildcards "
             "(e.g. --text_field contexts[*].text)",
    )
    parser.add_argument(
        "--glob", default="**/*.jsonl*",
        help="Glob pattern for finding files (default: **/*.jsonl*)",
    )

    # Performance
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Encoding batch size per model (default: 64)",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=10000,
        help="Number of texts to process per streaming chunk (default: 10000). "
             "Only one chunk's embeddings are in memory at a time.",
    )
    parser.add_argument(
        "--max_docs", type=int, default=0,
        help="Maximum documents to process; 0 = unlimited (default: 0)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (default: auto-detect cuda/cpu)",
    )

    # Model configuration
    parser.add_argument(
        "--dtype", default=None, choices=list(_DTYPE_MAP.keys()),
        help="Model dtype (default: model's native dtype)",
    )
    parser.add_argument(
        "--attn_implementation", default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation (default: model default)",
    )
    parser.add_argument(
        "--torch_compile", action="store_true",
        help="torch.compile the underlying transformer model",
    )

    # Output
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write JSON results to",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"ERROR: {data_dir} is not a directory")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = _DTYPE_MAP[args.dtype] if args.dtype else None

    # ------------------------------------------------------------------
    # Load both models
    # ------------------------------------------------------------------
    print(f"Loading original model: {args.original}")
    t0 = time.time()
    model_original = load_model(
        args.original, device, dtype, args.attn_implementation, args.torch_compile,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print(f"Loading pruned model: {args.pruned}")
    t0 = time.time()
    model_pruned = load_model(
        args.pruned, device, dtype, args.attn_implementation, args.torch_compile,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Stream texts and compare embeddings chunk by chunk
    # ------------------------------------------------------------------
    print(f"\nReading texts from {data_dir} (glob: {args.glob})")
    all_similarities: List[np.ndarray] = []
    n_docs = 0
    pbar = tqdm(desc="Documents", unit="doc", dynamic_ncols=True)

    for chunk in iter_text_chunks(
        data_dir, args.glob, args.text_field, args.max_docs, args.chunk_size,
    ):
        emb_orig = model_original.encode(
            chunk, batch_size=args.batch_size,
            show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        )
        emb_pruned = model_pruned.encode(
            chunk, batch_size=args.batch_size,
            show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        )
        sims = cosine_similarity_paired(emb_orig, emb_pruned)
        all_similarities.append(sims)
        n_docs += len(chunk)
        pbar.update(len(chunk))
        pbar.set_postfix(mean_cos=f"{sims.mean():.4f}")

    pbar.close()

    if n_docs == 0:
        sys.exit("ERROR: no texts found — check --text_field and file contents")

    # ------------------------------------------------------------------
    # Aggregate and report
    # ------------------------------------------------------------------
    similarities = np.concatenate(all_similarities)
    stats = print_report(similarities)

    # ------------------------------------------------------------------
    # Optional JSON output
    # ------------------------------------------------------------------
    if args.output:
        results = {
            "original_model": args.original,
            "pruned_model": args.pruned,
            "data_dir": str(data_dir),
            "documents": n_docs,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "torch_compile": args.torch_compile,
            "cosine_similarity": stats,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
