#!/usr/bin/env python3
"""Encode a text corpus into a .npy file for isotropy analysis.

Reads text from a TSV or JSONL file, encodes it with a sentence-transformers
model, and saves the resulting float32 embeddings as a .npy file suitable for:

    pytest tests/test_embedding_isotropy.py::test_real_embeddings_are_isotropic \\
        --embeddings-npy embeddings.npy -s -m slow

Usage:
    python scripts/embed_to_npy.py passages.tsv -o embeddings.npy
    python scripts/embed_to_npy.py passages.tsv -o embeddings.npy \\
        --model ibm-granite/granite-embedding-278m-multilingual-r2 \\
        --text-col 2 --max-rows 5000 --batch-size 64
    python scripts/embed_to_npy.py corpus.jsonl -o embeddings.npy --text-field text

TSV input:
    Assumes tab-separated; --text-col (0-based) selects the text column.
    The first row is treated as a header if it does not look like a record.

JSONL input:
    Reads one JSON object per line; --text-field selects the key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _read_tsv(path: Path, text_col: int, max_rows: int | None) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and len(texts) >= max_rows:
                break
            parts = line.rstrip("\n").split("\t")
            if i == 0 and not parts[text_col].strip():
                continue  # skip empty header
            if text_col >= len(parts):
                continue
            text = parts[text_col].strip()
            if text:
                texts.append(text)
    return texts


def _read_jsonl(path: Path, text_field: str, max_rows: int | None) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_rows is not None and len(texts) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field, "").strip()
            if text:
                texts.append(text)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode a corpus to .npy embeddings for isotropy testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="TSV or JSONL corpus file.")
    parser.add_argument("-o", "--output", type=Path, default=Path("embeddings.npy"),
                        help="Output .npy file path.")
    parser.add_argument("--model", default="ibm-granite/granite-embedding-278m-multilingual-r2",
                        help="Sentence-transformers model name or local path.")
    parser.add_argument("--text-col", type=int, default=2,
                        help="0-based column index for text in TSV input.")
    parser.add_argument("--text-field", default="text",
                        help="JSON key for text in JSONL input.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Encode at most this many rows (useful for quick checks).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size.")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="L2-normalise embeddings before saving (default: on).")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="Save raw embeddings without L2 normalisation.")
    args = parser.parse_args()

    # Read texts
    suffix = args.input.suffix.lower()
    if suffix in (".jsonl", ".json"):
        texts = _read_jsonl(args.input, args.text_field, args.max_rows)
    else:
        texts = _read_tsv(args.input, args.text_col, args.max_rows)

    if not texts:
        print(f"ERROR: no texts found in {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Read {len(texts)} texts from {args.input}")

    # Encode
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers is not installed.\n"
              "  pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"Encoding with batch_size={args.batch_size} …")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    if args.normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.clip(norms, 1e-12, None)

    print(f"Embeddings shape: {embeddings.shape}  dtype: {embeddings.dtype}")
    np.save(args.output, embeddings)
    print(f"Saved → {args.output}")
    print()
    print("Run isotropy test with:")
    print(f"  pytest tests/test_embedding_isotropy.py::test_real_embeddings_are_isotropic "
          f"--embeddings-npy {args.output} -s -m slow")


if __name__ == "__main__":
    main()
