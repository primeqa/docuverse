#!/usr/bin/env python3
"""
Extract average throughput and latency from benchmark JSON files produced by
benchmark_embedding_timing.py, printing results in the order given by a models file.

Usage:
    # Use a models file to define order, look for JSONs in a directory:
    python extract_benchmark_stats.py --models-file new_models.dat --json-dir latency-bv/

    # Pass JSON files explicitly (printed in argument order):
    python extract_benchmark_stats.py latency-bv/benchmark_*.json

    # Pass JSON files but reorder by models file:
    python extract_benchmark_stats.py --models-file new_models.dat latency-bv/benchmark_*.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


def model_to_filename(model_name: str) -> str:
    """Convert a HuggingFace model name to its benchmark JSON filename stem."""
    return "benchmark_" + re.sub(r"[/\-]", "_", model_name)


def extract_stats(json_path: Path) -> dict | None:
    """Extract avg throughput and avg latency from a benchmark JSON file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: could not read {json_path}: {e}", file=sys.stderr)
        return None

    # Prefer computing directly from per-language results
    if "results" in data and data["results"]:
        all_entries = [
            entry
            for lang_results in data["results"].values()
            for entry in lang_results
        ]
        if all_entries:
            throughputs = [e["throughput"] for e in all_entries if "throughput" in e]
            latencies = [e["avg_latency_ms"] for e in all_entries if "avg_latency_ms" in e]
            if throughputs and latencies:
                return {
                    "throughput": sum(throughputs) / len(throughputs),
                    "latency_ms": sum(latencies) / len(latencies),
                    "num_langs": len(data["results"]),
                }

    # Fallback: parse the summary line from the output field
    if "output" in data:
        m = re.search(
            r"Avg throughput\s*=\s*([\d.]+)\s*spans/s,\s*Avg latency\s*=\s*([\d.]+)\s*ms",
            data["output"],
        )
        if m:
            return {
                "throughput": float(m.group(1)),
                "latency_ms": float(m.group(2)),
                "num_langs": None,
            }

    print(f"WARNING: could not extract stats from {json_path}", file=sys.stderr)
    return None


def load_models_file(path: Path) -> list[str]:
    """Read model names from a file.

    Supports both newline-separated and space-separated model names,
    and ignores blank tokens and lines starting with '#'.
    """
    models = []
    with open(path) as f:
        content = f.read()
    for token in re.split(r"[\s]+", content):
        token = token.strip()
        if token and not token.startswith("#"):
            models.append(token)
    return models


def main():
    parser = argparse.ArgumentParser(description="Extract benchmark stats from JSON files")
    parser.add_argument(
        "json_files",
        nargs="*",
        help="Explicit JSON files to process (optional)",
    )
    parser.add_argument(
        "--models-file", "-m",
        default="new_models.dat",
        help="File listing model names in desired output order (default: new_models.dat)",
    )
    parser.add_argument(
        "--json-dir", "-d",
        default=".",
        help="Directory to search for JSON files when model names are used (default: .)",
    )
    parser.add_argument(
        "--no-models-file",
        action="store_true",
        help="Ignore models file; print files in the order given on the command line",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)

    # Build the ordered list of (label, path) pairs to process
    entries: list[tuple[str, Path]] = []

    if args.no_models_file or (not Path(args.models_file).exists() and not args.json_files):
        # Just use whatever files were given
        if not args.json_files:
            parser.error("No JSON files given and no models file found. "
                         "Provide JSON files or a --models-file.")
        for p in args.json_files:
            path = Path(p)
            entries.append((path.stem, path))

    elif Path(args.models_file).exists():
        models = load_models_file(Path(args.models_file))
        json_lookup: dict[str, Path] = {}

        # Index explicit files by stem
        for p in args.json_files:
            path = Path(p)
            json_lookup[path.stem] = path

        # Index files found in json_dir
        for p in json_dir.glob("benchmark_*.json"):
            json_lookup.setdefault(p.stem, p)

        for model in models:
            stem = model_to_filename(model)
            if stem in json_lookup:
                entries.append((model, json_lookup[stem]))
            else:
                print(f"WARNING: no JSON found for model '{model}' (expected {stem}.json)",
                      file=sys.stderr)

    else:
        # Models file not found; fall back to explicit files
        if not args.json_files:
            parser.error(f"Models file '{args.models_file}' not found and no JSON files given.")
        if not args.no_models_file:
            print(f"WARNING: models file '{args.models_file}' not found; "
                  "printing in argument order.", file=sys.stderr)
        for p in args.json_files:
            path = Path(p)
            entries.append((path.stem, path))

    if not entries:
        print("No benchmark files to process.", file=sys.stderr)
        sys.exit(1)

    # Print results
    col_w = max(len(label) for label, _ in entries) + 2
    header = f"{'Model':<{col_w}}  {'Throughput (spans/s)':>22}  {'Avg Latency (ms)':>18}"
    print(header)
    print("-" * len(header))

    for label, path in entries:
        stats = extract_stats(path)
        if stats:
            langs = f"  ({stats['num_langs']} langs)" if stats["num_langs"] else ""
            print(
                f"{label:<{col_w}}  {stats['throughput']:>22.2f}  {stats['latency_ms']:>18.2f}"
                f"{langs}"
            )
        else:
            print(f"{label:<{col_w}}  {'N/A':>22}  {'N/A':>18}")


if __name__ == "__main__":
    main()
