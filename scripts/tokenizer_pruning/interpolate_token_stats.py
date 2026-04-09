#!/usr/bin/env python3
"""
Compute a weighted interpolation of token frequency histograms produced by
token_stats_from_jsonl.py and output a single sorted result file.

Each input file's raw counts are normalised to a probability distribution,
then the distributions are combined as a weighted average:

    p_interp(token) = sum_i( w_i * p_i(token) ) / sum_i(w_i)

Tokens absent from a file contribute 0 probability for that file.
The output histogram is sorted by interpolated probability (descending).

Usage:
    python interpolate_token_stats.py file1.json file2.json \\
        [--weights 0.6 0.4] \\
        [--output interpolated.json] \\
        [--top_k 50]
"""

import argparse
import sys
from pathlib import Path

import orjson


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_histogram(data: dict, path: str) -> list:
    """Return the histogram list from a token-stats JSON file."""
    if "histogram" in data:
        return data["histogram"]
    # fall back to top_N_most_frequent
    for key in data:
        if key.endswith("_most_frequent"):
            return data[key]
    raise KeyError(f"No histogram or '*_most_frequent' key found in {path}")


def _load(path: str) -> dict[int, float]:
    """Load a stats file and return {token_id: probability}."""
    raw = orjson.loads(Path(path).read_bytes())
    histogram = _find_histogram(raw, path)
    total = sum(e["count"] for e in histogram)
    if total == 0:
        raise ValueError(f"All counts are zero in {path}")
    return {e["token_id"]: e["count"] / total for e in histogram}, histogram


def _token_str(histogram: list) -> dict[int, str]:
    return {e["token_id"]: e["token"] for e in histogram}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+",
                        help="Two or more token-stats JSON files to interpolate")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Per-file weights (default: equal weights). "
                             "Must match the number of files if provided.")
    parser.add_argument("--output", default="interpolated.json",
                        help="Output path (default: interpolated.json)")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Keep only top-k tokens in output; -1 = all (default: -1)")
    args = parser.parse_args()

    if len(args.files) < 2:
        sys.exit("ERROR: at least 2 input files are required")

    weights = args.weights
    if weights is None:
        weights = [1.0] * len(args.files)
    elif len(weights) != len(args.files):
        sys.exit(f"ERROR: {len(args.files)} files but {len(weights)} weights")

    weight_sum = sum(weights)

    # ------------------------------------------------------------------
    # Load all files
    # ------------------------------------------------------------------
    probs: list[dict[int, float]] = []
    token_strings: dict[int, str] = {}

    for path in args.files:
        p, histogram = _load(path)
        probs.append(p)
        token_strings.update(_token_str(histogram))
        print(f"  {path}: {len(p):,} unique tokens")

    # ------------------------------------------------------------------
    # Weighted interpolation
    # ------------------------------------------------------------------
    all_token_ids: set[int] = set().union(*probs)
    print(f"Union vocabulary: {len(all_token_ids):,} tokens")

    interpolated: dict[int, float] = {}
    for tid in all_token_ids:
        interpolated[tid] = sum(
            w * p.get(tid, 0.0) for w, p in zip(weights, probs)
        ) / weight_sum

    # ------------------------------------------------------------------
    # Sort and optionally truncate
    # ------------------------------------------------------------------
    sorted_items = sorted(interpolated.items(), key=lambda x: x[1], reverse=True)
    if args.top_k != -1:
        sorted_items = sorted_items[:args.top_k]

    histogram_out = [
        {
            "token": token_strings.get(tid, f"<id:{tid}>"),
            "token_id": tid,
            "interpolated_prob": f"{prob:.6e}",
        }
        for tid, prob in sorted_items
    ]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    result = {
        "sources": [
            {"file": f, "weight": w} for f, w in zip(args.files, weights)
        ],
        "tokens_in_output": len(histogram_out),
        "histogram": histogram_out,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
    print(f"\nWrote {len(histogram_out):,} tokens → {out_path}")

    print("\nTop 10 tokens by interpolated probability:")
    for entry in histogram_out[:10]:
        print(f"  {entry['token']:20s}  {entry['interpolated_prob']}")


if __name__ == "__main__":
    main()
