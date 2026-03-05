#!/usr/bin/env python3
"""Compare quality scores and run-times across multiple .metrics files.

Usage:
    python scripts/compare_metrics.py output/*.metrics
    python scripts/compare_metrics.py file1.metrics file2.metrics file3.metrics
    python scripts/compare_metrics.py output/*.metrics --sort MRR@10
    python scripts/compare_metrics.py output/*.metrics --metrics M@1 M@10 MRR@10 NDCG@10
"""

import argparse
import json
import os
import re
import sys


def parse_metrics_file(path):
    """Parse a .metrics file and return quality scores and timing info."""
    with open(path) as f:
        content = f.read()

    result = {"file": path, "name": os.path.basename(path).replace(".metrics", "")}

    # Try the tabular format first (header line + values line)
    lines = content.strip().split("\n")
    scores = _parse_tabular(lines)
    if scores is None:
        scores = _parse_json(content)
    result["scores"] = scores or {}

    # Parse timing section
    result["timing"] = _parse_timing(lines)

    return result


def _parse_tabular(lines):
    """Parse the tabular 'Model  M@1  M@5 ...' format."""
    if not lines:
        return None
    header = lines[0]
    # Check for metric column headers
    if not re.search(r"\bM@\d+\b|\bMRR@\d+\b|\bNDCG@\d+\b", header):
        return None

    # Split header into column names
    cols = header.split()
    if len(cols) < 2 or len(lines) < 2:
        return None

    vals = lines[1].split()
    if len(vals) < 2:
        return None

    # First column is Model name; rest are metric values
    # The model name may contain spaces or be multi-token, so align from the right
    num_metrics = len(cols) - 1  # exclude "Model"
    metric_vals = vals[-num_metrics:]

    scores = {}
    for col, val in zip(cols[1:], metric_vals):
        try:
            scores[col] = float(val)
        except ValueError:
            pass
    return scores if scores else None


def _parse_json(content):
    """Parse old-style JSON metrics (success__WARNING etc.)."""
    try:
        data = json.loads(content.strip().split("\n")[0]
                          if not content.strip().startswith("{")
                          else content.split("\n\n")[0])
    except (json.JSONDecodeError, IndexError):
        # Try to find a JSON block
        try:
            brace_start = content.index("{")
            brace_depth = 0
            for i in range(brace_start, len(content)):
                if content[i] == "{":
                    brace_depth += 1
                elif content[i] == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        data = json.loads(content[brace_start:i + 1])
                        break
            else:
                return None
        except (ValueError, json.JSONDecodeError):
            return None

    scores = {}
    for key, val_dict in data.items():
        if isinstance(val_dict, dict):
            key_lower = key.lower()
            if "lienient" in key_lower or "lenient" in key_lower:
                prefix = "lenient_M"
            elif "success" in key_lower:
                prefix = "M"
            else:
                prefix = key
            for k_at, v in val_dict.items():
                scores[f"{prefix}@{k_at}"] = v
    return scores if scores else None


def _parse_timing(lines):
    """Extract timing info from the Timing: section."""
    timing = {}
    in_timing = False
    for line in lines:
        if line.strip().startswith("Timing:"):
            in_timing = True
            continue
        if in_timing:
            if line.startswith("=") or line.startswith("***"):
                break
            if not line.strip() or line.strip().startswith("Name"):
                continue
            # e.g. "  ingest_and_test   180.3s   100.0%  ..."
            # or   "    search            6.3s     3.5%  ..."
            m = re.match(r"^(\s*)(\S+)\s+([\d.]+)s", line)
            if m:
                indent = len(m.group(1))
                name = m.group(2)
                secs = float(m.group(3))
                if indent == 0:  # top-level = total
                    timing["total"] = secs
                elif indent == 2:  # direct children (ingest, search, etc.)
                    timing[name] = secs
                # skip deeper nesting
    return timing


def _fmt_val(val):
    """Format a single cell value as a string."""
    if val is None:
        return "--"
    elif isinstance(val, float):
        return f"{val:.1f}" if val >= 60 else f"{val:.3f}"
    else:
        return str(val)


def format_table(rows, columns):
    """Format a list of dicts into a table string."""
    name_width = max(len(r["name"]) for r in rows) + 2
    name_width = max(name_width, len("Model") + 2)

    # Compute per-column width: max of header length and all formatted values, plus padding
    col_widths = {}
    for col in columns:
        w = len(col)
        for row in rows:
            w = max(w, len(_fmt_val(row.get(col))))
        col_widths[col] = w + 2  # 2 chars padding

    header = f"{'Model':<{name_width}}"
    for col in columns:
        header += f"{col:>{col_widths[col]}}"
    sep = "-" * len(header)

    lines = [header, sep]
    for row in rows:
        line = f"{row['name']:<{name_width}}"
        for col in columns:
            line += f"{_fmt_val(row.get(col)):>{col_widths[col]}}"
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare quality scores and run-times across .metrics files.")
    parser.add_argument("files", nargs="+", help=".metrics files to compare")
    parser.add_argument("--sort", default=None,
                        help="Sort by this metric column (e.g. MRR@10, NDCG@10)")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Show only these quality metrics (e.g. M@1 M@10 MRR@10 NDCG@10)")
    parser.add_argument("--timing", nargs="*", default=None,
                        help="Show only these timing keys (e.g. total ingest search). "
                             "Default: all available.")
    parser.add_argument("--no-timing", action="store_true",
                        help="Hide timing section")
    parser.add_argument("--no-quality", action="store_true",
                        help="Hide quality section")
    args = parser.parse_args()

    # Parse all files
    results = []
    for path in args.files:
        if not os.path.isfile(path):
            print(f"Warning: {path} not found, skipping.", file=sys.stderr)
            continue
        results.append(parse_metrics_file(path))

    if not results:
        print("No valid .metrics files found.", file=sys.stderr)
        sys.exit(1)

    # Determine quality columns
    all_score_keys = []
    seen = set()
    for r in results:
        for k in r["scores"]:
            if k not in seen:
                all_score_keys.append(k)
                seen.add(k)

    quality_cols = args.metrics if args.metrics else all_score_keys

    # Determine timing columns
    all_timing_keys = []
    seen_t = set()
    for r in results:
        for k in r["timing"]:
            if k not in seen_t:
                all_timing_keys.append(k)
                seen_t.add(k)

    timing_cols = args.timing if args.timing is not None else all_timing_keys

    # Build unified rows
    rows = []
    for r in results:
        row = {"name": r["name"]}
        for k in quality_cols:
            row[k] = r["scores"].get(k)
        for k in timing_cols:
            row[k] = r["timing"].get(k)
        rows.append(row)

    # Sort
    if args.sort:
        sort_key = args.sort
        rows.sort(key=lambda r: r.get(sort_key) if r.get(sort_key) is not None else -1,
                  reverse=True)

    # Print quality table
    if not args.no_quality and quality_cols:
        print("=== Quality Scores ===\n")
        print(format_table(rows, quality_cols))
        print()

    # Print timing table
    if not args.no_timing and timing_cols:
        timing_display = [f"{k} (s)" for k in timing_cols]
        # Build timing rows with renamed keys
        timing_rows = []
        for row in rows:
            tr = {"name": row["name"]}
            for k, display in zip(timing_cols, timing_display):
                tr[display] = row.get(k)
            timing_rows.append(tr)
        print("=== Timing (seconds) ===\n")
        print(format_table(timing_rows, timing_display))
        print()


if __name__ == "__main__":
    main()
