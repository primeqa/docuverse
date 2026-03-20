#!/usr/bin/env python3
"""Compare timing statistics across multiple model runs.

Takes a list of .timing.json files and timing keys, produces an Excel
spreadsheet with one tab per timing key, listing timing statistics for
each model. Statistics can be divided by a batch size (auto-detected
from .metrics files if not provided).

Usage:
    python scripts/compare_timing.py \
        --timing_files output/model1.timing.json output/model2.timing.json \
        --keys "ingest_and_test::ingest::encode" "ingest_and_test::search::retrieve::encode::model_forward" \
        --output comparison.xlsx \
        --batch_size 128

    # Auto-detect batch size from .metrics files:
    python scripts/compare_timing.py \
        --timing_files output/model1.timing.json output/model2.timing.json \
        --keys "ingest_and_test::ingest::encode" \
        --output comparison.xlsx
"""

import argparse
import json
import os
import re
import sys
import yaml

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare timing statistics across model runs and produce an Excel spreadsheet."
    )
    parser.add_argument(
        "--timing_files", "-t", nargs="+", required=True,
        help="List of .timing.json files to compare"
    )
    parser.add_argument(
        "--keys", "-k", nargs="+", required=True,
        help="Timing keys to extract (e.g. 'ingest_and_test::ingest::encode')"
    )
    parser.add_argument(
        "--output", "-o", default="timing_comparison.xlsx",
        help="Output Excel file (default: timing_comparison.xlsx)"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=None,
        help="Batch size to divide statistics by. If not provided, auto-detect from .metrics files."
    )
    parser.add_argument(
        "--per_item", action="store_true", default=False,
        help="Divide statistics by batch size to show per-item timings"
    )
    parser.add_argument(
        "--stats", "-s", nargs="+",
        default=["count", "sum", "mean", "median", "std", "p95", "p99", "max"],
        help="Statistics to include in the output"
    )
    return parser.parse_args()


def extract_model_name(timing_file):
    """Extract a short model name from the timing file path."""
    basename = os.path.basename(timing_file)
    # Strip .timing.json suffix
    for suffix in [".timing.json", ".json"]:
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
            break
    # Strip common prefixes for readability
    for prefix in ["unified-search-full-", "unified-search-short23k-", "unified-search-"]:
        if basename.startswith(prefix):
            basename = basename[len(prefix):]
            break
    return basename


def read_bulk_batch_from_metrics(metrics_file):
    """Read bulk_batch from a .metrics file's config section.

    The config is written as a YAML string with literal '\\n' for newlines,
    on the line(s) following '****** Config: *******'.
    """
    if not os.path.exists(metrics_file):
        return None

    with open(metrics_file, "r") as f:
        lines = f.readlines()

    # Find the config section
    config_start = None
    for i, line in enumerate(lines):
        if "****** Config: *******" in line:
            config_start = i + 1
            break

    if config_start is None:
        return None

    # Collect config lines until the next separator or end of file
    config_text = ""
    for i in range(config_start, len(lines)):
        line = lines[i].strip()
        if line.startswith("=" * 5):
            break
        config_text += line

    # The config has literal \n instead of actual newlines
    config_yaml = config_text.replace("\\n", "\n")

    try:
        config = yaml.safe_load(config_yaml)
    except yaml.YAMLError:
        return None

    # Look for bulk_batch in the retriever section
    if isinstance(config, dict):
        retriever = config.get("retriever", {})
        if isinstance(retriever, dict):
            return retriever.get("bulk_batch")

    return None


def get_batch_size_for_file(timing_file, default_batch_size=None):
    """Get batch size for a timing file, from argument or .metrics auto-detection."""
    if default_batch_size is not None:
        return default_batch_size

    # Try to find a corresponding .metrics file
    metrics_file = timing_file.replace(".timing.json", ".metrics")
    batch_size = read_bulk_batch_from_metrics(metrics_file)
    if batch_size is not None:
        return batch_size

    return None


def load_timing_data(timing_file):
    """Load timing data from a .timing.json file."""
    with open(timing_file, "r") as f:
        return json.load(f)


def shorten_key_for_tab(key):
    """Shorten a timing key for use as an Excel tab name (max 31 chars)."""
    # Take the last 2-3 components, use '.' separator (Excel forbids ':')
    parts = key.split("::")
    short = ".".join(parts[-2:]) if len(parts) > 2 else key.replace("::", ".")
    # Excel sheet names cannot contain: : \ / ? * [ ]
    for ch in [":", "\\", "/", "?", "*", "[", "]"]:
        short = short.replace(ch, "_")
    if len(short) > 31:
        short = short[-31:]
    return short


def write_sheet(ws, key, timing_files, timing_data_list, batch_sizes, stats):
    """Write one sheet for a timing key."""
    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    header_align = Alignment(horizontal="center", wrap_text=True)
    num_format = "#,##0.000"
    int_format = "#,##0"
    thin_border = Border(
        bottom=Side(style="thin", color="B0B0B0")
    )

    # Title row
    ws.cell(row=1, column=1, value=f"Key: {key}").font = Font(bold=True, size=12)

    # Header row
    row = 3
    ws.cell(row=row, column=1, value="Model").font = header_font
    ws.cell(row=row, column=1).fill = header_fill
    ws.cell(row=row, column=2, value="Batch Size").font = header_font
    ws.cell(row=row, column=2).fill = header_fill
    ws.cell(row=row, column=2).alignment = header_align

    col = 3
    for stat in stats:
        cell = ws.cell(row=row, column=col, value=stat)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        col += 1

    # Data rows
    for i, timing_file in enumerate(timing_files):
        row = 4 + i
        model_name = extract_model_name(timing_file)
        data = timing_data_list[i]
        batch_size = batch_sizes[i]

        ws.cell(row=row, column=1, value=model_name)

        if batch_size is not None:
            ws.cell(row=row, column=2, value=batch_size).number_format = int_format
        else:
            ws.cell(row=row, column=2, value="-")

        statistics = data.get("statistics", {})
        key_stats = statistics.get(key)

        if key_stats is None:
            # Key not found in this file - mark all stats as N/A
            col = 3
            for stat in stats:
                ws.cell(row=row, column=col, value="N/A")
                col += 1
            continue

        col = 3
        divisor = batch_size if batch_size is not None else 1
        for stat in stats:
            value = key_stats.get(stat)
            if value is not None:
                if stat == "count":
                    # Count is never divided by batch size
                    ws.cell(row=row, column=col, value=value).number_format = int_format
                else:
                    divided = value / divisor
                    ws.cell(row=row, column=col, value=divided).number_format = num_format
            else:
                ws.cell(row=row, column=col, value="N/A")
            col += 1

        # Light border for readability
        for c in range(1, col):
            ws.cell(row=row, column=c).border = thin_border

    # Auto-fit column widths
    ws.column_dimensions["A"].width = 50
    ws.column_dimensions["B"].width = 12
    for c in range(3, 3 + len(stats)):
        ws.column_dimensions[get_column_letter(c)].width = 14


def main():
    args = parse_args()

    # Load all timing data
    timing_data_list = []
    batch_sizes = []
    for tf in args.timing_files:
        if not os.path.exists(tf):
            print(f"Warning: {tf} not found, skipping.", file=sys.stderr)
            timing_data_list.append({"statistics": {}})
            batch_sizes.append(None)
            continue
        timing_data_list.append(load_timing_data(tf))
        bs = get_batch_size_for_file(tf, args.batch_size)
        batch_sizes.append(bs)
        model_name = extract_model_name(tf)
        print(f"  {model_name}: batch_size={bs}")

    # Determine if we should divide by batch size
    if not args.per_item and args.batch_size is None:
        # If user didn't ask for per-item, set all batch sizes to None (no division)
        if not any(bs is not None for bs in batch_sizes):
            pass  # all None already
        else:
            # If some batch sizes were auto-detected but --per_item not set,
            # still show the batch size column but don't divide
            pass

    if args.per_item:
        print(f"Dividing statistics by batch size (per-item mode)")
    else:
        # Don't divide - set effective batch sizes to None for the write step
        effective_batch_sizes = [None] * len(batch_sizes)

    # Create workbook
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    for key in args.keys:
        tab_name = shorten_key_for_tab(key)
        ws = wb.create_sheet(title=tab_name)
        write_sheet(
            ws, key, args.timing_files, timing_data_list,
            batch_sizes if args.per_item else [None] * len(batch_sizes),
            args.stats
        )

    wb.save(args.output)
    print(f"\nSaved to {args.output}")
    print(f"  {len(args.keys)} tab(s): {', '.join(shorten_key_for_tab(k) for k in args.keys)}")
    print(f"  {len(args.timing_files)} model(s) compared")


if __name__ == "__main__":
    main()