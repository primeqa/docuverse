#!/usr/bin/env python3
"""Extract completed runs from log files and produce an exclusion file for mrl_sweep.sh.

Scans log files for lines containing 'docuverse/utils/ingest_and_test.py',
extracts --model_name, --max_doc_length, --config, and --matryoshka_dim,
and writes a tab-separated exclusion file.

Usage:
    python scripts/make_exclude_list.py -o exclude.tsv logs/*.out
    python scripts/make_exclude_list.py --only-successful -o exclude.tsv logs/*.out
"""

import argparse
import re
import shlex
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract completed runs from log files to create an mrl_sweep.sh exclusion file."
    )
    parser.add_argument(
        "log_files", nargs="+",
        help="Log files to scan"
    )
    parser.add_argument(
        "-o", "--output", default="-",
        help="Output exclusion file (default: stdout)"
    )
    parser.add_argument(
        "--only-successful", action="store_true",
        help="Only include runs that completed successfully (no errors detected)"
    )
    return parser.parse_args()


def extract_arg(tokens, flag):
    """Extract the value following a flag from a token list."""
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def extract_runs_from_file(filepath, only_successful=False):
    """Extract (model, max_doc_length, config, dim) tuples from a log file."""
    try:
        with open(filepath, "r", errors="replace") as f:
            content = f.read()
    except OSError as e:
        print(f"Warning: cannot read {filepath}: {e}", file=sys.stderr)
        return []

    lines = content.splitlines()
    runs = []

    for line in lines:
        # Match lines containing the ingest_and_test.py command
        if "docuverse/utils/ingest_and_test.py" not in line:
            continue

        # Extract the command portion starting from 'python' or the script path
        match = re.search(r'(python\s+\S*docuverse/utils/ingest_and_test\.py\s+.*)', line)
        if not match:
            # Try without 'python' prefix (e.g. direct invocation)
            match = re.search(r'(\S*docuverse/utils/ingest_and_test\.py\s+.*)', line)
        if not match:
            continue

        cmd_str = match.group(1)
        try:
            tokens = shlex.split(cmd_str)
        except ValueError:
            # Malformed quoting - try basic split
            tokens = cmd_str.split()

        model = extract_arg(tokens, "--model_name") or "*"
        mdl = extract_arg(tokens, "--max_doc_length") or "*"
        config = extract_arg(tokens, "--config") or "*"
        dim = extract_arg(tokens, "--matryoshka_dim") or "*"

        runs.append((model, mdl, config, dim))

    if only_successful and runs:
        # Check for error indicators in the file
        error_patterns = [
            r"XXXXXXXXXXXX",           # runCmd failure marker
            r"Traceback \(most recent call last\)",
            r"^ERROR:",
            r"RuntimeError:",
            r"CUDA out of memory",
        ]
        has_error = any(
            re.search(pat, content, re.MULTILINE)
            for pat in error_patterns
        )
        if has_error:
            return []

    return runs


def main():
    args = parse_args()

    all_runs = []
    seen = set()

    for logfile in args.log_files:
        runs = extract_runs_from_file(logfile, args.only_successful)
        for run in runs:
            key = tuple(run)
            if key not in seen:
                seen.add(key)
                all_runs.append(run)

    # Write output
    if args.output == "-":
        out = sys.stdout
    else:
        out = open(args.output, "w")

    out.write("# Exclusion file for mrl_sweep.sh --exclude\n")
    out.write("# model\tmax_doc_length\tconfig\tdim\n")
    for model, mdl, config, dim in all_runs:
        out.write(f"{model}\t{mdl}\t{config}\t{dim}\n")

    if args.output != "-":
        out.close()
        print(f"Wrote {len(all_runs)} exclusion(s) to {args.output}", file=sys.stderr)
    else:
        print(f"# {len(all_runs)} exclusion(s) total", file=sys.stderr)


if __name__ == "__main__":
    main()
