#!/usr/bin/env python3
"""Print a tabular summary of benchmark JSON files in the timings/ directory.

Each table is grouped by context length (config.max_text_length); within a
group, rows are sorted by throughput descending. One row is emitted per
(file, backend) pair so multi-backend runs show one row per backend.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


_BACKEND_SUFFIXES = ('-openvino', '-ovino', '-onnx')


def _short_model(name):
    if not name:
        return '?'
    short = os.path.basename(name.rstrip('/'))
    if not short or set(short) <= {'.'}:
        return name or '?'
    # Strip trailing backend tags (local copies are often named e.g.
    # "granite-embedding-278-ovino"); the backend column already conveys this.
    for suffix in _BACKEND_SUFFIXES:
        if short.lower().endswith(suffix):
            short = short[: -len(suffix)]
            break
    return short


def _short_date(ts):
    return ts.split('T', 1)[0] if ts else ''


_BACKEND_SHORT = {'pytorch': 'pt', 'openvino': 'ov', 'llama_cpp': 'llamacpp'}


def _precision_from_filename(filename, backend):
    """Extract the backend's precision tag from a filename written by
    sentence_transformer_backend_comparison.py — its naming convention is
    <model>.<backend1>_<prec1>[-<backend2>_<prec2>...].l<len>.<device>[...]
    so the backend/precision pairs live in the second dot-segment, joined
    with '-'. Returns None if no <backend>_<prec> token matches.
    """
    short = _BACKEND_SHORT.get(backend, backend)
    parts = filename.split('.')
    if len(parts) < 2:
        return None
    for token in parts[1].split('-'):
        if token.startswith(short + '_'):
            return token[len(short) + 1:]
    return None


def _collect_rows(json_paths):
    rows = []
    for p in json_paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}", file=sys.stderr)
            continue
        cfg = d.get('config') or {}
        perf = d.get('performance') or {}
        meta = d.get('metadata') or {}
        for backend, entry in perf.items():
            if not isinstance(entry, dict):
                continue
            fname = os.path.basename(p)
            precision = _precision_from_filename(fname, backend) or cfg.get('precision', '?')
            rows.append({
                'file': fname,
                'context': cfg.get('max_text_length'),
                'model': cfg.get('model', '?'),
                'backend': backend,
                'precision': precision,
                'device': cfg.get('device', '?'),
                'samples': cfg.get('num_sentences'),
                'batch': cfg.get('batch_size'),
                'time_s': entry.get('time_seconds'),
                'throughput': entry.get('throughput_docs_per_sec'),
                'date': _short_date(meta.get('timestamp', '')),
            })
    return rows


def _format_table(rows, columns):
    headers = [c[1] for c in columns]
    keys = [c[0] for c in columns]
    fmts = [c[2] for c in columns]
    string_rows = []
    for r in rows:
        sr = []
        for k, fmt in zip(keys, fmts):
            v = r.get(k)
            sr.append('-' if v is None else fmt(v))
        string_rows.append(sr)
    widths = [
        max(len(h), max((len(s[i]) for s in string_rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = '  '
    lines = [sep.join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append(sep.join('-' * w for w in widths))
    for s in string_rows:
        lines.append(sep.join(c.ljust(w) for c, w in zip(s, widths)))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', '--timings-dir', dest='dir', default='timings',
                    help='Directory of timing JSON files (default: timings).')
    valid_sort_keys = ('model', 'backend', 'precision', 'time_s',
                       'throughput', 'date', 'file')
    ap.add_argument('--sort', default='throughput',
                    help='Comma-separated sort columns within each '
                         'context-length group; first key is most significant '
                         f'(default: throughput). Valid: {", ".join(valid_sort_keys)}')
    ap.add_argument('--reverse', action='store_true',
                    help='Flip the natural sort order on every key '
                         '(descending for throughput, ascending for the rest).')
    args = ap.parse_args()

    sort_keys = [k.strip() for k in args.sort.split(',') if k.strip()]
    if not sort_keys:
        ap.error('--sort must specify at least one column')
    for k in sort_keys:
        if k not in valid_sort_keys:
            ap.error(f"invalid --sort key '{k}' (valid: {', '.join(valid_sort_keys)})")

    paths = sorted(glob.glob(os.path.join(args.dir, '*.json')))
    if not paths:
        print(f'No JSON files in {args.dir}', file=sys.stderr)
        return 1

    rows = _collect_rows(paths)
    if not rows:
        print('No usable rows found.', file=sys.stderr)
        return 1

    columns = [
        ('model', 'Model', _short_model),
        ('backend', 'Backend', str),
        ('precision', 'Prec', str),
        ('device', 'Dev', str),
        ('samples', 'Samples', lambda v: f'{v}'),
        ('batch', 'Batch', lambda v: f'{v}'),
        ('time_s', 'Time(s)', lambda v: f'{v:.2f}'),
        ('throughput', 'Throughput', lambda v: f'{v:.2f}'),
        ('date', 'Date', str),
        ('file', 'File', str),
    ]

    descending_by_default = {'throughput'}

    def _key_value(row, key):
        v = row.get(key)
        # Sort by the displayed value for the model column so trimmed-suffix
        # rows ("granite-embedding-311m-multilingual-r2-onnx" and "...-ovino")
        # group with the canonical name they render as.
        if key == 'model':
            v = _short_model(v) if v else v
        return (1, 0) if v is None else (0, v)

    def _ctx_key(c):
        return (1, 0) if c is None else (0, c)

    by_ctx = defaultdict(list)
    for r in rows:
        by_ctx[r['context']].append(r)

    for ctx in sorted(by_ctx.keys(), key=_ctx_key):
        title = (f'Context length: {ctx}'
                 if ctx is not None else 'Context length: unbounded')
        # Stable-sort right-to-left so the first key listed is most significant.
        group = list(by_ctx[ctx])
        for k in reversed(sort_keys):
            desc = (k in descending_by_default) ^ args.reverse
            group.sort(key=lambda row, _k=k: _key_value(row, _k), reverse=desc)
        print()
        print('=' * 78)
        print(f'  {title}  ({len(group)} run(s))')
        print('=' * 78)
        print(_format_table(group, columns))

    return 0


if __name__ == '__main__':
    sys.exit(main())
