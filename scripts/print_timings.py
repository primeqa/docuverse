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


def _short_model(name):
    if not name:
        return '?'
    short = os.path.basename(name.rstrip('/'))
    if not short or set(short) <= {'.'}:
        return name or '?'
    return short


def _short_date(ts):
    return ts.split('T', 1)[0] if ts else ''


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
            rows.append({
                'file': os.path.basename(p),
                'context': cfg.get('max_text_length'),
                'model': cfg.get('model', '?'),
                'backend': backend,
                'precision': cfg.get('precision', '?'),
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
    ap.add_argument('--sort', default='throughput',
                    choices=['model', 'backend', 'precision', 'time_s',
                             'throughput', 'date', 'file'],
                    help='Sort column within each context-length group '
                         '(default: throughput).')
    ap.add_argument('--reverse', action='store_true',
                    help='Flip the natural sort order (descending for '
                         'throughput, ascending for everything else).')
    args = ap.parse_args()

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

    descending_by_default = args.sort in ('throughput',)
    reverse = descending_by_default ^ args.reverse

    def _sort_key(r):
        v = r.get(args.sort)
        return (1, 0) if v is None else (0, v)

    def _ctx_key(c):
        return (1, 0) if c is None else (0, c)

    by_ctx = defaultdict(list)
    for r in rows:
        by_ctx[r['context']].append(r)

    for ctx in sorted(by_ctx.keys(), key=_ctx_key):
        title = (f'Context length: {ctx}'
                 if ctx is not None else 'Context length: unbounded')
        group = sorted(by_ctx[ctx], key=_sort_key, reverse=reverse)
        print()
        print('=' * 78)
        print(f'  {title}  ({len(group)} run(s))')
        print('=' * 78)
        print(_format_table(group, columns))

    return 0


if __name__ == '__main__':
    sys.exit(main())
