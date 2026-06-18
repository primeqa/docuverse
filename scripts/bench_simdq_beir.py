#!/usr/bin/env python
"""bench_simdq_beir.py — run the R0–R6 simdq recipe sweep.

For each recipe in the spec's matrix, materialize a YAML config, run
ingest+retrieve+evaluate via the existing DocUVerse CLI as a
subprocess, parse the printed NDCG@10 / recall@100 numbers, capture
on-disk index size and end-to-end wall time, then write a single CSV.

Usage:
    python scripts/bench_simdq_beir.py \\
        --dataset fiqa \\
        --base-yaml config/beir_simdq_base.yaml \\
        --passages data/fiqa/passages.jsonl \\
        --queries  data/fiqa/queries.jsonl \\
        --qrels    data/fiqa/qrels.tsv \\
        --project-dir /scratch/simdq-fiqa \\
        --out simdq_recipe_sweep_fiqa.csv

Recipes (spec section 10):
    R0  symmetric Hamming   + rescore alpha=10   family=hamming, d=D
    R1  asym b=1            no  rescore     family=asymmetric, b=1, alpha=1
    R2  asym b=1            +   rescore alpha=10 b=1, alpha=10
    R3  asym b=2            no  rescore     b=2, alpha=1
    R4  asym b=2, halve dim no  rescore     b=2, d=D/2, projection=random_orthogonal
    R5  asym b=4, halve dim no  rescore     b=4, d=D/2, projection=random_orthogonal
    R6  float baseline                       (skipped: run via existing dense engine)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import yaml


RECIPES = [
    # (recipe_id, label, simdq_overrides)
    ("R0", "hamming + rescore alpha=10",
     {"simdq_family": "hamming", "simdq_b": 1, "simdq_projection": "identity",
      "simdq_d": None, "simdq_store_floats": True, "simdq_rescore_alpha": 10}),
    ("R1", "asym b=1, no rescore",
     {"simdq_family": "asymmetric", "simdq_b": 1, "simdq_projection": "identity",
      "simdq_d": None, "simdq_store_floats": False, "simdq_rescore_alpha": 1}),
    ("R2", "asym b=1 + rescore alpha=10",
     {"simdq_family": "asymmetric", "simdq_b": 1, "simdq_projection": "identity",
      "simdq_d": None, "simdq_store_floats": True, "simdq_rescore_alpha": 10}),
    ("R3", "asym b=2, no rescore",
     {"simdq_family": "asymmetric", "simdq_b": 2, "simdq_projection": "identity",
      "simdq_d": None, "simdq_store_floats": False, "simdq_rescore_alpha": 1}),
    ("R4", "asym b=2, d=D/2, random_orthogonal",
     {"simdq_family": "asymmetric", "simdq_b": 2, "simdq_projection": "random_orthogonal",
      "simdq_d": "HALF", "simdq_store_floats": False, "simdq_rescore_alpha": 1}),
    ("R5", "asym b=4, d=D/2, random_orthogonal",
     {"simdq_family": "asymmetric", "simdq_b": 4, "simdq_projection": "random_orthogonal",
      "simdq_d": "HALF", "simdq_store_floats": False, "simdq_rescore_alpha": 1}),
    # R6 (float baseline) intentionally not run from this script — uses
    # existing dense engine of choice (lancedb-dense / faiss / milvus-dense).
]


def _materialize_yaml(base: dict, dataset: str, recipe_id: str, recipe_overrides: dict,
                      passages: str, queries: str, qrels: str, project_dir: str,
                      encoder_dim_hint: int | None) -> dict:
    """Apply per-recipe overrides; resolve simdq_d='HALF' to D/2."""
    cfg = json.loads(json.dumps(base))   # deep copy

    cfg["search_engine"]["index_name"] = f"{dataset}_simdq_{recipe_id}"
    cfg["search_engine"]["project_dir"] = project_dir
    cfg["retrieval"]["input_passages"] = passages
    cfg["retrieval"]["input_queries"] = queries
    cfg["evaluation"]["qrels"] = qrels

    se = cfg["search_engine"]
    for k, v in recipe_overrides.items():
        if v == "HALF":
            if encoder_dim_hint is None:
                raise RuntimeError(
                    f"{recipe_id}: simdq_d=HALF needs --encoder-dim to know D/2"
                )
            se[k] = encoder_dim_hint // 2
        else:
            se[k] = v
    return cfg


_NDCG_RE   = re.compile(r"NDCG@10[:\s=]+([0-9.]+)", re.IGNORECASE)
_RECALL_RE = re.compile(r"Recall@100[:\s=]+([0-9.]+)", re.IGNORECASE)


def _parse_metrics(stdout: str) -> dict:
    out = {}
    m = _NDCG_RE.search(stdout)
    if m:
        out["ndcg10"] = float(m.group(1))
    m = _RECALL_RE.search(stdout)
    if m:
        out["recall100"] = float(m.group(1))
    return out


def _index_size_bytes(project_dir: str, index_name: str) -> int:
    p = Path(project_dir) / "simdq_data" / index_name
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _run_recipe(yaml_path: Path) -> tuple[str, float, int]:
    """Invoke ingest_and_test as a subprocess. Returns (stdout, wall_seconds, exit_code)."""
    t0 = time.time()
    cmd = [
        "python", "-m", "docuverse.utils.ingest_and_test",
        "--config", str(yaml_path),
        "--actions", "ire",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    return res.stdout + "\n" + res.stderr, elapsed, res.returncode


def main():
    ap = argparse.ArgumentParser(
        description="Run R0–R6 simdq recipe sweep on a BEIR-format dataset."
    )
    ap.add_argument("--dataset", required=True, help="dataset slug used in index_name")
    ap.add_argument("--base-yaml", required=True, type=Path,
                    help="base YAML template (config/beir_simdq_base.yaml)")
    ap.add_argument("--passages", required=True,
                    help="path to passages JSONL (id, text, title)")
    ap.add_argument("--queries", required=True,
                    help="path to queries JSONL (id, text)")
    ap.add_argument("--qrels", required=True,
                    help="path to qrels TSV (q-id, corpus-id, score)")
    ap.add_argument("--project-dir", required=True,
                    help="directory where simdq_data/ sub-dirs are written")
    ap.add_argument("--encoder-dim", type=int, default=768,
                    help="encoder hidden dim (= D) — needed to resolve simdq_d=HALF")
    ap.add_argument("--out", required=True,
                    help="output CSV path")
    ap.add_argument("--recipes", nargs="*", default=None,
                    help="subset of recipe IDs to run (default: all R0–R5)")
    ap.add_argument("--keep-yaml-dir", default=None,
                    help="if set, written YAMLs are kept here for debugging")
    args = ap.parse_args()

    base = yaml.safe_load(args.base_yaml.read_text())

    yaml_dir = Path(args.keep_yaml_dir) if args.keep_yaml_dir else \
               Path(args.project_dir) / "_sweep_yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(args.project_dir, exist_ok=True)

    rows = []
    for rid, label, overrides in RECIPES:
        if args.recipes and rid not in args.recipes:
            continue
        cfg = _materialize_yaml(base, args.dataset, rid, overrides,
                                args.passages, args.queries, args.qrels,
                                args.project_dir, args.encoder_dim)
        yaml_path = yaml_dir / f"{args.dataset}_{rid}.yaml"
        yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        index_name = cfg["search_engine"]["index_name"]
        # Always start from a clean index dir to keep recipes independent.
        idx_dir = Path(args.project_dir) / "simdq_data" / index_name
        if idx_dir.exists():
            shutil.rmtree(idx_dir)

        print(f"\n=== {rid} — {label} ===")
        stdout, elapsed_s, rc = _run_recipe(yaml_path)
        metrics = _parse_metrics(stdout)
        idx_bytes = _index_size_bytes(args.project_dir, index_name)
        rows.append({
            "recipe_id": rid,
            "label": label,
            "simdq_family": cfg["search_engine"].get("simdq_family"),
            "b": cfg["search_engine"].get("simdq_b"),
            "d": cfg["search_engine"].get("simdq_d"),
            "projection": cfg["search_engine"].get("simdq_projection"),
            "store_floats": cfg["search_engine"].get("simdq_store_floats"),
            "rescore_alpha": cfg["search_engine"].get("simdq_rescore_alpha"),
            "ndcg10": metrics.get("ndcg10"),
            "recall100": metrics.get("recall100"),
            "wall_seconds": round(elapsed_s, 2),
            "index_bytes": idx_bytes,
            "exit_code": rc,
        })
        print(f"  ndcg10={metrics.get('ndcg10')}  "
              f"recall100={metrics.get('recall100')}  "
              f"wall={elapsed_s:.0f}s  index={idx_bytes/1e6:.1f}MB  rc={rc}")

    # Write CSV
    if not rows:
        print("\nNo recipes matched --recipes filter; no CSV written.")
        return
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
