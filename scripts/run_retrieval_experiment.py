#!/usr/bin/env python
"""run_retrieval_experiment.py — one-shot retrieval quality × speed experiment
runner: simdq, Fagin TA, Milvus, and FAISS on the same vectors and queries.

Reproduces reports like docs/simdq_euler.watson.ibm.com_nq_real.md end-to-end:

  1. Build the simdq index directories (asym b=2 + 1-bit hamming) for each
     model if they don't exist yet: encode dataset.corpus_jsonl with the
     model's encoder (cached as .npy), SimdqIndex.build + save both families
     with store_floats=True, and write engine_metadata.json with the id_map.
     Prebuilt directories (e.g. from the DocUVerse ingestion pipeline, with
     its chunking) are used as-is when they exist; --force-build rebuilds.
  2. Encode the dataset's queries with each configured encoder → .npy
     (skipped when the cached .npy already exists; --force-encode redoes it).
  3. Quality: Recall@10/@K, MRR@10, nDCG@10 vs gold labels, plus agreement
     with the exact FLAT fp32 scan, for asym b=2 / 1-bit hamming (each with
     and without the fp32 rescore stage) and for the enabled Milvus / FAISS /
     Fagin TA baselines (FLAT/FlatIP, HNSW, and the exact Threshold Algorithm
     scan) on the same vectors and queries.
  4. Speed: Phase 1 (asym b=2, default BLAS and BLAS=1), Phase 6/8 (hamming
     SoA) — reusing the sweep code in scripts/investigate_simdq_hardware.py
     (these thread-scaling sweeps are opt-in via --thread-sweeps; off by
     default) — and a Phase 5b head-to-head of Milvus FLAT, Milvus HNSW,
     FAISS FlatIP,
     FAISS HNSW, Fagin TA (FaginIndex, built in-memory from the same fp32
     vectors), and the simdq variants, each with an agree@K-vs-exact
     column, on the same corpora + queries.
  5. Write one combined markdown report (+ a .json with the raw numbers, a
     .sweeps.svg chart of the thread-scaling sweeps, and a .frontier.svg
     chart of throughput vs nDCG@10 per engine).

Datasets and models come from a YAML config (see experiments/simdq/nq_real.yaml)
and every scalar setting can be overridden on the command line; models can
also be given entirely on the command line via repeatable --model flags.
The YAML is read through the DocUVerse config reader (read_config_file), so
Jinja2 templating works: {{var}} references other config keys, filters like
short_model are available, and CLI overrides apply before rendering.

Usage:
    # Everything from YAML:
    python scripts/run_retrieval_experiment.py --config experiments/simdq/nq_real.yaml

    # YAML + overrides:
    python scripts/run_retrieval_experiment.py --config experiments/simdq/nq_real.yaml \
        --workers 32 --no-milvus --out docs/simdq_mybox_nq.md

    # No YAML at all:
    python scripts/run_retrieval_experiment.py \
        --dataset-name nq_dev \
        --queries-jsonl benchmark/nq_new/nq-dev-fixed.jsonl \
        --model "tag=granite-97m-r2 (384d);encoder=ibm-granite/granite-embedding-97m-multilingual-r2;asym=experiments/nq_new/simq_data/nq_dev-simdq-simdq-granite97m-512-100-20260306;hamming=experiments/nq_new/simq_data/nq_dev-simdq-simdq-hamming97m-512-100-20260306"

    # Only refresh the speed section (skip quality), or vice versa:
    python scripts/run_retrieval_experiment.py --config ... --skip-quality
    python scripts/run_retrieval_experiment.py --config ... --skip-speed --no-milvus

    # Skip individual engines. --no-milvus / --no-faiss / --no-fagin / --no-simdq
    # each drop that engine from the quality and Phase 5b tables (and, for
    # simdq, the thread-scaling sweeps too). --no-simdq still BUILDS the asym
    # index — it is the fp32 vector source the FLAT / Milvus / FAISS / Fagin
    # baselines all scan — it just skips timing and scoring simdq itself:
    python scripts/run_retrieval_experiment.py --config ... --no-simdq
"""
from __future__ import annotations

import argparse
import bz2
import csv
import json
import math
import os
import re
import shlex
import sqlite3
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

# --- BLAS threading must be pinned BEFORE numpy / faiss import ------------- #
# FAISS bundles its own OpenBLAS built against OpenMP (libgomp). That build
# sizes an internal per-thread "memory region" buffer table once, at library
# init, from OMP_NUM_THREADS (default = core count). Phase 5b then drives it
# from `workers` concurrent query threads; at D=768 that many simultaneous
# callers overrun the fixed table -> "BLAS : ... too many memory regions" ->
# segfault. A runtime threadpool_limits() / omp_set_num_threads(1) cannot
# shrink an already-allocated table, so it must be capped via the environment
# before the library loads. OPENBLAS_NUM_THREADS is left at the full core
# count so numpy's *separate* pthreads OpenBLAS (used by the Phase 1 "default
# BLAS" sweep) can still scale up at runtime via threadpool_limits.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_REL = Path(__file__).resolve().relative_to(REPO_ROOT)

from scripts.investigate_simdq_hardware import (  # noqa: E402
    detect_isa, run_sweep_phase, _load_source, _load_queries, _conc_ms_per_q,
)

import platform  # noqa: E402


# ---------- config --------------------------------------------------------- #

DEFAULT_SETTINGS = {
    "top_k": 100,
    "alpha": 10,                # K' = min(top_k * alpha, 256) for +rescore
    "speed_queries": 200,       # timed queries per speed sweep
    "warmup": 20,
    "workers": 16,              # concurrent query workers (Phase 5b + quality)
    "threads": [1, 2, 4, 8, 16, 0],
    "quality_workers": 8,       # outer thread pool for quality searches
    "milvus_uri": "http://localhost:19530",
    "run_milvus": True,
    "run_quality": True,
    "run_speed": True,
    "run_faiss": True,          # FAISS FlatIP + HNSW rows in the head-to-head
    "run_fagin": True,          # Fagin TA row in quality + head-to-head
    "run_simdq": True,          # simdq asym b=2 + 1-bit hamming variants/sweeps
                                # (the asym index dir is still built either way —
                                #  it's the fp32 vector source for the baselines)
    "run_thread_sweeps": False,  # Phase 1 / Phase 6-8 thread-scaling sweeps
                                 # (+ .sweeps.svg); opt-in via --thread-sweeps
    "fagin_batch_rows": 64,     # sorted-access rows per TA round
    "fagin_epsilon": 0.0,       # additive halting slack; 0.0 = exact
    "fagin_max_depth": 0,       # sorted-access depth cap; 0 = unlimited (exact)
    "fagin_schedule": "lockstep",  # 'lockstep' (round-robin) or 'steepest'
    "encode_batch_size": 64,
    "simdq_b": 2,               # bits/dim for the asymmetric family
    "index_root": "experiments/simdq/indexes",  # auto-built indexes live here
    "results_db": "experiments/simdq/results.db",  # persistent per-method store
    # HNSW parameters (Milvus HNSW and FAISS HNSW use the same settings)
    "hnsw_m": 16,
    "hnsw_ef_construction": 200,
    "hnsw_ef": 128,             # search-time ef, clamped to >= top_k
    "seed": 42,
}

DEFAULT_DATASET = {
    "name": "dataset",
    "queries_jsonl": None,
    # dotted paths into each jsonl row; list indices are allowed ("input.0.text")
    "text_field": "input.0.text",
    "id_field": "task_id",
    "relevant_field": "relevant",
    # id_map entries look like `docidprefix_start-end-substart-sublen`; gold
    # labels use the first N dash-separated components. 0 = use ids verbatim.
    "docid_join_components": 2,
    # corpus to encode + quantize when index dirs need to be built.
    # .jsonl / .jsonl.bz2 / .tsv (tab-separated with a header row).
    "corpus_file": None,
    "corpus_id_field": "id",
    "corpus_text_field": "text",
    "corpus_title_field": None,   # if set, encode "<title> <text>"
}


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# CLI arg name -> settings key, for scalar settings overridable from the CLI.
_CLI_SETTINGS_MAP = [
    ("top_k", "top_k"), ("alpha", "alpha"),
    ("speed_queries", "speed_queries"), ("warmup", "warmup"),
    ("workers", "workers"), ("milvus_uri", "milvus_uri"),
    ("index_root", "index_root"), ("seed", "seed"),
    ("results_db", "results_db"),
]


def _cli_overrides(args: argparse.Namespace) -> dict:
    """CLI args as dotted-path overrides, applied before Jinja2 rendering
    (so templates in the YAML see the overridden values) and again after
    the defaults merge (so they also work without a YAML)."""
    ov: dict = {}
    if args.dataset_name:
        ov["dataset.name"] = args.dataset_name
    if args.queries_jsonl:
        ov["dataset.queries_jsonl"] = args.queries_jsonl
    if args.corpus_file:
        ov["dataset.corpus_file"] = args.corpus_file
    for cli_key, cfg_key in _CLI_SETTINGS_MAP:
        v = getattr(args, cli_key)
        if v is not None:
            ov[f"settings.{cfg_key}"] = v
    if args.threads:
        ov["settings.threads"] = [int(x) for x in args.threads.split(",")]
    if args.fagin_epsilon:
        ov["settings.fagin_epsilon"] = [float(x)
                                        for x in args.fagin_epsilon.split(",")]
    if args.fagin_schedule:
        scheds = [s.strip() for s in args.fagin_schedule.split(",") if s.strip()]
        valid = {"lockstep", "steepest", "lockstep_norm", "steepest_norm"}
        bad = [s for s in scheds if s not in valid]
        if bad:
            sys.exit(f"--fagin-schedule: unknown schedule(s) {bad}; "
                     f"choose from {sorted(valid)}")
        ov["settings.fagin_schedule"] = scheds[0] if len(scheds) == 1 else scheds
    if args.no_milvus:
        ov["settings.run_milvus"] = False
    if args.no_faiss:
        ov["settings.run_faiss"] = False
    if args.no_fagin:
        ov["settings.run_fagin"] = False
    if args.no_simdq:
        ov["settings.run_simdq"] = False
    if args.thread_sweeps:
        ov["settings.run_thread_sweeps"] = True
    if args.skip_quality:
        ov["settings.run_quality"] = False
    if args.skip_speed:
        ov["settings.run_speed"] = False
    if args.out:
        ov["output"] = args.out
    return ov


def load_config(args: argparse.Namespace) -> dict:
    cfg: dict = {"dataset": dict(DEFAULT_DATASET),
                 "settings": dict(DEFAULT_SETTINGS),
                 "models": [], "output": None}
    overrides = _cli_overrides(args)
    if args.config:
        # DocUVerse config reader: YAML/JSON + dotted-path overrides +
        # Jinja2 rendering ({{var}} referencing other config keys, filters
        # like short_model, multi-pass resolution).
        from docuverse.utils import read_config_file
        raw = read_config_file(str(_resolve(args.config)), overrides) or {}
        cfg["dataset"].update(raw.get("dataset") or {})
        cfg["settings"].update(raw.get("settings") or {})
        cfg["models"] = list(raw.get("models") or [])
        cfg["output"] = raw.get("output")

    # CLI --model entries are appended (or replace, if no YAML models).
    for spec in args.model or []:
        m: dict = {}
        for kv in spec.split(";"):
            kv = kv.strip()
            if not kv:
                continue
            k, _, v = kv.partition("=")
            m[k.strip()] = v.strip()
        # normalize CLI shorthand keys
        if "asym" in m:
            m["asym_index"] = m.pop("asym")
        if "hamming" in m:
            m["hamming_index"] = m.pop("hamming")
        cfg["models"].append(m)

    # Re-apply the CLI overrides on top of the defaults merge — a no-op for
    # values already applied inside read_config_file, and the only
    # application in --config-less mode.
    for dotted, v in overrides.items():
        parts = dotted.split(".")
        target = cfg
        for p in parts[:-1]:
            target = target[p]
        target[parts[-1]] = v

    if not cfg["models"]:
        sys.exit("no models configured — pass --config or --model")
    for m in cfg["models"]:
        if "tag" not in m:
            sys.exit(f"model entry missing required key 'tag': {m}")
    return cfg


# ---------- dataset / encoding --------------------------------------------- #

def _get_dotted(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def load_dataset(ds_cfg: dict) -> tuple[list[str], list[str], list[list[str]]]:
    """Read the queries jsonl → (texts, qids, relevant doc-id lists)."""
    path = _resolve(ds_cfg["queries_jsonl"])
    texts, qids, relevant = [], [], []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            texts.append(str(_get_dotted(row, ds_cfg["text_field"])))
            qids.append(str(_get_dotted(row, ds_cfg["id_field"])))
            try:
                rel = _get_dotted(row, ds_cfg["relevant_field"]) or []
            except (KeyError, IndexError, TypeError):
                rel = []
            relevant.append([str(r) for r in rel])
    print(f"# dataset: {len(texts)} queries from {path}")
    return texts, qids, relevant


def _slug(tag: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in tag).strip("_")


def _encode_texts(model_cfg: dict, texts: list[str], settings: dict,
                  what: str) -> np.ndarray:
    """Encode texts with the model's sentence-transformers encoder."""
    encoder = model_cfg.get("encoder")
    if not encoder:
        sys.exit(f"model '{model_cfg['tag']}': no 'encoder' configured, "
                 f"but the {what} vectors need to be computed")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# encoding {len(texts)} {what} texts with {encoder} on {device}")
    # local dir (absolute or repo-relative) beats a HF hub id of the same name
    enc_path = _resolve(encoder)
    model = SentenceTransformer(str(enc_path) if enc_path.exists() else encoder,
                                device=device)
    model.max_seq_length = int(model_cfg.get("max_seq_length", 512))
    emb = model.encode(texts, batch_size=int(settings["encode_batch_size"]),
                       show_progress_bar=True, normalize_embeddings=True,
                       convert_to_numpy=True).astype(np.float32, copy=False)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.ascontiguousarray(emb)


def ensure_query_npy(model_cfg: dict, ds_name: str, texts: list[str],
                     settings: dict, force: bool) -> Path:
    """Return the path of this model's encoded-queries .npy, encoding if needed."""
    default = (REPO_ROOT / "scratch" / "simdq_exp" /
               f"{ds_name}_{_slug(model_cfg['tag'])}.npy")
    out = _resolve(model_cfg.get("queries_npy") or default)
    if out.exists() and not force:
        print(f"# queries[{model_cfg['tag']}]: reusing {out}")
        return out
    emb = _encode_texts(model_cfg, texts, settings, "query")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    print(f"#   wrote {out}  shape={emb.shape}")
    return out


# ---------- corpus / index building ----------------------------------------- #

def load_corpus(ds_cfg: dict) -> tuple[list[str], list[str]]:
    """Read dataset.corpus_file → (ids, texts). jsonl(.bz2) or tsv."""
    path = _resolve(ds_cfg["corpus_file"])
    id_f, text_f = ds_cfg["corpus_id_field"], ds_cfg["corpus_text_field"]
    title_f = ds_cfg.get("corpus_title_field")
    ids: list[str] = []
    texts: list[str] = []

    def add(row, i):
        try:
            doc_id = str(_get_dotted(row, id_f))
        except (KeyError, IndexError, TypeError):
            doc_id = f"doc_{i}"
        text = str(_get_dotted(row, text_f))
        if title_f:
            try:
                title = str(_get_dotted(row, title_f) or "")
            except (KeyError, IndexError, TypeError):
                title = ""
            if title:
                text = f"{title} {text}"
        ids.append(doc_id)
        texts.append(text)

    if ".tsv" in path.suffixes or path.suffix == ".tsv":
        with path.open(newline="") as f:
            for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
                add(row, i)
    else:
        opener = bz2.open if path.suffix == ".bz2" else open
        with opener(path, "rt") as f:
            for i, line in enumerate(f):
                if line.strip():
                    add(json.loads(line), i)
    print(f"# corpus: {len(ids)} documents from {path}")
    return ids, texts


def ensure_indexes(model_cfg: dict, ds_cfg: dict, settings: dict,
                   corpus_cache: dict, force_build: bool) -> None:
    """Fill in model_cfg['asym_index'/'hamming_index'], building them if needed.

    A directory counts as built when its meta.json exists. Freshly built
    indexes encode dataset.corpus_file rows as-is (no chunking — unlike the
    DocUVerse ingestion pipeline) and get an engine_metadata.json whose
    id_map is the corpus ids verbatim.
    """
    from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
    tag = model_cfg["tag"]
    slug = _slug(tag)
    ds_name = ds_cfg["name"]
    b = int(settings["simdq_b"])
    root = _resolve(settings["index_root"])
    plans = [
        ("asym_index", "asymmetric", b, root / f"{ds_name}_{slug}_asym_b{b}"),
        ("hamming_index", "hamming", None, root / f"{ds_name}_{slug}_hamming"),
    ]
    todo = []
    for key, family, fam_b, default_dir in plans:
        d = _resolve(model_cfg.get(key) or default_dir)
        model_cfg[key] = str(d)
        if force_build or not (d / "meta.json").exists():
            todo.append((key, family, fam_b, d))
        else:
            print(f"# index[{tag}/{family}]: reusing {d}")
    if not todo:
        return

    if not ds_cfg.get("corpus_file"):
        missing = ", ".join(str(d) for _, _, _, d in todo)
        sys.exit(f"model '{tag}': index dir(s) {missing} don't exist and "
                 f"dataset.corpus_file is not set — point asym_index/"
                 f"hamming_index at prebuilt dirs or configure a corpus")
    if "ids" not in corpus_cache:
        corpus_cache["ids"], corpus_cache["texts"] = load_corpus(ds_cfg)
    ids, texts = corpus_cache["ids"], corpus_cache["texts"]

    # per-model corpus vectors, cached next to the query cache
    default_npy = (REPO_ROOT / "scratch" / "simdq_exp" /
                   f"{ds_name}_{slug}_corpus.npy")
    npy = _resolve(model_cfg.get("corpus_npy") or default_npy)
    if npy.exists() and not force_build:
        vecs = np.ascontiguousarray(np.load(npy).astype(np.float32, copy=False))
        print(f"# corpus vecs[{tag}]: reusing {npy}  shape={vecs.shape}")
    else:
        vecs = _encode_texts(model_cfg, texts, settings, "corpus")
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, vecs)
        print(f"#   wrote {npy}  shape={vecs.shape}")
    if vecs.shape[0] != len(ids):
        sys.exit(f"model '{tag}': corpus vectors {npy} have {vecs.shape[0]} "
                 f"rows but corpus_file has {len(ids)} docs — stale cache? "
                 f"re-run with --force-build")

    for key, family, fam_b, d in todo:
        print(f"# building {family} index for {tag} -> {d}")
        idx = SimdqIndex.build(vecs, family=family, b=fam_b, d=None,
                               projection="identity", store_floats=True,
                               encoder_id=model_cfg.get("encoder"))
        d.parent.mkdir(parents=True, exist_ok=True)
        idx.save(d)
        (d / "engine_metadata.json").write_text(
            json.dumps({"id_map": ids, "metadata": {}}))
        print(f"#   saved {d}  (N={vecs.shape[0]}, D={vecs.shape[1]})")


# ---------- quality --------------------------------------------------------- #

def _base_docid(chunk_id: str, join_components: int) -> str:
    if join_components <= 0:
        return chunk_id
    parts = chunk_id.split("-")
    return ("-".join(parts[:join_components])
            if len(parts) >= join_components else chunk_id)


def _search_batch(idx, qs, K, K_prime, workers) -> np.ndarray:
    def one(q):
        idxs, _ = idx.search(q, K=K, K_prime=K_prime, num_threads=1)
        return idxs
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(one, qs))
    return np.stack(out, axis=0)


def _flat_topk(vecs_f32: np.ndarray, qs: np.ndarray, K: int,
               chunk_q: int = 64) -> np.ndarray:
    out = np.empty((qs.shape[0], K), dtype=np.int64)
    for s in range(0, qs.shape[0], chunk_q):
        e = min(s + chunk_q, qs.shape[0])
        scores = qs[s:e] @ vecs_f32.T
        part = np.argpartition(-scores, K - 1, axis=1)[:, :K]
        rows = np.arange(e - s)[:, None]
        order = np.argsort(-scores[rows, part], axis=1)
        out[s:e] = part[rows, order]
    return out


def _metrics(retrieved_ids: np.ndarray, id_map: list[str],
             relevant: list[list[str]], join_components: int) -> dict:
    """Recall@10 / Recall@K (K = retrieved width), MRR@10, nDCG@10."""
    N_q = retrieved_ids.shape[0]
    r10 = rK = mrr = ndcg = 0.0
    n_scored = 0
    for i in range(N_q):
        rel = set(relevant[i])
        if not rel:
            continue
        n_scored += 1
        hit10: set[str] = set()
        hitK: set[str] = set()
        found_rank = None
        dcg = 0.0
        for rank, ci in enumerate(retrieved_ids[i]):
            ci = int(ci)
            if ci < 0:  # padding for engines that returned fewer than K hits
                continue
            base = _base_docid(id_map[ci], join_components)
            if base in rel:
                if rank < 10:
                    if base not in hit10:
                        dcg += 1.0 / math.log2(rank + 2)
                    hit10.add(base)
                    if found_rank is None:
                        found_rank = rank
                hitK.add(base)
        n_rel = len(rel)
        r10 += len(hit10) / n_rel
        rK += len(hitK) / n_rel
        if found_rank is not None:
            mrr += 1.0 / (found_rank + 1)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(n_rel, 10)))
        ndcg += (dcg / idcg) if idcg > 0 else 0.0
    return {"recall@10": r10 / n_scored, "recall@K": rK / n_scored,
            "MRR@10": mrr / n_scored, "nDCG@10": ndcg / n_scored,
            "n_queries": n_scored}


def run_quality(models: list[dict], query_npys: list[Path],
                relevant: list[list[str]], settings: dict,
                join_components: int) -> tuple[list[dict], list[dict]]:
    """Returns (metric_rows, agree_rows) across all models × variants."""
    from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
    K = int(settings["top_k"])
    K_re = max(K, min(K * int(settings["alpha"]), 256))
    workers = int(settings["quality_workers"])
    metric_rows: list[dict] = []
    agree_rows: list[dict] = []
    for m, q_npy in zip(models, query_npys):
        tag = m["tag"]
        asym_dir = _resolve(m["asym_index"])
        ham_dir = _resolve(m["hamming_index"])
        print(f"\n## quality — {tag}  (asym={asym_dir.name}, ham={ham_dir.name})")
        qs = np.ascontiguousarray(np.load(q_npy).astype(np.float32, copy=False))
        id_map = json.loads(
            (asym_dir / "engine_metadata.json").read_text())["id_map"]
        id_map_h = json.loads(
            (ham_dir / "engine_metadata.json").read_text())["id_map"]
        assert id_map == id_map_h, f"{tag}: asym / hamming id_maps diverge"
        if qs.shape[0] != len(relevant):
            sys.exit(f"{tag}: {qs.shape[0]} query vectors vs "
                     f"{len(relevant)} gold rows — stale queries_npy? "
                     f"re-run with --force-encode")

        idx_a = SimdqIndex.load(asym_dir)
        idx_h = SimdqIndex.load(ham_dir)
        assert idx_a.has_floats and idx_h.has_floats, f"{tag}: floats.bin missing"
        vecs_f32 = np.asarray(idx_a.floats_mmap, dtype=np.float32)

        ret_flat = _flat_topk(vecs_f32, qs, K=K)
        flat10 = [set(map(int, ret_flat[i, :10])) for i in range(qs.shape[0])]
        flatK = [set(map(int, ret_flat[i])) for i in range(qs.shape[0])]

        def record(label: str, ret: np.ndarray) -> None:
            mm = _metrics(ret, id_map, relevant, join_components)
            metric_rows.append({"encoder": tag, "variant": label, **mm})
            print(f"  {label:<24s} R@10={mm['recall@10']:.4f}  "
                  f"R@{K}={mm['recall@K']:.4f}  MRR@10={mm['MRR@10']:.4f}  "
                  f"nDCG@10={mm['nDCG@10']:.4f}")
            a10 = np.mean([len(set(map(int, ret[i, :10])) & flat10[i]) / 10
                           for i in range(qs.shape[0])])
            aK = np.mean([len(set(map(int, ret[i])) & flatK[i]) / K
                          for i in range(qs.shape[0])])
            agree_rows.append({"encoder": tag, "variant": label,
                               "agree@10": float(a10), "agree@K": float(aK)})
            print(f"  {label:<24s} vs FLAT: agree@10={a10:.4f}  agree@{K}={aK:.4f}")

        if settings.get("run_simdq", True):
            variants = [("asym b=2, +rescore", idx_a, K_re),
                        ("asym b=2, no rescore", idx_a, K),
                        ("1-bit ham, +rescore", idx_h, K_re),
                        ("1-bit ham, no rescore", idx_h, K)]
            for label, idx, kp in variants:
                record(label, _search_batch(idx, qs, K=K, K_prime=kp,
                                            workers=workers))

        # Milvus / FAISS / Fagin baselines on the same fp32 vectors + queries,
        # so every system in the Phase 5b head-to-head also gets a quality row.
        for name, call in iter_baseline_engines(vecs_f32, settings, K):
            try:
                record(name, _call_batch(call, qs, K, workers))
            except Exception as e:
                print(f"  {name} skipped: {e}", file=sys.stderr)

        mm = _metrics(ret_flat, id_map, relevant, join_components)
        metric_rows.append({"encoder": tag,
                            "variant": "FLAT (fp32 IP, exact)", **mm})
        print(f"  {'FLAT (fp32 IP, exact)':<24s} R@10={mm['recall@10']:.4f}  "
              f"R@{K}={mm['recall@K']:.4f}  MRR@10={mm['MRR@10']:.4f}  "
              f"nDCG@10={mm['nDCG@10']:.4f}")
    return metric_rows, agree_rows


# ---------- head-to-head: Milvus / FAISS / simdq ---------------------------- #

def iter_baseline_engines(vecs: np.ndarray, st: dict, K: int):
    """Yield (name, search_call) for the enabled Milvus / FAISS / Fagin
    baselines.

    search_call(q) returns K corpus row indices. Shared by the quality phase
    and the Phase 5b speed benchmark so both report the same systems. Engines
    that fail to set up print a "skipped" note instead of raising; the Milvus
    collection is dropped once the generator moves past its engines (or is
    closed early).
    """
    import os
    import time

    N, D = vecs.shape
    hnsw_m = int(st["hnsw_m"])
    hnsw_efc = int(st["hnsw_ef_construction"])
    hnsw_ef = max(int(st["hnsw_ef"]), K)

    # --- Milvus: FLAT then HNSW on the same collection (insert once) ---
    if st["run_milvus"]:
        cli = coll = None
        try:
            from pymilvus import MilvusClient, DataType
            cli = MilvusClient(uri=st["milvus_uri"])
            coll = f"hwbench_{D}_{N}"
            if cli.has_collection(coll):
                cli.drop_collection(coll)
            schema = cli.create_schema(auto_id=False,
                                       enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("v", DataType.FLOAT_VECTOR, dim=D)
            cli.create_collection(coll, schema=schema)
            for s in range(0, N, 20000):
                e = min(s + 20000, N)
                cli.insert(coll, [{"id": i, "v": vecs[i]}
                                  for i in range(s, e)])
            for name, index_params, sp in [
                    ("Milvus FLAT (exact)",
                     {"index_type": "FLAT", "metric_type": "IP"},
                     {"metric_type": "IP"}),
                    (f"Milvus HNSW (M={hnsw_m}, ef={hnsw_ef})",
                     {"index_type": "HNSW", "metric_type": "IP",
                      "params": {"M": hnsw_m,
                                 "efConstruction": hnsw_efc}},
                     {"metric_type": "IP",
                      "params": {"ef": hnsw_ef}})]:
                ip = cli.prepare_index_params()
                ip.add_index(field_name="v", **index_params)
                cli.create_index(coll, ip)
                cli.load_collection(coll)
                time.sleep(2)
                yield name, (lambda q, sp=sp: [
                    h["id"] for h in cli.search(
                        coll, [q.tolist()], limit=K,
                        search_params=sp, anns_field="v")[0]])
                cli.release_collection(coll)
                cli.drop_index(coll, "v")
        except Exception as e:
            print(f"    Milvus skipped: {e}", file=sys.stderr)
        finally:
            if cli is not None and coll is not None:
                try:
                    cli.drop_collection(coll)
                except Exception:
                    pass

    # --- FAISS: FlatIP then HNSW, in-process (no RPC) ---
    if st["run_faiss"]:
        try:
            import faiss
            faiss.omp_set_num_threads(os.cpu_count() or 1)  # fast build
            fl = faiss.IndexFlatIP(D)
            fl.add(vecs)
            hn = faiss.IndexHNSWFlat(D, hnsw_m,
                                     faiss.METRIC_INNER_PRODUCT)
            hn.hnsw.efConstruction = hnsw_efc
            hn.add(vecs)
            hn.hnsw.efSearch = hnsw_ef
            faiss.omp_set_num_threads(1)  # single-threaded per query
        except Exception as e:
            print(f"    FAISS skipped: {e}", file=sys.stderr)
        else:
            yield ("FAISS FlatIP (exact)",
                   lambda q: fl.search(q[None, :], K)[1][0])
            yield (f"FAISS HNSW (M={hnsw_m}, ef={hnsw_ef})",
                   lambda q: hn.search(q[None, :], K)[1][0])

    # --- Fagin TA: in-process Threshold Algorithm over per-dim sorted lists ---
    # fagin_epsilon may be a single value or a list (epsilon sweep): the index
    # is built once and yielded once per epsilon, so both the quality phase and
    # the Phase 5b speed benchmark get one (sequentially timed) row per epsilon.
    if st.get("run_fagin", True):
        f_batch = int(st["fagin_batch_rows"])
        f_depth = int(st["fagin_max_depth"])
        sched_cfg = st.get("fagin_schedule", "lockstep")
        f_schedules = (list(sched_cfg)
                       if isinstance(sched_cfg, (list, tuple))
                       else [sched_cfg])
        eps_cfg = st["fagin_epsilon"]
        f_epsilons = ([float(e) for e in eps_cfg]
                      if isinstance(eps_cfg, (list, tuple))
                      else [float(eps_cfg)])
        try:
            from docuverse.engines.retrieval.fagin.fagin_index import FaginIndex
            fg = FaginIndex.build(vecs)   # in-memory, same fp32 vectors (once)
        except Exception as e:
            print(f"    Fagin skipped: {e}", file=sys.stderr)
        else:
            # TA = lockstep (round-robin) Threshold Algorithm; TASD = the
            # steepest-descent schedule (advance the dim with the largest
            # marginal threshold drop). GTA/GTASD = the norm-aware ("gradient")
            # variants using the water-filling halting bound that exploits
            # ||x||_2 = 1. Each configured schedule × epsilon gets its own row,
            # so a single run can compare TA vs GTA (or the full four) directly.
            algo_of = {"lockstep": "TA", "steepest": "TASD",
                       "lockstep_norm": "GTA", "steepest_norm": "GTASD"}
            for f_schedule in f_schedules:
                algo = algo_of.get(f_schedule, "TA")
                for f_eps in f_epsilons:
                    name = (f"Fagin {algo} (exact)"
                            if f_eps == 0.0 and f_depth == 0
                            else f"Fagin {algo} (eps={f_eps:g}, depth={f_depth})")
                    yield name, (lambda q, f_eps=f_eps, f_schedule=f_schedule:
                                 fg.search(
                                     q, K=K, batch=f_batch, epsilon=f_eps,
                                     max_depth=f_depth, num_threads=1,
                                     schedule=f_schedule)[0])


def _call_batch(call, qs: np.ndarray, K: int, workers: int) -> np.ndarray:
    """Run call(q) -> ids over all queries; pad short results with -1."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rets = list(ex.map(call, qs))
    out = np.full((len(rets), K), -1, dtype=np.int64)
    for i, r in enumerate(rets):
        r = np.asarray(list(r), dtype=np.int64)[:K]
        out[i, :r.shape[0]] = r
    return out


def bench_head_to_head(sources, query_specs, st) -> list[dict]:
    """Concurrent ms/query + agreement-vs-exact for every engine per source.

    Systems: Milvus FLAT, Milvus HNSW, FAISS FlatIP, FAISS HNSW, Fagin TA,
    and simdq asym-b2 / 1-bit hamming (each ±rescore). Same harness for all:
    `workers`
    concurrent query threads, each engine call single-threaded (matches
    `scripts/bench_simdq_vs_milvus.py`). agree@K is overlap with the exact
    fp32 top-K on the timed queries — 1.0 for exact engines by construction,
    the recall/latency trade-off knob for HNSW and codes-only simdq.
    """
    from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
    from threadpoolctl import threadpool_limits

    K = int(st["top_k"])
    K_re = max(K, min(K * int(st["alpha"]), 256))
    warmup, nq = int(st["warmup"]), int(st["speed_queries"])
    workers = int(st["workers"])
    hnsw_m = int(st["hnsw_m"])
    hnsw_efc = int(st["hnsw_ef_construction"])
    hnsw_ef = max(int(st["hnsw_ef"]), K)
    rng = np.random.default_rng(int(st["seed"]))
    print(f"\n## Phase 5b — head-to-head Milvus / FAISS / Fagin / simdq  "
          f"(workers={workers}, HNSW M={hnsw_m}, efC={hnsw_efc}, ef={hnsw_ef})")

    out: list[dict] = []
    # Pin OpenBLAS to 1 thread for the whole head-to-head: the concurrency is
    # meant to come from the `workers` query threads, with each engine call
    # single-threaded. Without this, every worker's numpy matmul (_flat_topk's
    # `qs @ vecs.T`, simdq's `cand_floats @ q_proj`) fans out across OpenBLAS's
    # own thread pool; workers × that nesting exhausts OpenBLAS's internal
    # buffer table ("too many memory regions" → segfault) on the larger dims.
    # user_api="blas" leaves FAISS's OpenMP build parallelism untouched.
    with threadpool_limits(limits=1, user_api="blas"):
        for (label, vecs), q_spec in zip(sources, query_specs):
            N, D = vecs.shape
            qs_pool = _load_queries(q_spec, D, warmup + nq, rng)
            qs = qs_pool[warmup:warmup + nq]
            flat_ids = _flat_topk(vecs, qs, K)
            flat_sets = [set(map(int, flat_ids[i])) for i in range(nq)]
            systems: list[dict] = []
            print(f"  ### {label}  (D={D}, N={N}, "
                  f"q={'real' if q_spec else 'synth'})")

            def run_system(name: str, call) -> None:
                """call(q) -> iterable of K corpus ids (row indices)."""
                for i in range(warmup):
                    call(qs_pool[i])
                ms = _conc_ms_per_q(call, qs, workers)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    rets = list(ex.map(call, qs))
                agree = float(np.mean(
                    [len(set(map(int, r)) & flat_sets[i]) / K
                     for i, r in enumerate(rets)]))
                systems.append({"name": name, "ms": ms, "agree": agree})
                print(f"    {name:<38s} {ms:8.2f} ms/q   agree@{K}={agree:.4f}")

            # --- Milvus / FAISS / Fagin baselines (shared with quality phase) ---
            for name, call in iter_baseline_engines(vecs, st, K):
                try:
                    run_system(name, call)
                except Exception as e:
                    print(f"    {name} skipped: {e}", file=sys.stderr)

            # --- simdq: asym b=2 and 1-bit hamming, each ±rescore ---
            if st.get("run_simdq", True):
                for family, b, fam_label in [
                        ("asymmetric", int(st["simdq_b"]), None),
                        ("hamming", None, "1-bit ham")]:
                    fam_label = fam_label or f"asym b={b}"
                    idx = SimdqIndex.build(vecs, family=family, b=b, d=None,
                                           projection="identity",
                                           store_floats=True)
                    for mode, kp in [("+rescore", K_re), ("no rescore", K)]:
                        run_system(f"simdq {fam_label} ({mode})",
                                   lambda q, kp=kp: idx.search(
                                       q, K=K, K_prime=kp, num_threads=1)[0])
                    del idx

            out.append({"label": label, "D": int(D), "N": int(N),
                        "queries": ("real" if q_spec else "synth"),
                        "systems": systems})
    return out


# ---------- report ---------------------------------------------------------- #

def _sweep_md_table(rows, threads, mode):
    hdr = "| source | dim | " + " | ".join(
        f"t={t if t else 'all'}" for t in threads) + " |"
    sep = "|---" * (len(threads) + 2) + "|"
    body = [f"| {lbl} | {D} | "
            + " | ".join(f"{out[mode][t]:.2f}" for t in threads) + " |"
            for lbl, D, out in rows]
    return "\n".join([hdr, sep] + body)


def _sweep_md_pair(rows, threads, K, K_re):
    return (f"**With fp32 rescore** (`K'={K_re}`, gather + "
            f"`cand_floats @ q_proj` + argsort):\n\n"
            + _sweep_md_table(rows, threads, "rescore") + "\n\n"
            + f"**No rescore** (`K'=K={K}`, codes-only ranking — Hamming "
            f"distance / b=2 scaled scores):\n\n"
            + _sweep_md_table(rows, threads, "no_rescore"))


# Categorical slots 1-5 of the dataviz reference palette, (light, dark) —
# validated (lightness band, chroma, CVD separation, contrast) on the
# matching surfaces with the skill's validate_palette.js. Sub-3:1 light
# slots and the dark 8-12 CVD floor band require direct labels or a table
# view alongside — both charts provide them.
_CHART_COLORS = [("#2a78d6", "#3987e5"),   # blue
                 ("#1baf7a", "#199e70"),   # aqua
                 ("#eda100", "#c98500"),   # yellow
                 ("#008300", "#008300"),   # green
                 ("#4a3aa7", "#9085e9")]   # violet


def _chart_css(n_colors: int) -> str:
    """Shared chart chrome: ink/grid/surface tokens + the first n categorical
    slots, light + dark via prefers-color-scheme (works in <img>/GitHub)."""
    light = " ".join(f"--c{i}:{c[0]};" for i, c in
                     enumerate(_CHART_COLORS[:n_colors]))
    dark = " ".join(f"--c{i}:{c[1]};" for i, c in
                    enumerate(_CHART_COLORS[:n_colors]))
    return "\n".join([
        "<style>",
        'svg{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;}',
        ":root{--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;"
        "--ink-3:#898781;--grid:#e1e0d9;--axis:#c3c2b7;" + light + "}",
        "@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;"
        "--ink:#ffffff;--ink-2:#c3c2b7;--ink-3:#898781;--grid:#2c2c2a;"
        "--axis:#383835;" + dark + "}}",
        "</style>"])


def _log_ticks(lo: float, hi: float, max_ticks: int = 9) -> tuple:
    """(ymin, ymax, ticks) on a 1-2-5 log ladder enclosing [lo, hi]."""
    cands = [(m * 10.0 ** e, m)
             for e in range(int(math.floor(math.log10(lo))) - 1,
                            int(math.ceil(math.log10(hi))) + 1)
             for m in (1, 2, 5)]
    ymin = max((v for v, _ in cands if v <= lo * (1 + 1e-9)), default=lo)
    ymax = min((v for v, _ in cands if v >= hi * (1 - 1e-9)), default=hi)
    ticks = [(v, m) for v, m in cands if ymin <= v <= ymax]
    if len(ticks) > max_ticks:  # too dense: keep decades only
        ticks = [(v, m) for v, m in ticks if m == 1] or ticks
    return ymin, ymax, [v for v, _ in ticks]


def _marker_svg(shape: int, x: float, y: float, fill: str) -> str:
    """Series marker with a 2px surface ring (shape encodes the model)."""
    ring = 'stroke="var(--surface)" stroke-width="2"'
    if shape % 4 == 0:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{fill}" {ring}/>'
    if shape % 4 == 1:
        return (f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
                f'fill="{fill}" {ring}/>')
    if shape % 4 == 2:
        return (f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
                f'transform="rotate(45 {x:.1f} {y:.1f})" fill="{fill}" {ring}/>')
    return (f'<polygon points="{x:.1f},{y - 5:.1f} {x + 5:.1f},{y + 4:.1f} '
            f'{x - 5:.1f},{y + 4:.1f}" fill="{fill}" {ring}/>')


def _sweep_chart_svg(sweeps, threads, tags, K, K_re) -> str:
    """One line chart for all thread sweeps: median ms/query (log y) vs
    num_threads. Hue = sweep config, marker shape = model/index, solid =
    +rescore / dashed = codes-only. Light + dark mode via
    prefers-color-scheme inside the SVG (works in <img>/GitHub markdown).
    """
    n_t = len(threads)
    vals = [out[mode][t]
            for _, rows in sweeps for _, _, out in rows
            for mode in ("rescore", "no_rescore") for t in threads]
    lo, hi = min(vals), max(vals)
    ymin, ymax, ticks = _log_ticks(lo, hi)

    W, H = 920, 448
    L, R = 56, 24
    p_top, p_bot = 122, 396
    xs = [L + i * (W - L - R) / max(n_t - 1, 1) for i in range(n_t)]
    ly0, ly1 = math.log10(ymin), math.log10(ymax)
    if ly1 - ly0 < 1e-9:
        ly0, ly1 = ly0 - 0.5, ly1 + 0.5

    def Y(v: float) -> float:
        return p_bot - (math.log10(v) - ly0) / (ly1 - ly0) * (p_bot - p_top)

    e: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" aria-label="Median search '
        f'latency (ms/query, log scale) vs number of threads">',
        _chart_css(len(sweeps)),
        f'<rect width="{W}" height="{H}" rx="8" fill="var(--surface)"/>',
        f'<text x="{L}" y="26" font-size="15" font-weight="600" '
        f'fill="var(--ink)">SimdqIndex.search — median latency vs. '
        f'threads</text>',
        f'<text x="{L}" y="46" font-size="12" fill="var(--ink-2)">'
        f'ms/query, log scale — lower is better</text>',
    ]

    # legend row 1: hue = sweep config; row 2: marker = model, dash = mode
    def txt_w(s: str) -> float:
        return 6.1 * len(s)

    x = float(L)
    for ci, (cfg_label, _) in enumerate(sweeps):
        e.append(f'<line x1="{x:.0f}" y1="66" x2="{x + 20:.0f}" y2="66" '
                 f'stroke="var(--c{ci})" stroke-width="2.5" '
                 f'stroke-linecap="round"/>')
        e.append(f'<text x="{x + 26:.0f}" y="70" font-size="12" '
                 f'fill="var(--ink-2)">{cfg_label}</text>')
        x += 26 + txt_w(cfg_label) + 22
    x = float(L)
    for si, tag in enumerate(tags):
        e.append(_marker_svg(si, x + 5, 88, "var(--ink-3)"))
        e.append(f'<text x="{x + 16:.0f}" y="92" font-size="12" '
                 f'fill="var(--ink-2)">{tag}</text>')
        x += 16 + txt_w(tag) + 22
    for dash, lbl in (("", f"+rescore (K′={K_re})"),
                      (' stroke-dasharray="5 4"', f"no rescore (K′=K={K})")):
        e.append(f'<line x1="{x:.0f}" y1="88" x2="{x + 20:.0f}" y2="88" '
                 f'stroke="var(--ink-2)" stroke-width="2"{dash} '
                 f'stroke-linecap="round"/>')
        e.append(f'<text x="{x + 26:.0f}" y="92" font-size="12" '
                 f'fill="var(--ink-2)">{lbl}</text>')
        x += 26 + txt_w(lbl) + 22

    # grid + axes (hairline, recessive)
    e.append(f'<text x="8" y="{p_top - 10}" font-size="11" '
             f'fill="var(--ink-3)">ms/query</text>')
    for v in ticks:
        yy = Y(v)
        e.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{W - R}" y2="{yy:.1f}" '
                 f'stroke="var(--grid)" stroke-width="1"/>')
        e.append(f'<text x="{L - 8}" y="{yy + 4:.1f}" font-size="11" '
                 f'text-anchor="end" fill="var(--ink-3)">{v:g}</text>')
    e.append(f'<line x1="{L}" y1="{p_bot}" x2="{W - R}" y2="{p_bot}" '
             f'stroke="var(--axis)" stroke-width="1"/>')
    for i, t in enumerate(threads):
        e.append(f'<text x="{xs[i]:.1f}" y="{p_bot + 20}" font-size="11" '
                 f'text-anchor="middle" fill="var(--ink-3)">'
                 f'{t if t else "all"}</text>')
    e.append(f'<text x="{L + (W - L - R) / 2:.0f}" y="{p_bot + 40}" '
             f'font-size="11" text-anchor="middle" fill="var(--ink-3)">'
             f'num_threads</text>')

    # data: lines first, markers on top (rings keep crossings legible)
    marks: list[str] = []
    for ci, (_, rows) in enumerate(sweeps):
        for si, (_, _, out) in enumerate(rows):
            for mode, dash in (("rescore", ""),
                               ("no_rescore", ' stroke-dasharray="5 4"')):
                pts = " ".join(f"{xs[i]:.1f},{Y(out[mode][t]):.1f}"
                               for i, t in enumerate(threads))
                e.append(f'<polyline points="{pts}" fill="none" '
                         f'stroke="var(--c{ci})" stroke-width="2" '
                         f'stroke-linejoin="round" '
                         f'stroke-linecap="round"{dash}/>')
                marks += [_marker_svg(si, xs[i], Y(out[mode][t]),
                                      f"var(--c{ci})")
                          for i, t in enumerate(threads)]
    e += marks
    e.append("</svg>")
    return "\n".join(e) + "\n"


_ENGINE_FAMILIES = ["Milvus", "FAISS", "simdq asym b=2", "simdq 1-bit ham",
                    "Fagin"]


def _frontier_chart_svg(models, metric_rows, head, workers) -> str | None:
    """Per-model scatter of throughput (queries/s, log y) vs nDCG@10 for
    every Phase 5b engine that also has a quality row. Small multiples with
    shared axes; hue = engine family; every point direct-labeled (the
    identity channel the sub-3:1 hues and dark-mode CVD floor require).
    Returns None when nothing joins (e.g. quality or speed was skipped).
    """
    by_ev = {(r["encoder"], r["variant"]): r for r in metric_rows}

    def family_of(name: str) -> int:
        if name.startswith("Milvus"):
            return 0
        if name.startswith("FAISS"):
            return 1
        if name.startswith("Fagin"):
            return 4
        return 2 if name.startswith("simdq asym") else 3

    def quality_of(tag: str, name: str):
        simdq_var = _simdq_name_to_variant(name)
        if simdq_var is not None:
            return by_ev.get((tag, simdq_var))
        # baselines: exact-name quality row; exact engines fall back to the
        # FLAT ceiling (same result by construction)
        row = by_ev.get((tag, name))
        if row is None and "(exact)" in name:
            row = by_ev.get((tag, "FLAT (fp32 IP, exact)"))
        return row

    panels = []
    for m, hrow in zip(models, head):
        pts = []
        for s in hrow["systems"]:
            q = quality_of(m["tag"], s["name"])
            if not q or s["ms"] <= 0:
                continue
            label = s["name"].split(" (")[0]
            if s["name"].startswith("simdq "):
                mode = "+rescore" if "+rescore" in s["name"] else "no rescore"
                label = f"{label[6:]} {mode}"
            pts.append((label, family_of(s["name"]),
                        float(q["nDCG@10"]), 1000.0 / float(s["ms"])))
        if pts:
            panels.append((m["tag"], pts))
    if not panels:
        return None

    all_x = [x for _, pts in panels for _, _, x, _ in pts]
    all_y = [y for _, pts in panels for _, _, _, y in pts]
    pad = max((max(all_x) - min(all_x)) * 0.08, 0.004)
    x0, x1 = min(all_x) - pad, max(all_x) + pad
    ymin, ymax, yticks = _log_ticks(min(all_y), max(all_y))

    # linear x ticks on a 1/2/2.5/5 step, ~4 intervals
    raw = (x1 - x0) / 4
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    dec = next(d for d in range(6) if abs(round(step, d) - step) < step * 1e-6)
    xticks = []
    v = math.ceil(x0 / step) * step
    while v <= x1 + step * 1e-6:
        xticks.append(v)
        v += step

    n = len(panels)
    L, PW, G, RP = 64, 390, 36, 16
    W = L + n * PW + (n - 1) * G + RP
    p_top, p_bot = 118, 360
    H = 412
    ly0, ly1 = math.log10(ymin), math.log10(ymax)
    if ly1 - ly0 < 1e-9:
        ly0, ly1 = ly0 - 0.5, ly1 + 0.5

    def Y(v: float) -> float:
        return p_bot - (math.log10(v) - ly0) / (ly1 - ly0) * (p_bot - p_top)

    e: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" aria-label="Throughput '
        f'(queries per second, log scale) vs nDCG@10 for each engine, one '
        f'panel per model">',
        _chart_css(len(_ENGINE_FAMILIES)),
        f'<rect width="{W}" height="{H}" rx="8" fill="var(--surface)"/>',
        f'<text x="{L}" y="26" font-size="15" font-weight="600" '
        f'fill="var(--ink)">Throughput vs. retrieval quality — Phase 5b '
        f'engines</text>',
        f'<text x="{L}" y="46" font-size="12" fill="var(--ink-2)">'
        f'queries/s across {workers} concurrent workers (log scale) vs '
        f'nDCG@10 — up and right is better</text>',
    ]

    def txt_w(s: str, px: float = 6.1) -> float:
        return px * len(s)

    lx = float(L)
    for fi in sorted({f for _, pts in panels for _, f, _, _ in pts}):
        e.append(f'<circle cx="{lx + 5:.0f}" cy="66" r="4.5" '
                 f'fill="var(--c{fi})" stroke="var(--surface)" '
                 f'stroke-width="2"/>')
        e.append(f'<text x="{lx + 16:.0f}" y="70" font-size="12" '
                 f'fill="var(--ink-2)">{_ENGINE_FAMILIES[fi]}</text>')
        lx += 16 + txt_w(_ENGINE_FAMILIES[fi]) + 22

    e.append(f'<text x="8" y="{p_top - 10}" font-size="11" '
             f'fill="var(--ink-3)">queries/s</text>')
    for v in yticks:
        e.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" font-size="11" '
                 f'text-anchor="end" fill="var(--ink-3)">{v:,.10g}</text>')

    for pi, (tag, pts) in enumerate(panels):
        px = L + pi * (PW + G)

        def X(v: float, px: float = px) -> float:
            return px + (v - x0) / (x1 - x0) * PW

        e.append(f'<text x="{px + PW / 2:.0f}" y="{p_top - 10}" '
                 f'font-size="12.5" font-weight="600" text-anchor="middle" '
                 f'fill="var(--ink)">{tag}</text>')
        for v in yticks:
            e.append(f'<line x1="{px}" y1="{Y(v):.1f}" x2="{px + PW}" '
                     f'y2="{Y(v):.1f}" stroke="var(--grid)" '
                     f'stroke-width="1"/>')
        for v in xticks:
            e.append(f'<line x1="{X(v):.1f}" y1="{p_top}" x2="{X(v):.1f}" '
                     f'y2="{p_bot}" stroke="var(--grid)" stroke-width="1"/>')
            e.append(f'<text x="{X(v):.1f}" y="{p_bot + 18}" font-size="11" '
                     f'text-anchor="middle" fill="var(--ink-3)">'
                     f'{v:.{dec}f}</text>')
        e.append(f'<line x1="{px}" y1="{p_bot}" x2="{px + PW}" y2="{p_bot}" '
                 f'stroke="var(--axis)" stroke-width="1"/>')
        e.append(f'<text x="{px + PW / 2:.0f}" y="{p_bot + 38}" '
                 f'font-size="11" text-anchor="middle" '
                 f'fill="var(--ink-3)">nDCG@10</text>')

        # points + direct labels (greedy vertical de-collision per panel)
        labs = []
        for label, fi, xv, yv in pts:
            cx, cy = X(xv), Y(yv)
            e.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" '
                     f'fill="var(--c{fi})" stroke="var(--surface)" '
                     f'stroke-width="2"/>')
            w = txt_w(label, 5.8)
            right = cx + 10 + w <= px + PW
            labs.append({"y": cy + 4, "label": label,
                         "x": cx + 10 if right else cx - 10,
                         "anchor": "start" if right else "end",
                         "span": (cx + 10, cx + 10 + w) if right
                                 else (cx - 10 - w, cx - 10)})
        labs.sort(key=lambda d: d["y"])
        for i, b in enumerate(labs):
            for a in labs[:i]:
                overlap = (b["span"][0] < a["span"][1]
                           and a["span"][0] < b["span"][1])
                if overlap and b["y"] - a["y"] < 13:
                    b["y"] = a["y"] + 13
        for d in labs:
            e.append(f'<text x="{d["x"]:.1f}" y="{d["y"]:.1f}" '
                     f'font-size="11" text-anchor="{d["anchor"]}" '
                     f'fill="var(--ink-2)">{d["label"]}</text>')

    e.append("</svg>")
    return "\n".join(e) + "\n"


def _fagin_sweep_chart_svg(head: list[dict]) -> str | None:
    """Epsilon sweep for the Fagin schedules on one chart: ms/query (log y) vs
    agree@K (linear x). One connected line per schedule (TA/TASD/GTA/GTASD)
    threading through its epsilon points in increasing-epsilon order, each point
    labeled with its epsilon; hue = schedule. One panel per source (dim/N).

    This is the speed/accuracy trade-off the epsilon knob buys: as epsilon
    grows a schedule moves down (faster) and usually left (lower agreement).
    Norm-aware schedules (GTA/GTASD) sit further down-right for the same
    epsilon. Returns None when no source has >= 2 Fagin points to connect.
    """
    # Gather per (source panel) -> per schedule -> list of (eps, agree, ms).
    panels = []
    for row in head:
        by_algo: dict[str, list] = {}
        for s in row["systems"]:
            parsed = _parse_fagin_name(s["name"])
            if parsed is None or s["ms"] <= 0:
                continue
            algo, eps = parsed
            by_algo.setdefault(algo, []).append((eps, s["agree"], s["ms"]))
        # keep schedules with at least one point; sort each by epsilon
        series = {a: sorted(pts) for a, pts in by_algo.items() if pts}
        if series:
            panels.append((f"{row['label']} (D={row['D']})", series))
    if not panels:
        return None
    n_pts = sum(len(pts) for _, series in panels for pts in series.values())
    if n_pts < 2:
        return None

    all_ms = [ms for _, series in panels for pts in series.values()
              for _, _, ms in pts]
    all_ag = [ag for _, series in panels for pts in series.values()
              for _, ag, _ in pts]
    ymin, ymax, yticks = _log_ticks(min(all_ms), max(all_ms))
    x_lo, x_hi = min(all_ag), max(all_ag)
    xpad = max((x_hi - x_lo) * 0.08, 0.005)
    x0, x1 = max(0.0, x_lo - xpad), min(1.0, x_hi + xpad)
    if x1 - x0 < 1e-9:
        x0, x1 = x0 - 0.02, x1 + 0.02

    algos_present = [a for a in ("TA", "TASD", "GTA", "GTASD")
                     if any(a in series for _, series in panels)]

    n = len(panels)
    L, PW, G, RP = 64, 400, 40, 48   # RP leaves room for right-edge eps labels
    W = L + n * PW + (n - 1) * G + RP
    p_top, p_bot = 118, 360
    H = 412
    ly0, ly1 = math.log10(ymin), math.log10(ymax)
    if ly1 - ly0 < 1e-9:
        ly0, ly1 = ly0 - 0.5, ly1 + 0.5

    def Y(v: float) -> float:
        return p_bot - (math.log10(v) - ly0) / (ly1 - ly0) * (p_bot - p_top)

    e: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" aria-label="Fagin epsilon sweep: '
        f'latency (ms/query, log scale) vs agreement with exact top-K, one '
        f'line per schedule">',
        _chart_css(len(_CHART_COLORS)),
        f'<rect width="{W}" height="{H}" rx="8" fill="var(--surface)"/>',
        f'<text x="{L}" y="26" font-size="15" font-weight="600" '
        f'fill="var(--ink)">Fagin epsilon sweep — latency vs. agreement</text>',
        f'<text x="{L}" y="46" font-size="12" fill="var(--ink-2)">'
        f'ms/query (log scale) vs agree@K with the exact top-K; each line is '
        f'one schedule swept over epsilon (labeled) — down and right is '
        f'better</text>',
    ]

    def txt_w(s: str, px: float = 6.1) -> float:
        return px * len(s)

    # legend: hue = schedule
    lx = float(L)
    for a in algos_present:
        ci = _FAGIN_ALGO_COLOR[a]
        e.append(f'<line x1="{lx:.0f}" y1="66" x2="{lx + 20:.0f}" y2="66" '
                 f'stroke="var(--c{ci})" stroke-width="2.5" '
                 f'stroke-linecap="round"/>')
        e.append(f'<circle cx="{lx + 10:.0f}" cy="66" r="3.5" '
                 f'fill="var(--c{ci})" stroke="var(--surface)" '
                 f'stroke-width="1.5"/>')
        e.append(f'<text x="{lx + 26:.0f}" y="70" font-size="12" '
                 f'fill="var(--ink-2)">{a}</text>')
        lx += 26 + txt_w(a) + 22

    # shared y label + ticks (drawn per panel)
    e.append(f'<text x="8" y="{p_top - 10}" font-size="11" '
             f'fill="var(--ink-3)">ms/query</text>')

    for pi, (title, series) in enumerate(panels):
        px = L + pi * (PW + G)

        def X(v: float, px: float = px) -> float:
            return px + (v - x0) / (x1 - x0) * PW

        e.append(f'<text x="{px + PW / 2:.0f}" y="{p_top - 10}" '
                 f'font-size="12.5" font-weight="600" text-anchor="middle" '
                 f'fill="var(--ink)">{title}</text>')
        for v in yticks:
            e.append(f'<line x1="{px}" y1="{Y(v):.1f}" x2="{px + PW}" '
                     f'y2="{Y(v):.1f}" stroke="var(--grid)" stroke-width="1"/>')
            if pi == 0:
                e.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" font-size="11" '
                         f'text-anchor="end" fill="var(--ink-3)">{v:g}</text>')
        # x ticks: 5 evenly across [x0, x1]
        for k in range(5):
            xv = x0 + (x1 - x0) * k / 4
            e.append(f'<line x1="{X(xv):.1f}" y1="{p_top}" x2="{X(xv):.1f}" '
                     f'y2="{p_bot}" stroke="var(--grid)" stroke-width="1"/>')
            e.append(f'<text x="{X(xv):.1f}" y="{p_bot + 18}" font-size="11" '
                     f'text-anchor="middle" fill="var(--ink-3)">'
                     f'{xv:.3f}</text>')
        e.append(f'<line x1="{px}" y1="{p_bot}" x2="{px + PW}" y2="{p_bot}" '
                 f'stroke="var(--axis)" stroke-width="1"/>')
        e.append(f'<text x="{px + PW / 2:.0f}" y="{p_bot + 38}" '
                 f'font-size="11" text-anchor="middle" '
                 f'fill="var(--ink-3)">agree@K vs exact</text>')

        # one polyline per schedule through its epsilon points, markers + eps
        # labels on top.
        marks: list[str] = []
        for a in algos_present:
            pts = series.get(a)
            if not pts:
                continue
            ci = _FAGIN_ALGO_COLOR[a]
            xy = [(X(ag), Y(ms)) for _, ag, ms in pts]
            if len(xy) >= 2:
                poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
                e.append(f'<polyline points="{poly}" fill="none" '
                         f'stroke="var(--c{ci})" stroke-width="2" '
                         f'stroke-linejoin="round" stroke-linecap="round"/>')
            for (eps, ag, ms), (cx, cy) in zip(pts, xy):
                marks.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" '
                             f'fill="var(--c{ci})" stroke="var(--surface)" '
                             f'stroke-width="1.5"/>')
                marks.append(f'<text x="{cx + 6:.1f}" y="{cy - 6:.1f}" '
                             f'font-size="9.5" fill="var(--ink-3)">'
                             f'ε={eps:g}</text>')
        e += marks

    e.append("</svg>")
    return "\n".join(e) + "\n"


def _engine_scaling_lines(head: list[dict]) -> list[str]:
    """Analysis section: how each engine's latency scales from the smallest to
    the largest dimensionality in the run, and why graph ANN (HNSW) dominates
    the coordinate-wise Threshold Algorithm (Fagin) on dense embeddings.

    Data-driven: the speedups, agreement, and D-scaling factors are pulled from
    the Phase 5b `head` rows; only the mechanistic rationale is fixed prose.
    Returns [] when there aren't at least two dimensionalities to compare.
    """
    # one representative row per distinct D (smallest and largest)
    by_d: dict[int, dict] = {}
    for row in head:
        by_d.setdefault(int(row["D"]), row)
    if len(by_d) < 2:
        return []
    d_lo, d_hi = min(by_d), max(by_d)
    lo, hi = by_d[d_lo], by_d[d_hi]

    def find(row, *prefixes):
        for s in row["systems"]:
            if s["name"].startswith(prefixes):
                return s
        return None

    def scaling(*prefixes):
        a, b = find(lo, *prefixes), find(hi, *prefixes)
        if not (a and b and a["ms"] > 0):
            return None
        return a, b, b["ms"] / a["ms"]

    flat = scaling("FLAT", "FAISS FlatIP", "Milvus FLAT")
    hnsw = scaling("FAISS HNSW", "Milvus HNSW")
    # Fagin: compare the best-agreement (slowest) row at each D
    def best_fagin(row):
        cands = [s for s in row["systems"] if s["name"].startswith("Fagin")]
        return max(cands, key=lambda s: s["agree"]) if cands else None
    fg_lo, fg_hi = best_fagin(lo), best_fagin(hi)
    fagin = (fg_lo, fg_hi, fg_hi["ms"] / fg_lo["ms"]) if (
        fg_lo and fg_hi and fg_lo["ms"] > 0) else None

    L: list[str] = []
    A = L.append
    A("## Why graph ANN beats the Threshold Algorithm on dense embeddings")
    A("")
    A(f"On these dense granite embeddings the ranking is unambiguous: **HNSW "
      f"≫ exact brute force ≫ Fagin TA/TASD**. The decisive evidence is how "
      f"each engine's per-query latency scales as the vector dimensionality "
      f"grows from {d_lo} → {d_hi} (same corpus, N={lo['N']:,}):")
    A("")
    A("| engine | " + f"{d_lo}d ms/q | {d_hi}d ms/q | ×(→{d_hi}d) | "
      f"agree@K ({d_hi}d) |")
    A("|---|---|---|---|---|")
    if hnsw:
        a, b, r = hnsw
        A(f"| HNSW (graph ANN) | {a['ms']:.2f} | {b['ms']:.2f} | "
          f"**{r:.2f}×** | {b['agree']:.4f} |")
    if flat:
        a, b, r = flat
        A(f"| exact brute force (FLAT) | {a['ms']:.2f} | {b['ms']:.2f} | "
          f"{r:.2f}× | {b['agree']:.4f} |")
    if fagin:
        a, b, r = fagin
        A(f"| Fagin TA/TASD (best agree) | {a['ms']:.2f} | {b['ms']:.2f} | "
          f"{r:.2f}× | {b['agree']:.4f} |")
    A("")
    if hnsw and flat:
        a_h, _, _ = hnsw
        a_f, _, _ = flat
        sp_lo = a_f["ms"] / a_h["ms"] if a_h["ms"] > 0 else float("nan")
        b_h, b_f = hnsw[1], flat[1]
        sp_hi = b_f["ms"] / b_h["ms"] if b_h["ms"] > 0 else float("nan")
        A(f"**HNSW is the whole frontier.** It runs ~{sp_lo:.0f}× faster than "
          f"exact at {d_lo}d and ~{sp_hi:.0f}× faster at {d_hi}d for a fraction "
          f"of a percent of nDCG loss (agreement {hnsw[1]['agree']:.3f}). At "
          f"this N and D it is a near-free lunch, not a trade-off.")
        A("")
    A("**Why the scaling differs so sharply.** HNSW does a near-constant "
      "number of distance evaluations per query — an `ef`-bounded graph "
      "traversal of a few hundred to low-thousands of dot products, roughly "
      "independent of N. Doubling D only widens each of those few dot "
      "products, so its latency barely moves. It is the only engine whose "
      "cost decouples from the corpus size.")
    A("")
    A("Fagin's Threshold Algorithm is the opposite. TA halts when the running "
      "threshold `T = Σ_d frontier_d` (the best still-unseen score, summed "
      "over dimensions) falls below the k-th best confirmed score. That "
      "collapses quickly only when the score is a sum over **few or sparse** "
      "attributes. Dense embeddings are the pathological case: every one of "
      "the 384/768 dimensions contributes a little, so `T` decays extremely "
      "slowly and a large fraction of the corpus survives sorted access to be "
      "fully scored by random access — each such scoring a scattered, "
      "cache-hostile gather. Both effects grow with D at once (slower "
      "threshold decay **and** wider gathers), so Fagin scales *worse than "
      "linearly* in D" + (f" ({fagin[2]:.1f}× from {d_lo}→{d_hi}d, vs "
      f"{flat[2]:.1f}× for the linear-in-D exact scan)" if (fagin and flat)
      else "") + ".")
    A("")
    A("The `epsilon` slack knob only attacks the random-access count, and "
      "slowly: on this data the sorted-scan depth floors at a few percent "
      "almost immediately, but even aggressive slack still fully scores a "
      "majority of the corpus. To reach HNSW's agreement Fagin must run "
      "*slower than exact brute force*; to reach HNSW's latency it must set "
      "epsilon so high that recall collapses. It is never on the Pareto "
      "frontier here.")
    A("")
    A("**Takeaway.** For dense retrieval embeddings at this scale, use graph "
      "ANN (HNSW) or exact SIMD brute force. Coordinate-wise threshold "
      "pruning (Fagin TA) is a structural mismatch for dense vectors — it is "
      "built for sparse/few-attribute scoring — and these numbers are best "
      "read as a clean demonstration of *why* it loses, not as a competitive "
      "baseline.")
    A("")
    return L


def _fmt_epsilon(eps) -> str:
    """Render fagin_epsilon (scalar or swept list) for the report prose."""
    if isinstance(eps, (list, tuple)):
        return "{" + ", ".join(f"{float(e):g}" for e in eps) + "}"
    return f"{float(eps):g}"


def _fmt_schedule(sched) -> str:
    """Render fagin_schedule (scalar or swept list) for the report prose."""
    if isinstance(sched, (list, tuple)):
        return "{" + ", ".join(str(s) for s in sched) + "}"
    return str(sched)


def _parse_fagin_name(name: str):
    """('Fagin GTASD (eps=0.01, depth=0)') -> ('GTASD', 0.01); the exact rows
    ('Fagin TA (exact)') map to epsilon 0.0. Returns None for non-Fagin rows."""
    if not name.startswith("Fagin "):
        return None
    algo = name.split(" ", 2)[1]              # TA / TASD / GTA / GTASD
    if "(exact)" in name:
        return algo, 0.0
    m = re.search(r"eps=([0-9.eE+-]+)", name)
    return (algo, float(m.group(1))) if m else (algo, 0.0)


def _method_family(name: str) -> str:
    """Bucket a method/variant name into an engine family for the DB row.

    Handles both Phase 5b system names ('simdq asym b=2 (+rescore)',
    'Milvus FLAT (exact)') and quality variant labels ('asym b=2, +rescore',
    '1-bit ham, no rescore', 'FLAT (fp32 IP, exact)').
    """
    if name.startswith("Milvus"):
        return "Milvus"
    if name.startswith("FAISS"):
        return "FAISS"
    if name.startswith("Fagin"):
        return "Fagin"
    if name.startswith("FLAT"):
        return "FLAT"
    # simdq quality variants ('asym b=2, ...', '1-bit ham, ...') and Phase 5b
    # simdq system names ('simdq asym ...', 'simdq 1-bit ...')
    return "simdq"


def _simdq_name_to_variant(name: str):
    """'simdq asym b=2 (+rescore)' -> 'asym b=2, +rescore' (and the 1-bit
    variants). Returns None for non-simdq names. Single source of truth for the
    Phase 5b simdq name -> quality variant mapping, shared by the frontier chart
    and the results-DB join."""
    if not name.startswith("simdq "):
        return None
    mode = "+rescore" if "+rescore" in name else "no rescore"
    var = "asym b=2" if name.startswith("simdq asym") else "1-bit ham"
    return f"{var}, {mode}"


def _speed_name_to_variant(name: str) -> str:
    """Map a Phase 5b system name to the quality `variant` key it shares with
    metric_rows/agree_rows. simdq names fold onto the quality variants; every
    other engine uses its name verbatim."""
    return _simdq_name_to_variant(name) or name


def _join_method_rows(models, metric_rows, agree_rows, head):
    """Join quality (metric_rows), agreement (agree_rows), and Phase 5b speed
    (head) into one flat dict per (encoder, method), keyed by the quality
    `variant` label. Missing phases leave their columns as None.

    Returns a list of dicts with keys matching the `runs` table columns
    (minus run_id/run_ts/hostname/dataset/top_k/workers, which the writer adds).
    """
    metric_by = {(r["encoder"], r["variant"]): r for r in metric_rows}
    agree_by = {(r["encoder"], r["variant"]): r for r in agree_rows}

    # speed: (encoder, variant) -> system dict, using the model tag as encoder.
    speed_by = {}
    for m, hrow in zip(models, head or []):
        tag = m["tag"]
        for s in hrow.get("systems", []):
            speed_by[(tag, _speed_name_to_variant(s["name"]))] = s

    # union of every (encoder, method) key seen in any phase
    keys = set(metric_by) | set(agree_by) | set(speed_by)
    rows = []
    for enc, method in sorted(keys):
        mm = metric_by.get((enc, method))
        ag = agree_by.get((enc, method))
        sp = speed_by.get((enc, method))
        parsed = _parse_fagin_name(method)
        epsilon = parsed[1] if parsed is not None else None
        ms = sp["ms"] if sp else None
        rows.append({
            "encoder": enc,
            "method": method,
            "method_family": _method_family(method),
            "epsilon": epsilon,
            "ndcg_at_10": mm["nDCG@10"] if mm else None,
            "recall_at_10": mm["recall@10"] if mm else None,
            "recall_at_k": mm["recall@K"] if mm else None,
            "mrr_at_10": mm["MRR@10"] if mm else None,
            "agree_at_10": ag["agree@10"] if ag else None,
            "agree_at_k": ag["agree@K"] if ag else None,
            "n_queries": mm["n_queries"] if mm else None,
            "runtime_ms_per_q": ms,
            "queries_per_sec": (1000.0 / ms) if ms and ms > 0 else None,
        })
    return rows


_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    run_ts            TEXT NOT NULL,
    hostname          TEXT,
    dataset           TEXT,
    top_k             INTEGER,
    workers           INTEGER,
    encoder           TEXT,
    method            TEXT NOT NULL,
    method_family     TEXT,
    epsilon           REAL,
    ndcg_at_10        REAL,
    recall_at_10      REAL,
    recall_at_k       REAL,
    mrr_at_10         REAL,
    agree_at_10       REAL,
    agree_at_k        REAL,
    n_queries         INTEGER,
    runtime_ms_per_q  REAL,
    queries_per_sec   REAL
)
"""

_RUNS_COLUMNS = [
    "run_id", "run_ts", "hostname", "dataset", "top_k", "workers",
    "encoder", "method", "method_family", "epsilon",
    "ndcg_at_10", "recall_at_10", "recall_at_k", "mrr_at_10",
    "agree_at_10", "agree_at_k", "n_queries",
    "runtime_ms_per_q", "queries_per_sec",
]


def write_results_db(db_path, cfg, isa, run_id, run_ts,
                     metric_rows, agree_rows, head):
    """Append one row per (model x method) to the persistent SQLite `runs`
    table at db_path. Self-initializing (CREATE TABLE IF NOT EXISTS),
    append-only. Returns the number of rows written.
    """
    st = cfg["settings"]
    ds = cfg["dataset"]
    base = {
        "run_id": run_id,
        "run_ts": run_ts,
        "hostname": isa.get("hostname"),
        "dataset": ds.get("name"),
        "top_k": int(st["top_k"]),
        "workers": int(st.get("workers", 0)) or None,
    }
    joined = _join_method_rows(cfg["models"], metric_rows, agree_rows, head)
    rows = [tuple((base | r).get(c) for c in _RUNS_COLUMNS) for r in joined]

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.execute(_RUNS_SCHEMA)
        placeholders = ", ".join("?" for _ in _RUNS_COLUMNS)
        con.executemany(
            f"INSERT INTO runs ({', '.join(_RUNS_COLUMNS)}) "
            f"VALUES ({placeholders})", rows)
        con.commit()
    finally:
        con.close()
    return len(rows)


def _method_type(method: str, family: str) -> str:
    """Bucket a stored method into its display 'method type'. Fagin rows keep
    their schedule (Fagin TA / TASD / GTA / GTASD) so epsilon-swept rows of the
    same schedule group together; every other engine drops its parenthesized
    parameter suffix ('FAISS HNSW (M=16, ef=128)' -> 'FAISS HNSW'). simdq
    labels have no ' (' suffix and pass through unchanged.
    """
    if family == "Fagin":
        parsed = _parse_fagin_name(method)
        if parsed is not None:
            return f"Fagin {parsed[0]}"
    return method.split(" (")[0]


def _select_representative_epsilons(values) -> list:
    """From the distinct epsilon values present for one Fagin type, pick a few
    representative ones spanning the trade-off: exact (0.0) if present, plus the
    smallest, median, and largest nonzero values. Deduplicated, sorted
    ascending, at most 4. If <=4 distinct values exist, all are returned.
    """
    uniq = sorted(set(float(v) for v in values))
    if len(uniq) <= 4:
        return uniq
    picked = []
    if uniq[0] == 0.0:
        picked.append(0.0)
    nonzero = [v for v in uniq if v != 0.0]
    if nonzero:
        low = nonzero[0]
        high = nonzero[-1]
        # lower-median so a 4-nonzero list picks index 1 (e.g. 0.005 from
        # [0.001, 0.005, 0.01, 0.05])
        mid = nonzero[(len(nonzero) - 1) // 2]
        for v in (low, mid, high):
            if v not in picked:
                picked.append(v)
    return sorted(picked)


def _row_is_newer(a: dict, b: dict) -> bool:
    """True if row a is newer than row b (by run_ts, tiebreak on id)."""
    ka = (a.get("run_ts") or "", a.get("id") or 0)
    kb = (b.get("run_ts") or "", b.get("id") or 0)
    return ka > kb


def build_latest_summary(rows: list[dict]) -> list[dict]:
    """Reduce raw `runs` rows to the sorted display rows for the --latest
    table: the newest row per (encoder, method string), bucketed into
    (encoder, method type), with only representative epsilons kept for Fagin
    types. Keying on encoder keeps every model in the output — models sharing a
    method name (e.g. "FAISS HNSW") no longer collapse to whichever ran last.
    Each display row carries method_family + method_type alongside the metrics.
    """
    # 1) newest row per (encoder, exact method string)
    latest_by_method: dict = {}
    for r in rows:
        key = (r["encoder"], r["method"])
        cur = latest_by_method.get(key)
        if cur is None or _row_is_newer(r, cur):
            latest_by_method[key] = r

    # 2) group those into (encoder, method type) buckets
    by_type: dict = {}
    for r in latest_by_method.values():
        mt = _method_type(r["method"], r["method_family"])
        by_type.setdefault((r["encoder"], mt), []).append(r)

    # 3) per (encoder, type): non-epsilon -> latest single row;
    #    epsilon -> representatives
    display: list = []
    for (_enc, mt), group in by_type.items():
        has_eps = any(g.get("epsilon") is not None for g in group)
        if not has_eps:
            best = group[0]
            for g in group[1:]:
                if _row_is_newer(g, best):
                    best = g
            display.append(_display_row(best, mt))
            continue
        eps_values = [float(g["epsilon"]) for g in group
                      if g.get("epsilon") is not None]
        keep = set(_select_representative_epsilons(eps_values))
        # newest row per (type, epsilon), only for kept epsilons
        best_by_eps: dict = {}
        for g in group:
            e = g.get("epsilon")
            if e is None or float(e) not in keep:
                continue
            e = float(e)
            cur = best_by_eps.get(e)
            if cur is None or _row_is_newer(g, cur):
                best_by_eps[e] = g
        for e in sorted(best_by_eps):
            display.append(_display_row(best_by_eps[e], mt))

    # 4) sort by encoder, then family, then type, then epsilon (None first)
    display.sort(key=lambda d: (d["encoder"] or "", d["method_family"],
                                d["method_type"],
                                -1.0 if d["epsilon"] is None
                                else float(d["epsilon"])))
    return display


def _display_row(r: dict, method_type: str) -> dict:
    """Project a raw row into a display row for the --latest table."""
    return {
        "dataset": r.get("dataset"),
        "encoder": r.get("encoder"),
        "method_family": r.get("method_family"),
        "method_type": method_type,
        "epsilon": (None if r.get("epsilon") is None
                    else float(r["epsilon"])),
        "agree_at_10": r.get("agree_at_10"),
        "agree_at_k": r.get("agree_at_k"),
        "ndcg_at_10": r.get("ndcg_at_10"),
        "runtime_ms_per_q": r.get("runtime_ms_per_q"),
        "queries_per_sec": r.get("queries_per_sec"),
        "run_ts": r.get("run_ts"),
    }


# (column key, header) — single source of column order for table + CSV
_LATEST_COLUMNS = [
    ("dataset", "dataset"), ("encoder", "encoder"),
    ("method_type", "method_type"), ("epsilon", "epsilon"),
    ("agree_at_10", "agree@10"), ("agree_at_k", "agree@K"),
    ("ndcg_at_10", "nDCG@10"), ("runtime_ms_per_q", "ms/q"),
    ("queries_per_sec", "q/s"), ("run_ts", "run_ts"),
]


def _fmt_cell(key: str, value) -> str:
    """Format one display value for the table/CSV. None -> ''. Floats use
    compact fixed precision per column; epsilon uses :g."""
    if value is None:
        return ""
    if key == "epsilon":
        return f"{float(value):g}"
    if key in ("agree_at_10", "agree_at_k", "ndcg_at_10"):
        return f"{float(value):.4f}"
    if key in ("runtime_ms_per_q", "queries_per_sec"):
        return f"{float(value):.2f}"
    return str(value)


def render_latest_table(display_rows) -> str:
    """Aligned monospace table of the --latest display rows."""
    if not display_rows:
        return "no rows to show"
    headers = [h for _, h in _LATEST_COLUMNS]
    cells = [[_fmt_cell(k, r.get(k)) for k, _ in _LATEST_COLUMNS]
             for r in display_rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    def fmt_line(vals):
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(vals))
    lines = [fmt_line(headers), fmt_line(["-" * w for w in widths])]
    lines += [fmt_line(row) for row in cells]
    return "\n".join(lines)


def write_latest_csv(display_rows, path) -> None:
    """Write the --latest display rows as CSV to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([h for _, h in _LATEST_COLUMNS])
        for r in display_rows:
            w.writerow([_fmt_cell(k, r.get(k)) for k, _ in _LATEST_COLUMNS])


def query_latest_rows(db_path) -> list:
    """Read every row from the `runs` table as a list of dicts. Returns [] if
    the DB file or the table is absent."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    try:
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute("SELECT * FROM runs")
        except sqlite3.OperationalError:
            return []   # no `runs` table
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def run_latest_report(db_path, csv_path) -> None:
    """Print the latest-per-method-type summary to stdout; if csv_path is set,
    also write it as CSV. Prints a friendly message when there is nothing to
    show."""
    db_path = Path(db_path)
    rows = query_latest_rows(db_path)
    if not db_path.exists():
        print(f"no results DB at {db_path}")
        return
    if not rows:
        print(f"results DB has no rows yet ({db_path})")
        return
    display = build_latest_summary(rows)
    print(render_latest_table(display))
    if csv_path is not None:
        write_latest_csv(display, csv_path)
        print(f"# latest-csv -> {Path(csv_path)}")


# Fagin schedule -> hue slot in _CHART_COLORS (stable across the report so the
# epsilon-sweep chart and any future Fagin chart agree on colour per schedule).
_FAGIN_ALGO_COLOR = {"TA": 0, "TASD": 1, "GTA": 2, "GTASD": 3}


def write_report(out_path: Path, cfg: dict, isa: dict, corpus_sizes: list[int],
                 n_queries: int, query_npys: list[Path],
                 metric_rows, agree_rows, p1_default, p1_blas1, p68, head,
                 cmdline: str):
    ds, st = cfg["dataset"], cfg["settings"]
    models = cfg["models"]
    K = int(st["top_k"])
    K_re = max(K, min(K * int(st["alpha"]), 256))
    threads = st["threads"]
    lines: list[str] = []
    A = lines.append

    A(f"# retrieval experiment — {isa['hostname']} — {ds['name']}")
    A("")
    A(f"_Auto-generated by `{SCRIPT_REL}` on "
      f"{date.today().isoformat()}. dataset={ds['name']}, K={K}, K'={K_re}, "
      f"queries={st['speed_queries']} for speed"
      + (f" / {n_queries} for quality" if metric_rows else "")
      + f", workers={st['workers']}._")
    A("")
    A("**Dataset provenance.** "
      + " / ".join(f"{m['tag']}: {n:,} chunks from "
                   f"`{Path(m['asym_index']).name}`"
                   for m, n in zip(models, corpus_sizes))
      + (f" (corpus source: `{ds['corpus_file']}`)"
         if ds.get("corpus_file") else "")
      + f". Queries: **{n_queries} queries** from "
      + (f"`{ds['queries_jsonl']}`" if ds.get("queries_jsonl") else "<unset>")
      + ", encoded with "
      + ", ".join(f"`{m.get('encoder', '<precomputed>')}`" for m in models)
      + " (L2-normalized). Query vectors: "
      + ", ".join(f"`{p}`" for p in query_npys) + ".")
    A("")

    A("## Environment")
    A("")
    A(f"- CPU: **{isa['model']}**")
    for k in ("Socket(s)", "Core(s) per socket", "Thread(s) per core",
              "L1d cache", "L2 cache", "L3 cache"):
        if isa["cache"].get(k):
            A(f"- {k}: {isa['cache'][k]}")
    flags = [f"{label}: {'yes' if isa.get(k) else 'NO'}"
             for k, label in [("has_avx2", "AVX2"), ("has_avx512f", "AVX-512F"),
                              ("has_avx512vbmi", "AVX-512 VBMI"),
                              ("has_gfni", "GFNI"),
                              ("has_vpopcntdq", "VPOPCNTDQ"), ("fma", "FMA")]]
    A(f"- SIMD: {', '.join(flags)}")
    A(f"- Compiled `.so`: `{isa['compiled_so']}`")
    A(f"- Kernel path (from disassembly): **{isa['kernel_path']}** "
      f"(ymm={isa['so_counts'].get('ymm', 0)}, "
      f"zmm={isa['so_counts'].get('zmm', 0)}, "
      f"vfmadd={isa['so_counts'].get('vfmadd', 0)}, "
      f"vpmultishift={isa['so_counts'].get('vpmultishift', 0)})")
    A(f"- Python: {platform.python_version()}, numpy: {np.__version__}")
    A("")

    if metric_rows:
        A(f"## Retrieval quality on {ds['name']} queries")
        A("")
        A(f"Ground-truth: `{ds['relevant_field']}` doc IDs in "
          f"`{ds['queries_jsonl']}` ({n_queries} queries). "
          f"`FLAT (fp32 IP, exact)` is the exact-search ceiling — direct dense "
          f"scan of the same `floats_mmap` simdq was built from. Simdq "
          f"`+rescore` variants use `K'={K_re}` and re-rank against fp32 "
          f"`cand_floats @ q_proj`; `no rescore` keeps `K'=K={K}` and ranks "
          f"codes-only. Milvus / FAISS / Fagin TA rows (when those baselines "
          f"are enabled) search the same fp32 vectors with the same queries; "
          f"HNSW uses M={st['hnsw_m']}, "
          f"efConstruction={st['hnsw_ef_construction']}, "
          f"ef={max(int(st['hnsw_ef']), K)}.")
        A("")
        A(f"| encoder | variant | Recall@10 | Recall@{K} | MRR@10 | nDCG@10 |")
        A("|---|---|---|---|---|---|")
        for r in metric_rows:
            bold = r["variant"].startswith("FLAT")
            fmt = (lambda x: f"**{x:.4f}**") if bold else (lambda x: f"{x:.4f}")
            var = f"**{r['variant']}**" if bold else r["variant"]
            A(f"| {r['encoder']} | {var} | {fmt(r['recall@10'])} | "
              f"{fmt(r['recall@K'])} | {fmt(r['MRR@10'])} | "
              f"{fmt(r['nDCG@10'])} |")
        A("")
        A("### Overlap with FLAT top-K (agreement with the exact scan)")
        A("")
        A("How many of FLAT's top-K corpus *positions* each system recovers "
          "— a scan-quality metric that isolates code-lossiness (or HNSW "
          "graph recall) from encoder quality (this number ignores whether "
          "either is answering the query correctly).")
        A("")
        A(f"| encoder | variant | agree@10 | agree@{K} |")
        A("|---|---|---|---|")
        for r in agree_rows:
            A(f"| {r['encoder']} | {r['variant']} | "
              f"{r['agree@10']:.4f} | {r['agree@K']:.4f} |")
        A("")

    chart_svg = svg_name = None
    if p1_default is not None:
        sweeps = [(lbl, rows) for lbl, rows in [
            ("asym b=2 · default BLAS", p1_default),
            ("asym b=2 · BLAS=1", p1_blas1),
            ("Hamming SoA", p68)] if rows]
        svg_name = out_path.stem + ".sweeps.svg"
        chart_svg = _sweep_chart_svg(sweeps, threads,
                                     [m["tag"] for m in models], K, K_re)
        A("## Thread scaling — Phase 1 (asym b=2) & Phase 6/8 (Hamming SoA)")
        A("")
        A(f"Median ms/query for `SimdqIndex.search` as `num_threads` grows, "
          f"all sweeps in one chart. Color picks the sweep config: asym b=2 "
          f"with default BLAS, asym b=2 under `OPENBLAS_NUM_THREADS=1` (via "
          f"`threadpool_limits`), and the Hamming SoA kernel. Marker shape "
          f"picks the model/index. Solid lines include the fp32 rescore "
          f"stage (`K'={K_re}`: gather from `floats_mmap`, then "
          f"`cand_floats @ q_proj`, then argsort); dashed lines are "
          f"codes-only (`K'=K={K}`, short-circuiting the rescore block — "
          f"see `_search_asym` in `simdq_index.py`). Identity projection, "
          f"so the Phase-3 fix means there's no `W@q` to oversubscribe "
          f"BLAS.")
        A("")
        A(f"![Median ms/query (log scale) vs num_threads for every sweep "
          f"config, model, and rescore mode]({svg_name})")
        A("")
        A("> On hardware where the Phase-3 identity-skip is active, the "
          "BLAS=1 lines should sit on top of the default-BLAS lines; a "
          "visible gap means an unintended GEMV is still firing per query. "
          "The solid vs dashed gap isolates the cost of the fp32 rerank "
          "stage on top of the native scan.")
        A("")
        A("<details>")
        A("<summary>Raw sweep tables (median ms/query)</summary>")
        A("")
        A("### Phase 1 — asym b=2, default BLAS")
        A("")
        A(_sweep_md_pair(p1_default, threads, K, K_re))
        A("")
        A("### Phase 1 — asym b=2, `OPENBLAS_NUM_THREADS=1`")
        A("")
        A(_sweep_md_pair(p1_blas1, threads, K, K_re))
        A("")
        A("### Phase 6/8 — Hamming SoA")
        A("")
        A(_sweep_md_pair(p68, threads, K, K_re))
        A("")
        A("</details>")
        A("")

    if head:
        A(f"## Phase 5b — head-to-head: Milvus / FAISS / Fagin / simdq "
          f"({st['workers']}-way concurrent)")
        A("")
        A(f"ms/query with {st['workers']} concurrent query workers, each "
          f"engine call single-threaded (matches "
          f"`scripts/bench_simdq_vs_milvus.py`). `agree@{K}` is the overlap "
          f"with the exact fp32 top-{K} on the timed queries — 1.0 for exact "
          f"engines by construction; for HNSW it is the recall its "
          f"`ef={max(int(st['hnsw_ef']), K)}` buys, and for simdq "
          f"`no rescore` it isolates code-lossiness. HNSW built with "
          f"M={st['hnsw_m']}, efConstruction={st['hnsw_ef_construction']}. "
          f"Milvus engines pay the client RPC; FAISS, Fagin, and simdq are "
          f"in-process. Fagin TA runs Fagin's Threshold Algorithm over "
          f"per-dimension sorted lists (batch={int(st['fagin_batch_rows'])}, "
          f"epsilon={_fmt_epsilon(st['fagin_epsilon'])}, "
          f"max_depth={int(st['fagin_max_depth'])}, "
          f"schedule={_fmt_schedule(st.get('fagin_schedule', 'lockstep'))}; "
          f"epsilon=0 with unlimited depth is exact; TA/TASD use the classic "
          f"box threshold, GTA/GTASD the norm-aware water-filling bound). "
          f"Simdq `+rescore` uses K'={K_re}, "
          f"`no rescore` "
          f"K'=K={K}.")
        A("")
        for row in head:
            A(f"**{row['label']}** (dim={row['D']}, N={row['N']:,}, "
              f"queries={row.get('queries', 'synth')}):")
            A("")
            A(f"| system | ms/query | agree@{K} vs exact |")
            A("|---|---|---|")
            for s in row["systems"]:
                A(f"| {s['name']} | {s['ms']:.2f} | {s['agree']:.4f} |")
            A("")

        # Data-driven analysis of why HNSW dominates the Threshold Algorithm
        # on dense embeddings — only meaningful when Fagin actually ran and
        # there are >=2 dimensionalities to compare the D-scaling across.
        if st.get("run_fagin", True) and any(
                s["name"].startswith("Fagin")
                for row in head for s in row["systems"]):
            for line in _engine_scaling_lines(head):
                A(line)

    # Fagin epsilon-sweep chart: latency vs agreement, one line per schedule
    # through its epsilon points. Only meaningful when the sweep produced
    # multiple Fagin points (>=2 epsilons, or >=2 schedules).
    fagin_svg = fagin_name = None
    if head:
        fagin_svg = _fagin_sweep_chart_svg(head)
        if fagin_svg:
            fagin_name = out_path.stem + ".fagin_eps.svg"
    if fagin_svg:
        A("## Fagin epsilon sweep — latency vs. agreement")
        A("")
        A(f"Every Fagin schedule in the run, swept over "
          f"`epsilon={_fmt_epsilon(st['fagin_epsilon'])}`: concurrent "
          f"ms/query (log scale) against agree@{K} with the exact fp32 "
          f"top-{K}, one connected line per schedule "
          f"(schedule={_fmt_schedule(st.get('fagin_schedule', 'lockstep'))}). "
          f"Each point is one epsilon (labeled); epsilon=0 is the exact end of "
          f"each line. Down and right is better — the norm-aware GTA/GTASD "
          f"schedules use the tighter water-filling halting bound, so they "
          f"reach the same agreement at lower latency (and land further down "
          f"the line for the same epsilon).")
        A("")
        A(f"![Fagin epsilon sweep: ms/query (log scale) vs agree@{K}, one line "
          f"per schedule]({fagin_name})")
        A("")

    frontier_svg = frontier_name = None
    if metric_rows and head:
        frontier_name = out_path.stem + ".frontier.svg"
        frontier_svg = _frontier_chart_svg(models, metric_rows, head,
                                           int(st["workers"]))
    if frontier_svg:
        A("## Speed × quality frontier")
        A("")
        A(f"Every Phase 5b engine that also has a quality row, one panel "
          f"per model: aggregate throughput (queries/s = 1000 / concurrent "
          f"ms/query, {st['workers']} workers) against nDCG@10 on the gold "
          f"labels. Up and right is better. Exact engines sit on the FLAT "
          f"nDCG ceiling by construction; the vertical spread shows what "
          f"each approximate engine's speed buys, the horizontal spread "
          f"what it costs.")
        A("")
        A(f"![Throughput (queries/s, log scale) vs nDCG@10 per engine, "
          f"one panel per model]({frontier_name})")
        A("")

    if metric_rows and head:
        A("## Combined quality × speed takeaway")
        A("")
        A(f"At {ds['name']} retrieval quality (simdq +rescore variants, "
          f"compared against each Milvus / FAISS engine timed in Phase 5b):")
        A("")
        A("| model | simdq variant | nDCG@10 (vs FLAT) | "
          "ms/query @ concurrent | vs baselines |")
        A("|---|---|---|---|---|")
        by_ev = {(r["encoder"], r["variant"]): r for r in metric_rows}
        for m, hrow in zip(models, head):
            tag = m["tag"]
            flat = by_ev.get((tag, "FLAT (fp32 IP, exact)"))
            sys_by_name = {s["name"]: s for s in hrow["systems"]}
            baselines = [s for s in hrow["systems"]
                         if s["name"].startswith(("Milvus", "FAISS"))]
            for variant, sys_prefix in [
                    ("asym b=2, +rescore", "simdq asym"),
                    ("1-bit ham, +rescore", "simdq 1-bit")]:
                q = by_ev.get((tag, variant))
                srow = next((s for s in hrow["systems"]
                             if s["name"].startswith(sys_prefix)
                             and "+rescore" in s["name"]), None)
                if not (q and flat and srow):
                    continue
                delta = (q["nDCG@10"] - flat["nDCG@10"]) * 100
                dtxt = ("matches FLAT" if abs(delta) < 0.005
                        else f"{delta:+.1f} pts")
                vs = "; ".join(
                    f"{b['ms']/srow['ms']:.1f}× vs "
                    f"{b['name'].split(' (')[0]} ({b['ms']:.2f} ms)"
                    for b in baselines)
                A(f"| {tag} | {variant} | {q['nDCG@10']:.4f} / "
                  f"{flat['nDCG@10']:.4f} ({dtxt}) | **{srow['ms']:.2f} ms** "
                  f"| {vs or '—'} |")
        A("")

    A("## Reproduce")
    A("")
    A("```bash")
    A(cmdline)
    A("```")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    if chart_svg:
        (out_path.parent / svg_name).write_text(chart_svg)
    if frontier_svg:
        (out_path.parent / frontier_name).write_text(frontier_svg)
    if fagin_svg:
        (out_path.parent / fagin_name).write_text(fagin_svg)


# ---------- main ------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="YAML experiment config (see experiments/simdq/nq_real.yaml)")
    ap.add_argument("--model", action="append", default=None,
                    help="model spec as 'k=v;k=v' with keys tag, encoder, "
                         "asym, hamming, queries_npy, max_seq_length. "
                         "Repeatable; appended to YAML models.")
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--queries-jsonl", default=None,
                    help="jsonl with query text + gold relevant doc ids")
    ap.add_argument("--corpus-file", default=None,
                    help="corpus jsonl(.bz2)/tsv to encode + quantize when "
                         "index dirs need to be built")
    ap.add_argument("--index-root", dest="index_root", default=None,
                    help="directory for auto-built index dirs "
                         "(default: experiments/simdq/indexes)")
    ap.add_argument("--results-db", dest="results_db", default=None,
                    help="persistent SQLite file to append per-method results "
                         "to (default: experiments/simdq/results.db). Set to "
                         "an empty string to disable. With --latest, this "
                         "selects the DB to read (empty string falls back to "
                         "the default).")
    ap.add_argument("--latest", action="store_true",
                    help="print a summary table of the latest result per "
                         "method type from the results DB and exit (no "
                         "experiment is run); representative epsilons are "
                         "shown for Fagin schedules")
    ap.add_argument("--latest-csv", dest="latest_csv", default=None,
                    help="with --latest, also write the summary table as CSV "
                         "to this path")
    ap.add_argument("--force-build", action="store_true",
                    help="rebuild index dirs and corpus vectors even if they "
                         "already exist")
    ap.add_argument("--top-k", dest="top_k", type=int, default=None)
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--speed-queries", dest="speed_queries", type=int,
                    default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", default=None,
                    help="comma-separated num_threads for sweeps (0 = all)")
    ap.add_argument("--milvus-uri", dest="milvus_uri", default=None)
    ap.add_argument("--no-milvus", action="store_true")
    ap.add_argument("--no-faiss", action="store_true")
    ap.add_argument("--no-fagin", action="store_true")
    ap.add_argument("--fagin-epsilon", dest="fagin_epsilon", default=None,
                    help="comma-separated Fagin additive halting-slack "
                         "values to sweep (e.g. '0,0.001,0.005,0.01'); the "
                         "index is built once and each epsilon gets its own "
                         "quality row and its own sequentially-timed Phase 5b "
                         "row. Applied as a GRID across every --fagin-schedule "
                         "(schedule × epsilon), and plotted as one latency-vs-"
                         "agreement line per schedule. 0 = exact. Single value "
                         "also accepted.")
    ap.add_argument("--fagin-schedule", dest="fagin_schedule", default=None,
                    help="Fagin sorted-access schedule(s), comma-separated to "
                         "compare several in one run (each gets its own quality "
                         "+ Phase 5b row). Choices: 'lockstep' (round-robin "
                         "over active dims, weight-blind), 'steepest' (advance "
                         "the dim with the largest marginal threshold drop), "
                         "and their norm-aware variants 'lockstep_norm'/"
                         "'steepest_norm' (printed GTA/GTASD) which add the "
                         "water-filling halting bound exploiting ||x||_2=1: "
                         "tighter, fewer accesses. Exact at epsilon=0 for all. "
                         "E.g. 'lockstep,lockstep_norm,steepest_norm'. "
                         "Default: lockstep.")
    ap.add_argument("--no-simdq", action="store_true",
                    help="skip the simdq asym b=2 + 1-bit hamming runs "
                         "(quality variants, thread sweeps, and Phase 5b rows); "
                         "the asym index is still built as the baselines' fp32 "
                         "vector source")
    ap.add_argument("--thread-sweeps", dest="thread_sweeps",
                    action="store_true",
                    help="run the simdq thread-scaling sweeps (Phase 1 asym "
                         "b=2 default+BLAS=1, Phase 6/8 Hamming SoA) and emit "
                         "the .sweeps.svg chart; off by default")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--force-encode", action="store_true",
                    help="re-encode queries even if the cached .npy exists")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="report path; {hostname}/{dataset}/{date} templated. "
                         "Default: docs/simdq_{hostname}_{dataset}.md")
    args = ap.parse_args()

    if args.latest:
        db = _resolve(args.results_db or DEFAULT_SETTINGS["results_db"])
        run_latest_report(
            db, _resolve(args.latest_csv) if args.latest_csv else None)
        return

    cfg = load_config(args)
    ds, st, models = cfg["dataset"], cfg["settings"], cfg["models"]

    run_id = uuid.uuid4().hex
    run_ts = datetime.now().isoformat(timespec="seconds")

    isa = detect_isa()
    print(f"# host  : {isa['hostname']}")
    print(f"# cpu   : {isa['model']}")
    print(f"# kernel: {isa['kernel_path']}")

    # 1) index dirs — reuse prebuilt ones, or build from dataset.corpus_file
    corpus_cache: dict = {}
    for m in models:
        ensure_indexes(m, ds, st, corpus_cache, args.force_build)

    # 2) dataset + query encoding
    texts: list[str] = []
    relevant: list[list[str]] = []
    if ds.get("queries_jsonl"):
        texts, _qids, relevant = load_dataset(ds)
    elif st["run_quality"]:
        sys.exit("quality phase needs dataset.queries_jsonl (gold labels) — "
                 "set it or pass --skip-quality")
    query_npys = [ensure_query_npy(m, ds["name"], texts, st, args.force_encode)
                  for m in models]

    # 3) quality
    metric_rows: list[dict] = []
    agree_rows: list[dict] = []
    if st["run_quality"]:
        metric_rows, agree_rows = run_quality(
            models, query_npys, relevant, st,
            int(ds["docid_join_components"]))

    # 4) speed — reuse the investigate_simdq_hardware sweeps on the same
    #    corpora (floats stored in the asym index dirs) + encoded queries.
    sources = [_load_source(str(_resolve(m["asym_index"])), 0, None,
                            int(st["seed"])) for m in models]
    corpus_sizes = [int(v.shape[0]) for _, v in sources]
    query_specs = [str(p) for p in query_npys]
    n_needed = int(st["warmup"]) + int(st["speed_queries"])
    for m, p in zip(models, query_npys):
        n_have = np.load(p, mmap_mode="r").shape[0]
        if n_have < n_needed:
            sys.exit(f"{m['tag']}: {p} has {n_have} queries; speed sweep "
                     f"needs warmup+speed_queries={n_needed}")

    p1_default = p1_blas1 = p68 = None
    head = None
    if st["run_speed"]:
        if st.get("run_simdq", True) and st.get("run_thread_sweeps", False):
            common = dict(threads=st["threads"],
                          queries=int(st["speed_queries"]),
                          warmup=int(st["warmup"]), K=int(st["top_k"]),
                          alpha=int(st["alpha"]), seed=int(st["seed"]))
            p1_default = run_sweep_phase("Phase 1 — asym b=2", "asymmetric", 2,
                                         sources, query_specs, blas_threads=None,
                                         **common)
            p1_blas1 = run_sweep_phase("Phase 1 — asym b=2", "asymmetric", 2,
                                       sources, query_specs, blas_threads=1,
                                       **common)
            p68 = run_sweep_phase("Phase 6/8 — Hamming SoA", "hamming", None,
                                  sources, query_specs, blas_threads=None,
                                  **common)
        if (st["run_milvus"] or st["run_faiss"] or st.get("run_fagin", True)
                or st.get("run_simdq", True)):
            try:
                head = bench_head_to_head(sources, query_specs, st)
            except Exception as e:
                print(f"# Phase 5b skipped: {e}", file=sys.stderr)

    # 5) report
    out_tmpl = cfg["output"] or "docs/simdq_{hostname}_{dataset}.md"
    out_path = _resolve(out_tmpl.format(hostname=isa["hostname"],
                                        dataset=ds["name"],
                                        date=date.today().isoformat()))
    cmdline = f"python {SCRIPT_REL} " + " ".join(
        shlex.quote(a) for a in sys.argv[1:])
    write_report(out_path, cfg, isa, corpus_sizes,
                 len(texts) if texts else int(st["speed_queries"]),
                 query_npys, metric_rows, agree_rows,
                 p1_default, p1_blas1, p68, head, cmdline)
    print(f"\n# report -> {out_path}")

    raw = {"config": {"dataset": ds, "settings": st,
                      "models": models, "output": out_tmpl},
           "isa": {k: v for k, v in isa.items() if k != "so_counts"},
           "quality": metric_rows, "agreement": agree_rows,
           "speed": {
               "phase1_default": [(l, d, o) for l, d, o in (p1_default or [])],
               "phase1_blas1": [(l, d, o) for l, d, o in (p1_blas1 or [])],
               "phase68": [(l, d, o) for l, d, o in (p68 or [])],
               "phase5b": head},
           "date": date.today().isoformat()}
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(raw, indent=2, default=str))
    print(f"# raw    -> {json_path}")

    db_setting = st.get("results_db")
    if db_setting:
        db_path = _resolve(db_setting)
        try:
            n = write_results_db(db_path, cfg, isa, run_id, run_ts,
                                 metric_rows, agree_rows, head or [])
            print(f"# results-db -> {db_path}  (+{n} rows)")
        except Exception as e:
            print(f"# results-db skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
