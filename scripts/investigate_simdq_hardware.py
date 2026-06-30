#!/usr/bin/env python
"""investigate_simdq_hardware.py — end-to-end SIMD-Q hardware-portability bench.

Replays the full investigation from docs/simdq_768_investigation_avx2.md on
whichever machine it's invoked on, and writes a structured markdown report:

  1. ISA detection (lscpu flags + objdump of the compiled .so).
  2. Phase 1  — asymmetric b=2 dim x thread sweep, default BLAS and BLAS=1.
  3. Phase 6/8 — Hamming SoA dim x thread sweep.
  4. Phase 5b — (optional) Milvus FLAT vs simdq b=2 vs Hamming head-to-head,
                 16-way concurrent harness. Requires a Milvus standalone at
                 --milvus-uri (default http://localhost:19530).

By default, synthetic random unit vectors are used. Both Milvus FLAT and the
simdq scan are data-independent for *speed*, so synthetic numbers match real
data within noise (verified in docs/simdq_768_investigation_avx2.md). Pass
--vectors to swap in a real DB corpus and --queries-file to swap in
pre-encoded query vectors. Both accept .npy files (shape N×D, float32) or, for
--vectors, a SimdqIndex directory built with store_floats=True. Each source
becomes one row in the result tables; --N and --dims are ignored in that mode.

Usage:
    python scripts/investigate_simdq_hardware.py                       # quick smoke
    python scripts/investigate_simdq_hardware.py --N 276007            # full synthetic
    python scripts/investigate_simdq_hardware.py --no-milvus           # skip Milvus
    python scripts/investigate_simdq_hardware.py --rebuild             # build .so first
    # Real DB corpus, synthetic queries (queries are data-independent for speed):
    python scripts/investigate_simdq_hardware.py \
        --vectors path/to/nq97m.npy,path/to/nq311m.npy
    # Existing simdq index dirs (built with store_floats=True):
    python scripts/investigate_simdq_hardware.py \
        --vectors experiments/nq_new/simq_data/nq_dev-...-granite97m-...,\
experiments/nq_new/simq_data/nq_dev-...-granite311m-...
    # Real DB + real queries, paired 1:1:
    python scripts/investigate_simdq_hardware.py \
        --vectors db97m.npy,db311m.npy \
        --queries-file q97m.npy,q311m.npy

The report file is written to docs/simdq_<hostname>_investigation.md so
multiple machines' replays can sit side-by-side.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

# Defer simdq import: --rebuild needs to happen first.


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMDQ_DIR = REPO_ROOT / "docuverse" / "engines" / "retrieval" / "simdq"


# ---------- ISA detection -------------------------------------------------- #

def _read_cpuinfo() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in ("model name", "flags") and k not in info:
                info[k] = v
    except FileNotFoundError:
        pass
    return info


def _lscpu_cache() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        r = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            for key in ("L1d cache", "L2 cache", "L3 cache",
                        "Core(s) per socket", "Socket(s)", "Thread(s) per core"):
                if line.startswith(key):
                    out[key] = line.split(":", 1)[1].strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return out


def _objdump_isa_counts(so_path: Path) -> dict[str, int]:
    """Disassemble the simdq .so and count ISA-marker mnemonics."""
    counts = {"ymm": 0, "zmm": 0, "vfmadd": 0,
              "vpshufb": 0, "vpmultishift": 0, "vpdpbusd": 0}
    if not so_path.exists():
        return counts
    try:
        r = subprocess.run(["objdump", "-d", str(so_path)],
                           capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.SubprocessError):
        return counts
    for line in r.stdout.splitlines():
        for k in counts:
            if k in line:
                counts[k] += 1
    return counts


def detect_isa() -> dict:
    cpuinfo = _read_cpuinfo()
    flags = set(cpuinfo.get("flags", "").split())
    cache = _lscpu_cache()
    so_glob = sorted(SIMDQ_DIR.glob("_simdq_native*.so"))
    so_counts = _objdump_isa_counts(so_glob[0]) if so_glob else {}
    return {
        "model": cpuinfo.get("model name", "<unknown>"),
        "hostname": socket.gethostname(),
        "has_avx2": "avx2" in flags,
        "has_avx512f": "avx512f" in flags,
        "has_avx512vbmi": "avx512vbmi" in flags,
        "has_gfni": "gfni" in flags,
        "has_vpopcntdq": "avx512_vpopcntdq" in flags,
        "fma": "fma" in flags,
        "cache": cache,
        "compiled_so": str(so_glob[0]) if so_glob else None,
        "so_counts": so_counts,
        "kernel_path": (
            "AVX-512 (VBMI/Lever-1)"
            if so_counts.get("vpmultishift", 0) > 0
            else ("AVX-512" if so_counts.get("zmm", 0) > 0
                  else ("AVX2-FMA" if so_counts.get("ymm", 0) > 0 else "<scalar?>"))
        ),
    }


# ---------- shared utilities ---------------------------------------------- #

def _unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal((n, d), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x


def _load_source(spec: str | None, default_N: int, default_D: int | None,
                 seed: int) -> tuple[str, np.ndarray]:
    """Resolve a dataset spec into (label, vecs (np.float32, contiguous)).

    spec=None       -> synthetic random unit vectors at (default_N, default_D)
    spec="path.npy" -> np.load(path); shape (N, D)
    spec="path/"    -> SimdqIndex.load(path).floats_mmap (must have_floats=True)

    Both Milvus FLAT and the simdq scan are data-independent for *speed*, so
    `.npy` files / simdq indexes are the natural way to plug a real corpus
    (e.g. NQ encoder outputs) into the same sweep without re-encoding.
    """
    if spec is None:
        assert default_D is not None
        rng = np.random.default_rng(seed)
        return (f"synthetic-{default_D}d",
                _unit(default_N, default_D, rng))
    p = Path(spec)
    if p.suffix == ".npy":
        vecs = np.load(p)
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32, copy=False)
        vecs = np.ascontiguousarray(vecs)
        return (p.stem, vecs)
    # Otherwise treat as a SimdqIndex directory.
    from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
    idx = SimdqIndex.load(p)
    if not idx.has_floats:
        raise ValueError(
            f"simdq index {p} has no stored floats — rebuild with "
            f"store_floats=True, or pass an .npy file instead.")
    vecs = np.ascontiguousarray(
        np.asarray(idx.floats_mmap, dtype=np.float32))
    return (p.name, vecs)


def _load_queries(spec: str | None, D: int, n_required: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Resolve a query spec to a contiguous float32 (n_required, D) array.

    spec=None       -> synthetic random unit vectors at (n_required, D)
    spec="path.npy" -> np.load(path); must be float32 (auto-cast) with width D;
                       the first n_required rows are returned. If the file has
                       fewer queries than n_required, it errors.
    """
    if spec is None:
        return _unit(n_required, D, rng)
    p = Path(spec)
    q = np.load(p)
    if q.ndim != 2:
        raise ValueError(f"queries file {p} must be 2D, got shape {q.shape}")
    if q.shape[1] != D:
        raise ValueError(
            f"queries file {p} width={q.shape[1]}, but source D={D}; "
            f"queries must match the corpus dim.")
    if q.shape[0] < n_required:
        raise ValueError(
            f"queries file {p} has {q.shape[0]} queries; need {n_required}.")
    if q.dtype != np.float32:
        q = q.astype(np.float32, copy=False)
    return np.ascontiguousarray(q[:n_required])


def _resolve_query_specs(specs_str: str | None, n_sources: int) -> list[str | None]:
    """Pair --queries-file (None|single path|comma-list) with --vectors entries.

    None              -> [None, None, ...]  (all synthetic)
    "path.npy"        -> [path] * n_sources  (same file reused)
    "path1,path2,..." -> exactly n_sources entries; an empty slot means synthetic
    """
    if not specs_str:
        return [None] * n_sources
    parts = [p.strip() or None for p in specs_str.split(",")]
    if len(parts) == 1:
        return parts * n_sources
    if len(parts) != n_sources:
        raise ValueError(
            f"--queries-file lists {len(parts)} paths, but --vectors lists "
            f"{n_sources} sources. Either pass one shared file or one per source.")
    return parts


def _median_serial(call, qs) -> float:
    t = np.empty(len(qs), dtype=np.float64)
    for i in range(len(qs)):
        t0 = time.perf_counter()
        call(qs[i])
        t[i] = (time.perf_counter() - t0) * 1e3
    return float(np.median(t))


def _conc_ms_per_q(call, qs, workers: int) -> float:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(call, qs))
    return (time.perf_counter() - t0) / len(qs) * 1e3


# ---------- Phase 1 / 6/8: dim x thread sweep ----------------------------- #

def sweep_one(family, b, vecs, qs_pool, threads, warmup, queries, K, alpha):
    """Median ms/query at each thread count for one (family, b, source).

    vecs: (N, D) corpus, contiguous float32.
    qs_pool: (warmup + queries, D) query batch, the same one reused across t.
    """
    from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
    idx = SimdqIndex.build(
        vecs, family=family, b=b, d=None,
        projection="identity", store_floats=True,
    )
    K_prime = max(K, min(K * alpha, 256))
    out: dict[int, float] = {}
    for t in threads:
        for i in range(warmup):
            idx.search(qs_pool[i], K=K, K_prime=K_prime, num_threads=t)
        out[t] = _median_serial(
            lambda q: idx.search(q, K=K, K_prime=K_prime, num_threads=t),
            qs_pool[warmup:warmup + queries],
        )
    return out


def run_sweep_phase(label, family, b, sources, query_specs, threads, queries,
                    warmup, K, alpha, seed, blas_threads=None):
    """Run a sweep across sources, optionally pinning BLAS to `blas_threads`.

    sources: list of (label, vecs) tuples.
    query_specs: list parallel to sources; each entry is None (synthetic) or
        a path to a .npy file with (>=warmup+queries, D) pre-encoded queries.
    """
    from threadpoolctl import threadpool_limits
    print(f"\n## {label}  (BLAS={'default' if blas_threads is None else blas_threads})")
    ctx = (threadpool_limits(limits=blas_threads, user_api="blas")
           if blas_threads is not None else _NullCtx())
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, int, dict[int, float]]] = []
    with ctx:
        for (src_label, vecs), q_spec in zip(sources, query_specs):
            D = int(vecs.shape[1])
            qs_pool = _load_queries(q_spec, D, warmup + queries, rng)
            out = sweep_one(family, b, vecs, qs_pool, threads,
                            warmup, queries, K, alpha)
            rows.append((src_label, D, out))
            cells = "  ".join(f"t={t if t else 'all':>3}={out[t]:7.3f}"
                              for t in threads)
            qtag = "real" if q_spec else "synth"
            print(f"  {src_label:<28s} (D={D:4d}, q={qtag})  {cells}")
    return rows


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---------- Phase 5b: head-to-head vs Milvus FLAT ------------------------- #

def bench_head_to_head(uri, sources, query_specs, queries, K, alpha,
                       workers, warmup, seed):
    """Same 16-conc harness as scripts/bench_simdq_vs_milvus.py, with one
    Milvus FLAT + simdq b=2 + simdq Hamming triple per source.

    query_specs: parallel to sources; None = synthetic queries, else .npy path.
    """
    try:
        from pymilvus import MilvusClient, DataType
    except ImportError as e:
        print(f"  pymilvus unavailable: {e}", file=sys.stderr)
        return None
    from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex

    print(f"\n## Phase 5b — head-to-head vs Milvus FLAT  "
          f"(uri={uri}, workers={workers})")
    K_prime = max(K, min(K * alpha, 256))
    rng = np.random.default_rng(seed)
    cli = MilvusClient(uri=uri)
    out: list[dict] = []
    for (src_label, vecs), q_spec in zip(sources, query_specs):
        N, D = vecs.shape
        # warmup uses the first `warmup` slots, timed run uses the next `queries`
        qs_pool = _load_queries(q_spec, D, warmup + queries, rng)
        qs = qs_pool[warmup:warmup + queries]

        # --- Milvus FLAT ---
        coll = f"hwbench_{D}_{N}"
        if cli.has_collection(coll):
            cli.drop_collection(coll)
        schema = cli.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("v", DataType.FLOAT_VECTOR, dim=D)
        cli.create_collection(coll, schema=schema)
        for s in range(0, N, 20000):
            e = min(s + 20000, N)
            cli.insert(coll, [{"id": i, "v": vecs[i]} for i in range(s, e)])
        ip = cli.prepare_index_params()
        ip.add_index(field_name="v", index_type="FLAT", metric_type="IP")
        cli.create_index(coll, ip)
        cli.load_collection(coll)
        time.sleep(2)
        sp = {"metric_type": "IP"}
        m_call = lambda q: cli.search(
            coll, [q.tolist()], limit=K, search_params=sp, anns_field="v")
        for i in range(warmup):
            m_call(qs_pool[i])
        try:
            m_cc = _conc_ms_per_q(m_call, qs, workers)
        finally:
            cli.drop_collection(coll)

        # --- simdq b=2 ---
        idx2 = SimdqIndex.build(vecs, family="asymmetric", b=2, d=None,
                                projection="identity", store_floats=True)
        for i in range(warmup):
            idx2.search(qs_pool[i], K=K, K_prime=K_prime, num_threads=1)
        s2_cc = _conc_ms_per_q(
            lambda q: idx2.search(q, K=K, K_prime=K_prime, num_threads=1),
            qs, workers)

        # --- simdq Hamming ---
        idxh = SimdqIndex.build(vecs, family="hamming", b=None, d=None,
                                projection="identity", store_floats=True)
        for i in range(warmup):
            idxh.search(qs_pool[i], K=K, K_prime=K_prime, num_threads=1)
        sh_cc = _conc_ms_per_q(
            lambda q: idxh.search(q, K=K, K_prime=K_prime, num_threads=1),
            qs, workers)

        row = {"label": src_label, "D": D, "N": N,
               "queries": ("real" if q_spec else "synth"),
               "milvus": m_cc, "asym_b2": s2_cc, "hamming": sh_cc}
        out.append(row)
        qtag = "real" if q_spec else "synth"
        print(f"  {src_label:<28s} (D={D:4d},N={N},q={qtag})  "
              f"Milvus={m_cc:6.2f}  b=2={s2_cc:6.2f}  "
              f"1-bit={sh_cc:6.2f}  (b=2/1-bit ratio {s2_cc/sh_cc:5.1f}x)")
    return out


# ---------- report writer -------------------------------------------------- #

def write_report(report_path, isa, p1_default, p1_blas1, p68, head, params):
    """Render the markdown report — mirrors docs/simdq_768_investigation_avx2.md."""
    lines: list[str] = []
    A = lines.append
    A(f"# simdq hardware investigation — {isa['hostname']}")
    A("")
    A(f"_Auto-generated by `scripts/investigate_simdq_hardware.py` "
      f"on {date.today().isoformat()}. dataset={params['dataset']}, "
      f"K={params['K']}, K'={params['K_prime']}, "
      f"queries={params['queries']}, workers={params['workers']}._")
    A("")
    A("## Environment")
    A("")
    A(f"- CPU: **{isa['model']}**")
    for k in ("Socket(s)", "Core(s) per socket", "Thread(s) per core",
              "L1d cache", "L2 cache", "L3 cache"):
        if isa["cache"].get(k):
            A(f"- {k}: {isa['cache'][k]}")
    flags = []
    for k, label in [("has_avx2", "AVX2"), ("has_avx512f", "AVX-512F"),
                     ("has_avx512vbmi", "AVX-512 VBMI"),
                     ("has_gfni", "GFNI"), ("has_vpopcntdq", "VPOPCNTDQ"),
                     ("fma", "FMA")]:
        flags.append(f"{label}: {'yes' if isa[k] else 'NO'}")
    A(f"- SIMD: {', '.join(flags)}")
    A(f"- Compiled `.so`: `{isa['compiled_so']}`")
    A(f"- Kernel path (from disassembly): **{isa['kernel_path']}** "
      f"(ymm={isa['so_counts'].get('ymm', 0)}, "
      f"zmm={isa['so_counts'].get('zmm', 0)}, "
      f"vfmadd={isa['so_counts'].get('vfmadd', 0)}, "
      f"vpmultishift={isa['so_counts'].get('vpmultishift', 0)})")
    A(f"- Python: {platform.python_version()}, numpy: {np.__version__}")
    A("")

    def _sweep_md_table(rows, threads):
        hdr = "| source | dim | " + " | ".join(
            f"t={t if t else 'all'}" for t in threads) + " |"
        sep = "|---" * (len(threads) + 2) + "|"
        body = []
        for label, D, out in rows:
            cells = " | ".join(f"{out[t]:.2f}" for t in threads)
            body.append(f"| {label} | {D} | {cells} |")
        return "\n".join([hdr, sep] + body)

    A("## Phase 1 — asymmetric b=2 dim × thread sweep")
    A("")
    A("Median ms/query for `SimdqIndex.search` (projection + native scan + "
      "fp32 rescore). Identity projection, so the Phase-3 fix means there's "
      "no `W@q` to oversubscribe BLAS.")
    A("")
    A("### Default BLAS")
    A("")
    A(_sweep_md_table(p1_default, params["threads"]))
    A("")
    A("### `OPENBLAS_NUM_THREADS=1` (via `threadpool_limits`)")
    A("")
    A(_sweep_md_table(p1_blas1, params["threads"]))
    A("")
    A("> On hardware where the Phase-3 identity-skip is active, BLAS=1 "
      "should change ~nothing. If it does change a lot, an unintended "
      "GEMV is still firing per query.")
    A("")
    A("## Phase 6/8 — Hamming SoA dim × thread sweep")
    A("")
    A(_sweep_md_table(p68, params["threads"]))
    A("")

    if head:
        A("## Phase 5b — head-to-head vs Milvus FLAT (16-way concurrent)")
        A("")
        A("| source | dim | N | queries | Milvus FLAT | simdq b=2 | simdq 1-bit | b=2/1-bit |")
        A("|---|---|---|---|---|---|---|---|")
        for row in head:
            A(f"| {row['label']} | {row['D']} | {row['N']} | "
              f"{row.get('queries', 'synth')} | "
              f"{row['milvus']:.2f} | {row['asym_b2']:.2f} | "
              f"{row['hamming']:.2f} | "
              f"{row['asym_b2']/row['hamming']:.1f}× |")
        A("")
        A("ms/query, 16 query workers, each scan single-threaded (matches "
          "`scripts/bench_simdq_vs_milvus.py`).")
        A("")
    else:
        A("## Phase 5b — head-to-head vs Milvus FLAT")
        A("")
        A("_Skipped (`--no-milvus` or Milvus unreachable)._")
        A("")

    A("## How to compare against the reference runs")
    A("")
    A("- AVX-512 9950X3D reference: `docs/simdq_768_investigation.md`")
    A("- AVX2 5955WX reference: `docs/simdq_768_investigation_avx2.md`")
    A("")
    A("The big things to look for: does `t=1 → t=all` scale cleanly "
      "(Phase-3 fix in place), does Hamming/asym-b2 ratio match the silicon "
      "(2× on AVX-512+VBMI, ~40× on plain AVX2), and is Milvus FLAT close to "
      "5/12 ms at 384d/768d (memory + RPC bound).")
    report_path.write_text("\n".join(lines) + "\n")


# ---------- main ---------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=50000,
                    help="synthetic corpus size (ignored if --vectors is set)")
    ap.add_argument("--dims", default="384,768",
                    help="synthetic dims to sweep (ignored if --vectors is set)")
    ap.add_argument("--vectors", default=None,
                    help="comma-separated dataset specs to replace synthetic data. "
                         "Each spec is either a .npy file (shape N×D, float32) or "
                         "a SimdqIndex directory built with store_floats=True. "
                         "Example: --vectors path/to/nq97m.npy,path/to/nq311m.npy")
    ap.add_argument("--queries-file", default=None, dest="queries_file",
                    help="path(s) to a .npy of pre-encoded query vectors "
                         "(>= warmup+queries rows, width matching the source D). "
                         "Pass a single path to reuse it across all sources, or a "
                         "comma-separated list to pair 1:1 with --vectors entries. "
                         "An empty slot in the list means 'synthetic for this source'.")
    ap.add_argument("--threads", default="1,2,4,8,16,0",
                    help="comma-separated num_threads (0 = all cores)")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--alpha", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent query workers for Phase 5b")
    ap.add_argument("--milvus-uri", default="http://localhost:19530")
    ap.add_argument("--no-milvus", action="store_true",
                    help="skip Phase 5b (Milvus head-to-head)")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the .so via `python setup.py build_ext --inplace`")
    ap.add_argument("--out", default=None,
                    help="report path (default: docs/simdq_<hostname>_investigation.md)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    threads = [int(x) for x in args.threads.split(",")]
    K_prime = max(args.top_k, min(args.top_k * args.alpha, 256))

    if args.vectors:
        sources = [_load_source(s.strip(), args.N, None, args.seed)
                   for s in args.vectors.split(",") if s.strip()]
        print(f"# dataset: {len(sources)} loaded source(s)")
        for lbl, vecs in sources:
            print(f"#   {lbl}  shape={tuple(vecs.shape)}  dtype={vecs.dtype}")
    else:
        sources = [_load_source(None, args.N, D, args.seed) for D in dims]
        print(f"# dataset: synthetic, N={args.N}, dims={dims}")
    query_specs = _resolve_query_specs(args.queries_file, len(sources))
    if any(query_specs):
        for (lbl, _), qs in zip(sources, query_specs):
            print(f"#   queries[{lbl}] = "
                  f"{'real (' + qs + ')' if qs else 'synthetic'}")

    if args.rebuild:
        print("# rebuilding native extension via setup.py build_ext --inplace")
        r = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout); print(r.stderr, file=sys.stderr)
            sys.exit(f"setup.py build_ext failed (rc={r.returncode})")
        print("# rebuild ok")

    isa = detect_isa()
    print(f"# host  : {isa['hostname']}")
    print(f"# cpu   : {isa['model']}")
    print(f"# flags : AVX2={isa['has_avx2']}  AVX-512F={isa['has_avx512f']}  "
          f"VBMI={isa['has_avx512vbmi']}  GFNI={isa['has_gfni']}")
    print(f"# kernel: {isa['kernel_path']}  "
          f"(ymm={isa['so_counts'].get('ymm', 0)}, "
          f"zmm={isa['so_counts'].get('zmm', 0)}, "
          f"vpmultishift={isa['so_counts'].get('vpmultishift', 0)})")

    p1_default = run_sweep_phase(
        "Phase 1 — asym b=2", "asymmetric", 2, sources, query_specs, threads,
        args.queries, args.warmup, args.top_k, args.alpha, args.seed,
        blas_threads=None)
    p1_blas1 = run_sweep_phase(
        "Phase 1 — asym b=2", "asymmetric", 2, sources, query_specs, threads,
        args.queries, args.warmup, args.top_k, args.alpha, args.seed,
        blas_threads=1)
    p68 = run_sweep_phase(
        "Phase 6/8 — Hamming SoA", "hamming", None, sources, query_specs,
        threads, args.queries, args.warmup, args.top_k, args.alpha, args.seed,
        blas_threads=None)

    head = None
    if not args.no_milvus:
        try:
            head = bench_head_to_head(
                args.milvus_uri, sources, query_specs, args.queries,
                args.top_k, args.alpha, args.workers, args.warmup, args.seed)
        except Exception as e:
            print(f"# Phase 5b skipped: {e}", file=sys.stderr)
            head = None

    out_path = (Path(args.out) if args.out
                else REPO_ROOT / "docs" /
                f"simdq_{isa['hostname']}_investigation.md")
    dataset_label = (f"synthetic N={args.N}" if not args.vectors
                     else f"{len(sources)} loaded source(s) from --vectors")
    write_report(out_path, isa, p1_default, p1_blas1, p68, head,
                 params=dict(dataset=dataset_label,
                             K=args.top_k, K_prime=K_prime,
                             queries=args.queries, workers=args.workers,
                             threads=threads))
    print(f"\n# report -> {out_path}")


if __name__ == "__main__":
    main()
