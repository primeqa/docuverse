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

Synthetic random unit vectors are used throughout: both Milvus FLAT and the
simdq scan are data-independent for speed, so synthetic numbers match real-NQ
numbers within noise (verified in docs/simdq_768_investigation_avx2.md).

Usage:
    python scripts/investigate_simdq_hardware.py                  # quick smoke
    python scripts/investigate_simdq_hardware.py --N 276007       # full sweep
    python scripts/investigate_simdq_hardware.py --no-milvus      # skip Milvus
    python scripts/investigate_simdq_hardware.py --rebuild        # build .so first

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

def sweep_one(family, b, dim, N, threads, queries, warmup, K, alpha, seed):
    """Median ms/query at each thread count for one (family, b, dim, N)."""
    from docuverse.engines.retrieval.simdq.simdq_index import SimdqIndex
    rng = np.random.default_rng(seed)
    vecs = _unit(N, dim, rng)
    qs = _unit(queries + warmup, dim, rng)
    idx = SimdqIndex.build(
        vecs, family=family, b=b, d=None,
        projection="identity", store_floats=True,
    )
    K_prime = max(K, min(K * alpha, 256))
    out: dict[int, float] = {}
    for t in threads:
        for i in range(warmup):
            idx.search(qs[i], K=K, K_prime=K_prime, num_threads=t)
        out[t] = _median_serial(
            lambda q: idx.search(q, K=K, K_prime=K_prime, num_threads=t),
            qs[warmup:warmup + queries],
        )
    return out


def run_sweep_phase(label, family, b, dims, N, threads, queries, warmup,
                    K, alpha, seed, blas_threads=None):
    """Run a sweep across dims, optionally pinning BLAS to `blas_threads`."""
    from threadpoolctl import threadpool_limits
    print(f"\n## {label}  (BLAS={'default' if blas_threads is None else blas_threads})")
    ctx = (threadpool_limits(limits=blas_threads, user_api="blas")
           if blas_threads is not None else _NullCtx())
    rows: list[tuple[int, dict[int, float]]] = []
    with ctx:
        for D in dims:
            out = sweep_one(family, b, D, N, threads, queries, warmup,
                            K, alpha, seed)
            rows.append((D, out))
            cells = "  ".join(f"t={t if t else 'all':>3}={out[t]:7.3f}"
                              for t in threads)
            print(f"  D={D:4d}  {cells}")
    return rows


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---------- Phase 5b: head-to-head vs Milvus FLAT ------------------------- #

def bench_head_to_head(uri, N, dims, queries, K, alpha, workers, warmup, seed):
    """Same 16-conc harness as scripts/bench_simdq_vs_milvus.py, but synthetic
    in-process indexes — runs Milvus, simdq b=2, simdq Hamming at each dim."""
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
    for D in dims:
        vecs = _unit(N, D, rng)
        qs = _unit(queries, D, rng)

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
            m_call(qs[i])
        try:
            m_cc = _conc_ms_per_q(m_call, qs, workers)
        finally:
            cli.drop_collection(coll)

        # --- simdq b=2 ---
        idx2 = SimdqIndex.build(vecs, family="asymmetric", b=2, d=None,
                                projection="identity", store_floats=True)
        for i in range(warmup):
            idx2.search(qs[i], K=K, K_prime=K_prime, num_threads=1)
        s2_cc = _conc_ms_per_q(
            lambda q: idx2.search(q, K=K, K_prime=K_prime, num_threads=1),
            qs, workers)

        # --- simdq Hamming ---
        idxh = SimdqIndex.build(vecs, family="hamming", b=None, d=None,
                                projection="identity", store_floats=True)
        for i in range(warmup):
            idxh.search(qs[i], K=K, K_prime=K_prime, num_threads=1)
        sh_cc = _conc_ms_per_q(
            lambda q: idxh.search(q, K=K, K_prime=K_prime, num_threads=1),
            qs, workers)

        row = {"D": D, "milvus": m_cc, "asym_b2": s2_cc, "hamming": sh_cc}
        out.append(row)
        print(f"  D={D:4d}  Milvus={m_cc:6.2f}  b=2={s2_cc:6.2f}  "
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
      f"on {date.today().isoformat()}. N={params['N']}, K={params['K']}, "
      f"K'={params['K_prime']}, queries={params['queries']}, "
      f"workers={params['workers']}._")
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
        hdr = "| dim | " + " | ".join(
            f"t={t if t else 'all'}" for t in threads) + " |"
        sep = "|---" * (len(threads) + 1) + "|"
        body = []
        for D, out in rows:
            cells = " | ".join(f"{out[t]:.2f}" for t in threads)
            body.append(f"| {D} | {cells} |")
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
        A("| dim | Milvus FLAT | simdq b=2 | simdq 1-bit | b=2/1-bit |")
        A("|---|---|---|---|---|")
        for row in head:
            A(f"| {row['D']} | {row['milvus']:.2f} | {row['asym_b2']:.2f} | "
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
                    help="corpus size (276007 to mirror the reference run)")
    ap.add_argument("--dims", default="384,768",
                    help="comma-separated dims to sweep")
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
        "Phase 1 — asym b=2", "asymmetric", 2, dims, args.N, threads,
        args.queries, args.warmup, args.top_k, args.alpha, args.seed,
        blas_threads=None)
    p1_blas1 = run_sweep_phase(
        "Phase 1 — asym b=2", "asymmetric", 2, dims, args.N, threads,
        args.queries, args.warmup, args.top_k, args.alpha, args.seed,
        blas_threads=1)
    p68 = run_sweep_phase(
        "Phase 6/8 — Hamming SoA", "hamming", None, dims, args.N, threads,
        args.queries, args.warmup, args.top_k, args.alpha, args.seed,
        blas_threads=None)

    head = None
    if not args.no_milvus:
        try:
            head = bench_head_to_head(
                args.milvus_uri, args.N, dims, args.queries,
                args.top_k, args.alpha, args.workers, args.warmup, args.seed)
        except Exception as e:
            print(f"# Phase 5b skipped: {e}", file=sys.stderr)
            head = None

    out_path = (Path(args.out) if args.out
                else REPO_ROOT / "docs" /
                f"simdq_{isa['hostname']}_investigation.md")
    write_report(out_path, isa, p1_default, p1_blas1, p68, head,
                 params=dict(N=args.N, K=args.top_k, K_prime=K_prime,
                             queries=args.queries, workers=args.workers,
                             threads=threads))
    print(f"\n# report -> {out_path}")


if __name__ == "__main__":
    main()
