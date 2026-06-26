# `simdq` — In-tree SIMD-quantized retrieval engine (v1 design)

**Date:** 2026-06-15
**Status:** Design — pending implementation plan
**Branch base:** `v0.1.2`

## 1. Motivation & goal

DocUVerse currently delegates dense retrieval to external engines (Milvus,
Elasticsearch, LanceDB, FAISS, ChromaDB). For BEIR-scale evaluations
(≤ 10M vectors, typically ~5M) we want an **in-process baseline** that:

- Runs **as fast as physically possible** on a single host with SIMD CPU code,
  with no network or RPC overhead.
- Lets us explore the **bits-per-dimension × projection** design space that
  matters for binary-quantized retrieval — e.g. is asymmetric 2-bit at full
  dim better than 1-bit + float rescore at the same memory footprint?
- Produces head-to-head comparisons with the float ground-truth and with
  the binary recipes the existing engines expose.

The kernel work in `/home/raduf/sandbox2/embedding_hashes` (1-bit symmetric
Hamming over 768-bit codes, top-1, ~2.84B comparisons/s on 50M vectors via
`hb5`) is the starting point. v1 brings that work into DocUVerse and
generalises it to top-K + asymmetric multi-bit scans.

The headline research question this engine is built to answer:

> Does asymmetric 2-bit / 4-bit scalar quantization (à la ASH, Tepper &
> Willke 2026 internal pre-print) beat 1-bit-plus-float-rescore at the same
> byte budget?

## 2. Scope

### In scope (v1)

- New retrieval engine `simdq` under `docuverse/engines/retrieval/simdq/`.
- C/SIMD kernels for **symmetric 1-bit Hamming** (carried over from `hb5`,
  generalised to top-K) and **asymmetric scans** at `b ∈ {1, 2, 4}` bits per
  reduced dimension.
- Compile-time templated kernels for `D ∈ {384, 768, 1024, 1536}`. Reduced
  dimension `d ∈ {D, D/2}` only (so the kernel template still covers it).
- **Identity** and **random orthogonal** (`W = QR-of-Gaussian`) projections.
- Optional float rescore tier: per-vector fp16 vectors stored on disk and
  mmap'd at search time. Configurable via `store_floats: true|false`.
- Python C-extension binding (CPython C-API, no pybind11/cffi dependency).
- DocUVerse `SearchEngine` wrapper with config dataclass, dispatchable from
  the existing `ingest_and_test` CLI.
- Standalone C benchmark binaries preserved (renamed `bench_hamming`,
  `bench_asym_b1/2/4`).
- Test matrix covering both SIMD paths (AVX-512F native, AVX2 forced) and
  every `(family, b, D)` combination.

### Out of scope (v2 or later)

- **Learned projection** (full ASH training: alternating Procrustes update,
  PCA init). The harness exists to find out whether learning is *needed*;
  if random orthogonal already wins, ASH training adds little. Deferred.
- **Outer index** (IVF / HNSW). At ≤10M with the kernel's throughput, brute
  force already beats IVF/HNSW + float for our scale. Adding an outer index
  is a separate research question.
- **GPU kernels.** CPU SIMD only.
- **Index updates / appends.** Codes are SoA-packed contiguously; appending
  requires rewriting. Document as "rebuild" in CLI help.
- **Encoder dimensionality outside `{384, 768, 1024, 1536}`.** OpenAI-3072
  and other rare sizes fall back to the closest supported size with a
  caller-side projection, or are unsupported in v1.
- **Per-dim quantization scale.** Adds `d` floats per code (1.5× index size
  at b=4) for marginal published recall gain. Defer.

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DocUVerse SearchEngine wrapper (Python)                    │
│  docuverse/engines/retrieval/simdq/simdq_engine.py          │
│  - ingests texts → calls EmbeddingFunction → builds index   │
│  - search(query) → encode → projection → quantize → scan    │
│  - drives optional rescore stage                            │
└─────────────────────────────────────────────────────────────┘
                            ↓ Python ext
┌─────────────────────────────────────────────────────────────┐
│  Index API (Python binding to C extension)                  │
│  docuverse/engines/retrieval/simdq/_native/                 │
│  - SimdqIndex.build(vectors, b, d, W, store_floats)         │
│  - SimdqIndex.load(path) / .save(path)                      │
│  - SimdqIndex.search(query_float, K, K_prime)  → top-K      │
│  - SimdqIndex.search_batch(queries, K, K_prime)             │
└─────────────────────────────────────────────────────────────┘
                            ↓ C calls
┌─────────────────────────────────────────────────────────────┐
│  Kernels (C, AVX-512F + AVX2 fallback, OpenMP)              │
│  - scan_hamming_<D>           (existing hb5 work, repackaged)│
│  - scan_asym_b1_<D>           (new: float×1-bit asymmetric)  │
│  - scan_asym_b2_<D>           (new: float×2-bit asymmetric)  │
│  - scan_asym_b4_<D>           (new: float×4-bit asymmetric)  │
│  - topk_merge                 (per-thread heap + final merge)│
│  - rescore_fp16               (small SIMD dot product)       │
│  - D ∈ {384, 768, 1024, 1536}, fully unrolled per template   │
└─────────────────────────────────────────────────────────────┘
```

Three boundaries:

1. **Engine ↔ Index.** `SimdqIndex` is reusable independently of DocUVerse
   — kernel-only experiments import it directly.
2. **Python ↔ Native.** Only bulk arrays cross (numpy `ndarray`, byte
   buffers). No per-vector function calls.
3. **C kernel ↔ orchestration.** Kernels are pure functions over SoA byte
   arrays + scalar parameters. OpenMP shard/merge sits outside the inner
   loop, in a thin driver in `topk.c`.

## 4. Module layout

```
docuverse/engines/retrieval/simdq/
├── __init__.py
├── simdq_engine.py          # SearchEngine subclass
├── simdq_index.py           # thin Python wrapper around the native module
├── projection.py            # identity / random-orthogonal W
├── quantization.py          # per-vector scale fitting (numpy)
└── _native/
    ├── CMakeLists.txt
    ├── include/
    │   ├── simdq_layout.h    # WORDS, LANES, SoA macros
    │   ├── simdq_kernels.h   # scan_hamming_<D>, scan_asym_b<b>_<D>
    │   ├── simdq_topk.h      # per-thread heap + merge
    │   └── simdq_pack.h      # b-bit pack/unpack helpers
    ├── src/
    │   ├── kernels_avx512.c  # AVX-512F (no VPOPCNTDQ required for asym)
    │   ├── kernels_avx2.c    # AVX2 fallback
    │   ├── topk.c
    │   └── pack.c
    ├── bindings/
    │   └── module.c          # CPython C-API
    ├── bench/
    │   ├── bench_hamming.c   # = current hb5, repackaged
    │   ├── bench_asym_b1.c
    │   ├── bench_asym_b2.c
    │   └── bench_asym_b4.c
    └── tests/
        └── test_kernels.c
```

Module responsibilities:

| Module | Owns | Doesn't own |
|---|---|---|
| `simdq_engine.py` | `SearchEngine` interface, encoder calls, evaluation glue | quantization math, kernel selection |
| `simdq_index.py` | index lifecycle (build/save/load), mmap of fp16 floats | SIMD specifics |
| `projection.py` | `W ∈ R^{d×D}` (identity, random orthogonal) | training (deferred) |
| `quantization.py` | scale/offset fitting per vector, b-bit packing for ingest | scan-time math |
| `_native/` | SIMD scan + top-K + rescore, C tests, benches | Python orchestration, file I/O policy |

Dependencies added: none beyond DocUVerse's existing requirements — numpy
+ a C compiler with OpenMP.

## 5. Data flow

### Ingest

```
texts → EmbeddingFunction.encode() → Z ∈ R^{N×D}  (fp32)
                                          │
                                          ▼
                                 projection.apply(Z, W)   # W ∈ R^{d×D}
                                          │
                                          ▼
                                     Y ∈ R^{N×d}
                                          │
                          ┌───────────────┴────────────────┐
                          │                                │
                          ▼                                ▼  (if store_floats)
                quantization.fit_pack(Y, b)         cast Y to fp16
                          │                                │
                          ▼                                ▼
                  codes ∈ uint8^{N×⌈d·b/8⌉}          floats ∈ fp16^{N×d}
                  scales ∈ fp16^{N}
                          │                                │
                          └────────────┬───────────────────┘
                                       ▼
                              SimdqIndex.write(path)
```

### Single-query search

```
query_text
    │
    ▼
EmbeddingFunction.encode() → q ∈ R^D  (fp32)
    │
    ▼
project: q' = W · q          ∈ R^d  (fp32, no quantization — asymmetric)
    │
    ▼
scan_asym_b<b>_<D>(q', codes, scales, N) → top-K' (idx, score)
    │                                  ▲
    │                                  │  threaded shard/merge,
    │                                  │  per-thread heap of size K'/T
    ▼
[if K' > K and floats present]
rescore_fp16(q, floats[idx for idx in top-K'], K) → top-K
    │
    ▼
return top-K
```

Two query modes:

| Mode | K' | Rescore | Output |
|---|---|---|---|
| **codes-only** | K' = K | none | top-K from scan |
| **two-stage** | K' = α·K (α=10 default) | fp16 dot product over K' candidates | top-K rescored |

### Batched search

`search_batch(queries, K)` reuses the `hb3/hb4` query-batching trick: NQ
queries per DB load, cutting memory traffic by NQ× — generalised from
XOR-popcount to float-FMA.

### Threading

OpenMP shard/merge inside a single call. Each thread scans `N/T` codes,
keeps a thread-local top-K' heap; final merge across threads. No
cross-query parallelism in v1.

### Asymmetric inner-loop math (b=2 example)

The dominant per-code work is a `q' · v_i` accumulation in float, with
the per-vector scale applied as a single multiply (or divide) on the
reduced sum. Pseudocode for one shard:

```
DB code:  v_i ∈ {-3, -1, +1, +3}^d  packed at 2 bits/dim
          stored as uint8 nibbles + per-vector fp16 scale s_i
Query:    q' ∈ R^d  (full float, projected only)

acc = _mm512_setzero_ps()
for w in 0..d step LANES:
    q_chunk = _mm512_loadu_ps(q' + w)             # 16 floats
    v_chunk = unpack_b2_to_int8(codes_w_i)         # 16 int8 levels
    v_float = _mm512_cvtepi32_ps(v_chunk)
    acc     = _mm512_fmadd_ps(q_chunk, v_float, acc)

raw = _mm512_reduce_add_ps(acc)
score = apply_scale(raw, s_i)   # exact form pinned in implementation plan
```

Unpacking cost: b=2 = one `vpshufb` over a 16-byte LUT; b=4 = two
`vpshufb` per byte. Both cheaper than the FMA work they feed.

## 6. Kernels & quantization

### Kernel matrix

| Family | b | D variants | SIMD paths |
|---|---|---|---|
| Symmetric Hamming | 1 | 4 | AVX-512F+VPOPCNTDQ, AVX2 nibble-LUT |
| Asymmetric scan | 1 | 4 | AVX-512F (no VPOPCNTDQ needed), AVX2 |
| Asymmetric scan | 2 | 4 | AVX-512F, AVX2 |
| Asymmetric scan | 4 | 4 | AVX-512F, AVX2 |

**Total: 16 kernel symbols × 2 SIMD paths = 32 compiled symbols.**

Symbol naming:

```
scan_hamming_d{384,768,1024,1536}_{avx512,avx2}
scan_asym_b{1,2,4}_d{384,768,1024,1536}_{avx512,avx2}
```

A small dispatch table in `bindings/module.c` maps `(family, b, D) →
fn_ptr` at index-load time. SIMD path selected at build time via
`__AVX512F__` macro (existing pattern from `hamming_kernels.h`).

### Quantization

Each DB vector is mapped to integer levels in
`V_b = {2c − 2^b + 1 | c = 0,…,2^b−1}` (b=1 → ±1, b=2 → ±1, ±3, b=4 → 16
levels). A per-vector scale is stored alongside the code so the asymmetric
inner product `q' · v_i` can be rescaled to recover an unbiased estimate
of `Wq · y_i`.

The exact form of the per-vector scale and the corresponding scan-time
recovery factor follow the ASH paper. Two derivations need to land
together at implementation time: the `fit_pack` function in
`quantization.py`, and the divisor (or multiplier) inside the C scan
kernel — they must be consistent. **Pinning the exact formula is left to
the implementation plan**; the spec commits only to "per-vector fp16
scale, ASH-style".

Two scale modes selectable at build time:

1. **`per_vector` (default)** — one fp16 per code (2 bytes/code overhead,
   <0.5% of a 384-byte b=4 d=768 code).
2. **`global`** — one fp32 in the metadata header. Smaller index, slightly
   worse recall on heavy-tailed embedding norms.

`per_dim` is not in v1.

### Projection

```python
def identity(D: int) -> np.ndarray:
    return np.eye(D, dtype=np.float32)

def random_orthogonal(D: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(d, D)).astype(np.float32)
    Q, _ = np.linalg.qr(G.T)        # → Q ∈ R^{D×d}, columns orthonormal
    return Q.T                       # → W ∈ R^{d×D}
```

`d ∈ {D, D/2}` only in v1. `d = D/2` enables the ASH-style "halve dims,
double bits" comparison without learning anything.

## 7. Index format

On disk — a directory, not a single file:

```
<index_path>/
├── meta.json
├── W.npy              # projection matrix, d×D fp32
├── codes.bin          # SoA-packed codes, raw bytes
├── scales.bin         # per-vector fp16 scales (omitted in global mode)
└── floats.bin         # optional: fp16 vectors at d-dim, mmap'd at search
```

**`meta.json` schema:**

```json
{
  "format_version": 1,
  "n_vectors": 5000000,
  "D_orig": 768,
  "d": 768,
  "b": 2,
  "projection": "random_orthogonal",
  "projection_seed": 42,
  "quantization": "per_vector",
  "code_layout": "soa",
  "code_bytes_per_vector": 192,
  "has_floats": true,
  "float_dtype": "float16",
  "encoder_id": "ibm-granite/granite-embedding-311m-multilingual-r2",
  "build_timestamp": "2026-06-15T14:30:00Z"
}
```

Atomic writes: build to `<index_path>.tmp/`, fsync, rename. Standard
pattern, matches LanceDB.

`format_version` is a single integer. v1 is forward-only — no migration
path. If the format changes, rebuild.

mmap'd `floats.bin` is a flat file the OS pages directly. Single-file
containers (HDF5, Lance) add a layer between mmap and the bytes that
doesn't help us.

## 8. DocUVerse integration

### Config dataclass

Added to `docuverse/engines/search_engine_config_params.py`:

```python
@dataclass
class SimdqConfig:
    index_path: str | None = field(default=None,
                                   metadata={"help": "required at runtime"})
    db_engine: str = "simdq"
    b: int = field(default=2, metadata={"help": "bits per dim: 1, 2, or 4"})
    d: int | None = field(default=None, metadata={"help": "reduced dim; None = D"})
    projection: str = field(default="identity",
                            metadata={"help": "identity|random_orthogonal"})
    projection_seed: int = 42
    quantization: str = field(default="per_vector")
    store_floats: bool = field(default=True,
                               metadata={"help": "enable fp16 rescore tier"})
    rescore_alpha: int = field(default=10,
                               metadata={"help": "K' = alpha * K when rescoring"})
    num_threads: int = field(default=0, metadata={"help": "0 = OMP default"})
```

### `SearchEngine` API

- `ingest(corpus)` — pulls embeddings via `EmbeddingFunction`, calls
  `SimdqIndex.build(...)`, writes to `index_path`.
- `update(corpus)` — **not supported in v1**. Codes are SoA-contiguous;
  appending requires rewrite. CLI help documents as "rebuild."
- `search(query, top_k)` — encode → project → `index.search(K, K_prime)`
  → wrap as `SearchResult`.
- `info()` — returns `meta.json` contents for debugging.

### Factory dispatch

Add `"simdq"` to `create_retrieval_engine` in
`docuverse/engines/retrieval/__init__.py` (same place `lancedb-bm25`,
`lancedb-sparse` etc. live, per commit `f608b0c`).

### YAML config

```yaml
search_engine:
  db_engine: simdq
  index_path: /scratch/beir-fiqa-simdq-b2
  b: 2
  store_floats: true
embedding_function:
  model_name: ibm-granite/granite-embedding-278m-multilingual-r2
```

Run via existing CLI:

```bash
python -m docuverse.utils.ingest_and_test --config beir_fiqa_simdq.yaml --actions "ire"
```

### Standalone use

For kernel/recipe ablations outside the BEIR harness:

```python
from docuverse.engines.retrieval.simdq import SimdqIndex

index = SimdqIndex.build(
    vectors=Y,                   # numpy (N, D) fp32
    b=2, d=384,                  # halve dims, double bits
    projection="random_orthogonal", projection_seed=42,
    store_floats=True,
)
index.save("/scratch/idx-b2-d384")

index = SimdqIndex.load("/scratch/idx-b2-d384")
indices, scores = index.search(query_fp32, K=100, K_prime=1000)  # two-stage
indices, scores = index.search(query_fp32, K=100, K_prime=100)   # codes-only
```

## 9. Build integration

`pyproject.toml` gains a `setup.py`-driven C extension (or
`scikit-build-core` if preferred). The extension is built during
`pip install -e .`.

Compile flags: `-march=native -O3 -fopenmp`.
SIMD path detected at compile time via preprocessor (`__AVX512F__`,
`__AVX512VPOPCNTDQ__`, `__AVX2__`).

Standalone CMake build for `_native/` is preserved — kernel-only
benchmarking and CTest workflow continue to work.

## 10. Testing

### Layer 1 — kernel correctness (C, every commit)

`docuverse/engines/retrieval/simdq/_native/tests/test_kernels.c`. Built
twice via CMake:

- `kernels_native` — `-march=native` → AVX-512F path on dev box.
- `kernels_avx2` — `-mavx2 -mpopcnt` forced → AVX2 fallback.

Each `(family, b, D, simd_path)` gets:

1. **Randomized input.** N codes + 1 query, scalar reference loop, assert
   top-K indices and scores match within `1e-4` relative tolerance.
2. **Planted-best cases.** Insert known minimum at index 0, at index N-1,
   in the `n % LANES` tail, and across an OMP shard boundary. Assert
   planted index appears at rank 1.
3. **Top-K boundary.** K=1, K=10, K=K_prime, K=N.
4. **Tie-breaking.** Kernels may return any minimizer when several codes
   share the minimum (existing `hb5` convention). Tests use planted-unique
   cases when asserting a specific index.

Per-kernel runtime budget: < 2s. Total CI time: ~45s.

### Layer 2 — Python integration (pytest)

`tests/test_simdq_engine.py`, `tests/test_simdq_index.py`:

- **Round-trip.** Build N=10k D=768 b=2 index, save, load, assert metadata
  + first-row code bytes match.
- **Search agreement vs sklearn.** N=10k random Gaussians + 100 random
  queries: top-10 from `SimdqIndex` (codes-only b=4 + rescore α=10) must
  overlap ≥ 95% with `sklearn.NearestNeighbors` cosine top-10.
- **Engine dispatch.** YAML with `db_engine: simdq` resolves through
  `create_retrieval_engine` → `SimdqEngine` instance.
- **Float rescore disabled.** `store_floats=False` ingest; assert
  `floats.bin` absent on disk; codes-only search still works.
- **Mode parity.** `K_prime=K` and `K_prime=10K` on the same index produce
  *different* top-K — confirms rescore is doing something.

No mocks where avoidable — these tests use real numpy data, no encoder
calls.

### Layer 3 — recipe benchmarks

**(a) Kernel microbench.** `_native/bench/bench_*.c`. Same shape as
current `hb*` binaries. CLI: `<prog> [n] [reps]`. Reports cmp/s, GB/s.
Sweep over 500k, 1M, 50M. Goal: validate kernel throughput vs design
assumption (asymmetric b=2 should be memory-bound at 30+ GB/s).

**(b) BEIR recipe sweep.** `scripts/bench_simdq_beir.py`, runs through
DocUVerse's normal CLI. For each `(b, d, projection, rescore_α)`,
captures NDCG@10, recall@100, query latency p50/p99, index size, peak
RAM. Outputs CSV.

**v1 recipe matrix:**

| Recipe | b | d | projection | rescore_α | What it answers |
|---|---|---|---|---|---|
| **R0** symmetric Hamming + rescore | 1 | D | identity | 10 | Reproduces BPR / sentence-transformers |
| **R1** asym 1-bit, no rescore | 1 | D | identity | 1 | Pure code-scan baseline |
| **R2** asym 1-bit + rescore | 1 | D | identity | 10 | Asymmetric vs symmetric at 1-bit |
| **R3** asym b=2, no rescore | 2 | D | identity | 1 | **Does b=2 alone match R0/R2?** |
| **R4** asym b=2, halved dims | 2 | D/2 | random_orthogonal | 1 | Half dims, double bits |
| **R5** asym b=4, halved dims | 4 | D/2 | random_orthogonal | 1 | Same memory as R3 |
| **R6** float baseline | — | D | — | — | Recall ceiling (existing engines) |

**Headline question:** R0 vs R3 at the same byte budget. If R3 wins,
asymmetric multi-bit replaces the binary+rescore pattern; if R0 wins,
1-bit + float rescore stays the recommendation.

R6 (float baseline) is computed via existing dense engines — no new code.

## 11. Success criteria

After v1 ships, a single command produces a CSV with NDCG@10 / latency /
index-size for R0–R6 on a chosen BEIR dataset. From that table the team
can decide whether ASH-style learned projection (v2) is worth the
engineering, or whether random orthogonal already reaches the Pareto
frontier.

## 12. Open questions for v2

- Does learned `W` (full ASH training loop) add measurable recall over
  random orthogonal at the same `(b, d)`?
- Is an outer index (IVF coarse cluster) worth adding for >10M corpora,
  or does brute-force at 30+ GB/s win all the way up to typical RAM
  limits?
- AVX-512 VNNI: cross-lane FMA over int8 codes is faster than the
  cvtepi-then-FMA path used here. Worth a kernel variant on supported
  CPUs.
- Per-dim quantization scale: revisit if per-vector recall plateau is
  hit.
