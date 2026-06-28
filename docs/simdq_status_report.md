# simdq / DocUVerse — State of Affairs

_Report date: 2026-06-27. Scope: simdq engine bring-up on NQ + scifact with granite
embedding models, plus the bugs found and fixed along the way._

## TL;DR

- **One correctness bug mattered above all others:** the multi-threaded asymmetric
  scan kernels (`b1/b2/b4`) sharded the corpus on **non-byte-aligned** boundaries,
  so any run using >2 threads returned wrong top-K candidates. On granite-**311m**
  (768-dim) this collapsed NDCG@10 from **0.607 → 0.247**. Fixed; simdq now matches
  exact fp32 to ±0.001.
- **Quality is lossless** for asymmetric `b=2` + fp32 rescore on both encoders tested
  (granite-97m and granite-311m) once the kernel bug was fixed.
- **Speed: simdq is ~14–15× faster than Milvus FLAT at both dims** (search-only,
  same vectors/machine: 384d 0.42 vs 6.3 ms/q; 768d 0.89 vs 12.2 ms/q). The
  original "768-dim ~5× slower (49 ms)" was wrong on two counts, both fixed:
  (1) a thread-oversubscription bug (per-query identity `W@q` spawned an OpenBLAS
  pool fighting the scan threads) → 49→3.9 ms; (2) a scalar 2-bit unpack in the
  AVX-512 kernel → replaced with `vpmultishiftqb`+`vpshufb` → 3.9→0.9 ms. See
  `docs/simdq_768_investigation.md`.
- A batch of smaller correctness/usability/perf fixes landed (config typos, lazy GPU
  load, multi-core tokenization, matryoshka guard, deprecation warning, redundant
  re-tokenization).

---

## 1. Headline bug — parallel scan kernel byte-misalignment

**Symptom:** simdq on granite-311m gave catastrophic retrieval quality vs Milvus fp32:

| metric | simdq (buggy) | Milvus fp32 |
|---|---|---|
| NDCG@10 | 0.247 | 0.607 |
| Match@100 | 0.393 | 0.983 |

**Diagnosis path (all empirically verified, not assumed):**
1. Ruled out anisotropy — candidate recall@256 was 99.8% on a fresh index.
2. Ruled out query/document prompt mismatch — granite prompts are empty strings.
3. Ruled out the stored index — exact fp16 rescore over stored floats gave Match@100
   = 0.980; stored doc embeddings matched fresh encodes at cos = 1.0000.
4. Ruled out the batched query-embedding cache — every query mapped to its own vector.
5. **Isolated to thread count:** `idx.search` on the stored index gave
   `t=1→0.985, t=2→0.985, t=4→0.580, t=0(all)→0.415`. Quality degraded monotonically
   with threads — the signature of a sharding bug.

**Root cause:** `scan_asym_b{1,2,4}_topk_parallel` split work as
`i0 = t * ceil(N/T)`. The sub-byte-packed codes (b2 = 4 codes/byte) are read at byte
offset `(ii >> 2)`, which is only correct when a shard starts on a byte boundary. For
N=276,007: T=2 → chunk 138,004 (÷4, OK) but T=4 → chunk 69,002 (**not ÷4**) →
misaligned reads → wrong candidate indices. The engine runs `simdq_num_threads: 0`
(all cores), so it always hit the broken path.

**Fix (commit `30a1b17`):** round each shard's `chunk` up to a multiple of 64 so every
`i0` is byte-aligned, in `simdq_kernels_asym_b1/b2/b4.h`. Hamming was already safe (it
indexes full uint64 elements). Rebuilt the native extension.

**Verification:**
- All thread counts now return Match@100 = 0.985 (was 0.985/0.985/0.580/0.415).
- Added regression test `tests/test_simdq_index.py::test_parallel_scan_matches_single_thread`
  (b=1/2/4 at an unaligned N) — would have caught this; existing suite still green (33 tests).
- End-to-end re-run: simdq 311m now NDCG@10 0.607, Match@100 0.984 — matches Milvus.

> Note: the granite-**97m** (384-dim) results were "lucky" — that dim/N happened to
> avoid the worst misalignment. They're unaffected, but anything sensitive should be
> re-validated against the fixed kernel.

---

## 2. Quality — current state (NQ dev, asymmetric b=2 + fp32 rescore α=10)

| encoder | NDCG@10 (fp32 / simdq) | NDCG@100 (fp32 / simdq) | Match@100 (fp32 / simdq) |
|---|---|---|---|
| granite-97m (384d) | 0.576 / 0.576 | 0.610 / 0.609 | 0.974 / 0.974 |
| granite-311m (768d) | 0.607 / 0.607 | 0.634 / 0.634 | 0.983 / 0.984 |

**Lossless on both.** scifact (granite-278m, 768d) earlier showed the same parity.

---

## 3. Speed — current state

Re-benchmarked Milvus FLAT and simdq with **one identical harness**: search-only
(queries pre-encoded), same 276k vectors, same machine (Ryzen 9 9950X3D), Milvus
standalone server on :19530, 16-way concurrent. ms/query (lower better):

| encoder | Milvus FLAT fp32 | simdq b=2 | verdict |
|---|---|---|---|
| granite-97m (384d) | 6.3 ms | **0.42 ms** (2373 q/s) | simdq ~15× faster |
| granite-311m (768d) | 12.2 ms | **0.89 ms** (1128 q/s) | simdq ~14× faster |

(The Milvus numbers match the earlier ~7.6 / 9.4 ms within measurement noise.)

**Hamming is now the fastest lossless option** (768d, same harness): after storing
codes SoA to kill the per-query AoS→SoA transpose, flat Hamming + fp32 rescore runs
at **0.39 ms/q** (was 3.36) — faster than asym b=2 (0.81 ms), lossless (NDCG@10
0.607, Match@100 0.984), at half the code size (96 vs 192 B/doc). See
`docs/simdq_768_investigation.md` §8. The IVF outer index (§7) no longer helps at
L3-resident corpus sizes and is kept only for corpora ≫ L3.

Two fixes took 768d from the original 49 ms to 0.9 ms:

1. **Projection/BLAS oversubscription (49 → 3.9 ms).** Under `parallel_process`
   (one query per thread) the per-query identity projection `W@q` (768×768)
   crossed OpenBLAS's auto-parallel threshold, so each query-thread spawned its
   own ~16-thread BLAS pool on top of the scan's OpenMP threads → collapse. At
   384d the matrix stays below the threshold (why 384d never showed it). Fix:
   `SimdqIndex._project_query` skips the matmul for identity projection (bit-exact)
   and pins BLAS for the non-identity path.
2. **Vectorized AVX-512 unpack — Lever 1 (3.9 → 0.9 ms).** `b2`/`b4` unpacked
   2-bit/4-bit codes with a scalar loop into a stack array then reloaded (a
   store-forwarding stall). Replaced with an in-register `vpmultishiftqb` (VBMI)
   + `vpshufb` LUT unpack: ~4× faster single-thread (768d 52 → 13 ms), bit-identical.

Full diagnosis + measurements in `docs/simdq_768_investigation.md`. The standalone
scan-time sweep (`scripts/bench_simdq_scan.py`) confirms scan-only timings; note it
measures one query at a time, so it shows the serial number, not engine throughput.

---

## 4. Other fixes landed this session

| area | problem | fix |
|---|---|---|
| Milvus AUTOINDEX | `params: null` → pymilvus "Search params must be a dict" | `config/milvus_default_config.yaml` → `params: {}`; `milvus_dense.py` coerces None params and keeps search metric == index metric |
| Engine factory | unknown `db_engine` returned `None`, crashed 14 min later | `retrievers.py` raises a clear `ValueError` at startup (caught a `simqd`/`simdq` typo) |
| Tokenizer speed | `num_preprocessor_threads: 10` ran at ~150% CPU (~1 core) | Lazy GPU model load + CUDA-aware fork multiprocessing → ~1000% CPU, **14 min → ~40 s** |
| Query encode | per-query GPU encode under parallel_process | `precompute_query_embeddings` — one batched forward pass, cached per query |
| matryoshka_dim | `768` on a 384-dim model → confusing dim crash deep in ingest | `dense_embedding_function.py` validates `matryoshka_dim ≤ native dim` at load, clear error |
| Deprecation | `get_sentence_embedding_dimension` FutureWarning | version-compat helper preferring `get_embedding_dimension` |
| Redundant work | `verbose: True` re-tokenized every tile for the `tlen` stat | single-tile tiles reuse the length computed during tiling; ~halves tokenize for single-tile-dominant corpora |

---

## 5. Slides

`docs/slides_bit_hashing.tex` (53 pages, builds clean) gained a real-corpus NQ slide
covering **both** encoders: quality parity table + scan speed. simdq is now faster
than Milvus FLAT at **both** dims (~2.3× at 384d, ~2.4× at 768d); the slide notes the
earlier 768d "regression" and its root cause (BLAS oversubscription), now fixed.

---

## 6. Open items / recommendations

1. ~~**768-dim scan kernel speed.**~~ **Resolved** — was BLAS thread
   oversubscription in the per-query projection, not the kernel. Fixed in
   `SimdqIndex._project_query` (identity-skip + BLAS pin); 49 → 3.9 ms/query. The
   VBMI/GFNI unpack and `d=D/2` projection ideas remain valid future kernel
   optimizations but are no longer needed for parity. See
   `docs/simdq_768_investigation.md`.
2. **Query-level parallelism reworked (single-GPU).** The old path encoded inside
   `parallel_process` (multiprocessing+fork), which broke after the CUDA context
   was created — so it only scaled with one GPU per worker. New
   `SimdqEngine.search_all` does it in two phases: one batched GPU encode, then a
   *thread* pool over the GIL-releasing CPU scans (each scan single-threaded).
   `SearchEngine.search` delegates to it. See `docs/simdq_768_investigation.md` §4.
3. **Re-validate 97m numbers** against the fixed kernel (they predate it).
4. **`bench_simdq_scan.py`** is still untracked — commit if it should be kept.
5. **hypothesis** not installed → `tests/test_simdq_fuzz.py` is skipped in CI here.
