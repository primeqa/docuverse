# Recommendations for DocUVerse

Engineering inferences from the verified literature. Confidence: **medium** — the
underlying papers are solid, but mapping them onto specific DocUVerse engines is
not directly evidenced.

## Aggressive option: ASH (IBM internal paper)

**Especially relevant for DocUVerse:** ASH (Tepper × Willke, 2026 internal
pre-print, `papers/16-ASH-Tepper-Willke-IBM-2026.md`) is co-authored by Elastic
(the search engine DocUVerse already wraps) and IBM WatsonX (a different IBM
team than the one that trains / fine-tunes the granite-embedding model
DocUVerse uses, but still inside IBM). The combination of Elastic engineering
authorship and IBM-internal availability makes this a **strategically
well-positioned method** for our stack.

**When to use:** you want the strongest accuracy/speed tradeoff at iso-footprint
and you're willing to (a) train a small projection matrix offline, and (b) run
on AVX-512-capable CPUs.

**Recipe:**
1. Sample `~10·D` representative vectors from the corpus.
2. Initialize `W = R · P` where `P` is the top-`d` PCA eigenvectors and `R` is
   a random rotation in `SO(d)`.
3. Pick `(d, b)` from the design space — paper recommends:
   - `b=2, d=D/2` for **32× compression** (best for very large indexes).
   - `b=4, d=D/2` for **16× compression** (best for balanced accuracy/memory).
4. Alternate 20–30 iterations: quantize codes `v_i = quant_b(W x_i)`, then
   update `R` via orthogonal Procrustes.
5. At ingestion: store quantized codes + scale + offset (small FP16 header).
6. At query time: `Wq` (project, **don't quantize**) and dot product against
   the codes via masked-load FMA on AVX-512.

**Expected gains over plain binary+rescoring** at the same memory footprint:
+2–7 points terminal recall over RaBitQ at 32×; +6–23 over PQ FastScan.

**Caveats:** internal pre-print, no public reference implementation yet, AVX-512
required for headline speeds. If those are blockers, fall back to the default
binary-quantization+rescoring recipe below.

## Default recipe: binary quantization with continuous rescoring

**When to use:** in-domain corpora, strong off-the-shelf encoder available
(granite-embedding, BGE, mxbai-embed-large-v1, e5-base/large).

**Two-stage retrieval:**
1. **Candidate generation** — Hamming search over binary codes returns top-K' (K' ≫ K, e.g. K'=10·K).
2. **Rescoring** — recompute the float / fp16 dot product over candidates, return top-K.

**Memory:** 32× reduction on the primary index. Keep float vectors on cheaper
storage tier or recompute via the encoder.

**Expected accuracy:** 75–96% of float32 NDCG@10, depending on encoder strength.
Strong encoders (mxbai-embed-large-v1, BGE-large) sit at the top of that range.

## Engine-specific recipes

### Milvus (preferred)

```python
# Field schema
FieldSchema(name="emb_binary", dtype=DataType.BINARY_VECTOR, dim=768)
FieldSchema(name="emb_float",  dtype=DataType.FLOAT_VECTOR,  dim=768)

# Index on binary
collection.create_index(
    "emb_binary",
    {"index_type": "BIN_IVF_FLAT", "metric_type": "HAMMING", "params": {"nlist": 1024}},
)

# Two-stage search
candidates = collection.search(
    data=[query_bin], anns_field="emb_binary",
    param={"metric_type": "HAMMING", "params": {"nprobe": 16}},
    limit=K * 10,  # over-retrieve
)
final = rescore_with_float(candidates, query_float, top_k=K)
```

`BIN_IVF_FLAT` is the standard binary index in Milvus; `HAMMING` and `JACCARD`
metrics are first-class. Milvus 2.4+ also adds scalar quantization, useful as a
middle ground between float32 and 1-bit.

### LanceDB

Store binary as a `fixed_size_list<uint8>` column alongside the existing
fp16 / fp32 column. The columnar SQL layer makes two-stage rescoring natural and
fits cleanly into the existing `LanceDBHybridEngine` composition (commits c659c97,
d4badbd in this repo).

```python
# At ingestion
bits = np.packbits((emb > 0).astype(np.uint8), axis=-1)  # (N, d/8) uint8

# Schema: add 'emb_bin' (fixed_size_list<uint8, d/8>) next to 'vector'
# At query time:
#   1. ANN search on emb_bin with custom Hamming distance UDF
#   2. JOIN back to fp32 'vector' column for top-K' candidates
#   3. Rescore + return top-K
```

LanceDB's first-class FTS + vector hybrid (already wired up here for BM25 / sparse
hybrids) is a natural place to plug in a binary-then-rescore pipeline as a third
hybrid mode.

### Elasticsearch

Less mature for native 1-bit. **Recommended fallback for ES users:**
- Use `dense_vector` with `element_type=byte` for **int8** quantization (4×
  reduction, ≥99% accuracy retained).
- For 1-bit, defer to Milvus or LanceDB; ES native bit support has lagged.
- ES 8.16+ "Better Binary Quantization" via Lucene 9.12 is the path forward but
  requires testing — see the Elastic search-labs blog for the current state.

## Domain-adaptation recipe (for out-of-domain / heterogeneous corpora)

**When to use:** serving BEIR-style heterogeneous corpora, or a domain the encoder
wasn't trained on (medical, legal, code, niche enterprise data).

**Why:** Thakur et al. (ReNeuIR'23) show naive learning-to-hash drops up to **−14%
nDCG@10** zero-shot on BEIR — but a domain-adapted BPR head recovers to **+11.5%
nDCG@10** with **14× CPU speedup** at 32× memory.

**Recipe:**
1. Generate synthetic query–passage pairs on the target corpus with a generator
   (T5-style). This is the "GPL" (Generative Pseudo-Labeling) pipeline.
2. Fine-tune the dense encoder + a BPR-style binary head with multi-task loss:
   - Binary candidate generation loss (Hamming over `sign(W·h)`)
   - Continuous reranking loss (dot product over `h`)
3. Quantize and serve as in the default recipe above.

Reference: BPR (`studio-ousia/bpr`) + GPL (`UKPLab/gpl`).

## When to reach for the deep-hashing methods

For DocUVerse's text-retrieval focus, the classical deep-hashing methods (DPSH,
HashNet, Greedy Hash, CSQ) are mostly **not the right tool**:

- They were validated on image retrieval with class labels.
- The "binary quantization + rescoring" recipe with a strong pretrained encoder
  beats them on text without any extra training.
- Their tricks (continuation, straight-through, central similarity) are useful
  if you ever want to **train** a custom binary encoder end-to-end — most likely
  scenario: tightly controlled domain, very short codes (≤64 bits), and labeled
  similarity data.

If that scenario arrives, **Greedy Hash** is the easiest baseline (drop-in
straight-through `sign()`) and **CSQ** is the strongest reported.

## What to test first inside DocUVerse

Concrete experiments worth running, in order:

1. **Baseline + binary**: take the existing granite-embedding pipeline, add
   `np.packbits(emb > 0)` at ingestion, set up Milvus/LanceDB binary index +
   float rescore. Measure NDCG@10 / Recall@100 vs. float32 on your held-out eval.
2. **Code-length sweep**: 768 → 512 → 256 → 128 bits via Matryoshka-style
   truncation before packbits. Find the knee.
3. **PQ vs. binary at equal memory**: 8-byte PQ codes ≈ 64-bit binary at the same
   memory. Head-to-head on the same eval.
4. **ASH (IBM internal paper)**: implement the alternating-minimization training
   from `papers/16-ASH-Tepper-Willke-IBM-2026.md` over granite-embedding outputs.
   Compare `b=2, d=D/2` (32× compression) and `b=4, d=D/2` (16×) against the
   binary baseline from step 1. Reach out to the authors at IBM WatsonX /
   Elastic for a reference implementation if available.
5. **Domain adaptation** (only if 1–4 fall short): GPL-generated pairs + BPR head.

## Caveats baked in

- The "32× memory" number is exact bit-arithmetic.
- The "14× CPU speedup" is from BEIR-scale (~5M docs); larger corpora may need
  IVF coarse quantizers to preserve sublinear scaling.
- The "~96% accuracy preserved" is best-case (mxbai-embed-large-v1); your encoder
  may be lower (e5-base ~74%, all-MiniLM-L6-v2 ~94%).
- The widely-quoted "2-cycle Hamming distance" claim was refuted; real cost is
  `O(d/64)` popcounts.
- Cohere int8/binary launched March 2024; vector-DB native binary support is
  evolving — re-check Milvus / LanceDB / Elasticsearch release notes before
  committing.
