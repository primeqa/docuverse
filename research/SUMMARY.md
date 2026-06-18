# Approach Summaries

Methods are grouped by lineage. Each entry: one-line definition, mechanism, and
the unique selling point. See `papers/` for full records and links.

## 1. Data-independent hashing

### Locality-Sensitive Hashing (LSH) — Indyk & Motwani 1998 / Gionis-Indyk-Motwani VLDB 1999
The foundational hashing approach. A family of hash functions where the collision
probability is higher for nearby points than for distant ones. Achieves sublinear
approximate-NN query time `O(d · n^(1/(1+ε)))`, improving the prior `O(d · n^(1/ε))`
bound. **Data-independent**: no training corpus needed; hash family is randomized.
For cosine/inner-product similarity, the canonical instantiation is *random
hyperplane LSH* (Charikar 2002): one bit per random Gaussian projection, sign of
dot product. Hamming distance ≈ angular distance in the limit.

**Strength:** zero training, mathematical guarantees.
**Weakness:** code length must grow large to match data-dependent methods at the
same accuracy.

## 2. Classical data-dependent (learning-to-hash, unsupervised)

### Iterative Quantization (ITQ) — Gong & Lazebnik CVPR 2011
The canonical unsupervised hashing baseline (~1880 citations). Procrustean approach:
project to a c-dim PCA subspace, then find an orthogonal rotation R that minimizes
the quantization error `||sgn(VR) − VR||²` between the rotated PCA scores and the
binary hypercube vertices. Alternates between (a) updating the binary code via
`sgn()` and (b) updating R via SVD of `B^T V`. Outperforms LSH and Spectral Hashing.

**Strength:** simple, no labels needed, much better than LSH at short code lengths.
**Weakness:** purely unsupervised; cannot exploit semantic similarity supervision.

### Spherical Hashing — Heo et al. CVPR 2012 / TPAMI 2015
Replaces hyperplanes with hyperspheres. Each hash bit indicates whether a point lies
inside or outside a learned hypersphere. Tighter, more localized partitioning than
hyperplane-based methods; better matches the Gaussian-like density of real
embeddings.

**Strength:** stronger geometric locality than ITQ at the same code length.
**Weakness:** more complex training; less standard than ITQ.

### Wang et al. — A Survey on Learning to Hash (TPAMI 2018, arxiv 2014)
The taxonomy reference. Splits hashing into **data-independent** (LSH variants) vs.
**data-dependent / learning-to-hash**, with L2H further subdivided into
*pairwise-similarity-preserving*, *multiwise-similarity-preserving*,
*implicit-similarity-preserving*, and *quantization-based* methods. Use this when
deciding which family to pull from.

## 3. Deep hashing (supervised end-to-end)

The discrete `sign()` output makes optimization NP-hard; each method below proposes
a different relaxation trick.

### DPSH — Li et al. IJCAI 2016
First method to perform **simultaneous feature and binary-code learning** under
**pairwise similarity labels** with deep networks. Loss: max-likelihood of pairwise
labels under a sigmoid of inner product between continuous outputs, plus a
quantization regularizer pushing outputs toward `±1`.

**Strength:** end-to-end deep + pairwise labels.
**Weakness:** quadratic in pair count; quantization gap.

### HashNet — Cao et al. ICCV 2017
Continuation method: forward pass uses `tanh(β·x)`, with β increased over training
until it converges to `sign()`. Solves the vanishing-gradient problem by smoothly
annealing the discrete constraint.

**Strength:** principled solution to the gradient issue; strong image-retrieval
results.
**Weakness:** β-schedule is a hyperparameter.

### Greedy Hash — Su et al. NeurIPS 2018
Uses `sign()` strictly in forward pass, but copies the gradient through unchanged
in the backward pass (an "identity straight-through" estimator). Eliminates the
quantization gap entirely. Empirically robust; widely adopted as a baseline trick.

**Strength:** trivial to implement; no continuation schedule.
**Weakness:** straight-through is a heuristic, not a principled relaxation.

### CSQ (Central Similarity Quantization) — Yuan et al. CVPR 2020
Introduces a global "hash center" metric. Each class is assigned a Hadamard-derived
center on the binary hypercube; samples are pulled toward their class center and
pushed away from others in Hamming space. Reports **3–20% mAP gains** over prior
deep hashing SOTA on CIFAR-10, NUS-WIDE, ImageNet, MS COCO and video-retrieval
benchmarks.

**Strength:** strong global structure; Hadamard codes maximize inter-class Hamming
distance.
**Weakness:** image-retrieval benchmarks; transfer to dense text retrieval not
established.

### Luo et al. — A Survey on Deep Hashing (arxiv 2003.03369, 2020)
Organizes deep hashing into:
- **Deep supervised**: pairwise / ranking-based / pointwise / quantization
- **Deep unsupervised**: similarity-reconstruction / pseudo-label /
  prediction-free self-supervised

Use this to navigate the deep-hashing zoo when picking a baseline.

## 4. Modern binary quantization for retrieval

These are not novel hashing algorithms — they are **engineering recipes** that apply
1-bit quantization (`x > 0`) to outputs of a strong pretrained dense encoder, then
recover accuracy with a rescoring pass. The simplicity is the point.

### Cohere int8 & binary embeddings — March 2024
Production API. `embedding_types=[float, int8, uint8, binary, ubinary]` returned in
one call. Native vector-DB integration with Qdrant, Weaviate, MongoDB Atlas,
Pinecone. Memory savings: float32 → int8 = 4×; → binary = 32×.

### Sentence-Transformers embedding quantization — Reimers et al. 2024
Reference implementation (HuggingFace blog + sentence-transformers examples).
Two-stage retrieval: (1) Hamming search over binary codes returns a candidate set;
(2) rescore candidates with float embeddings (kept in RAM or recomputed). Reports
**up to 96.45%** of NDCG@10 preserved (`mxbai-embed-large-v1` MTEB: 54.39 → 52.46);
weaker encoders preserve 75–94%.

### Mixedbread Binary MRL — 2024
Combines **Matryoshka Representation Learning** (truncatable embedding dimensions)
with binary quantization. Lets you trade off code length × bits-per-dim
independently. Notable for training the encoder so binary codes at the truncated
length are still high-quality.

### BPR (Binary Passage Retriever) — Yamada, Asai, Hajishirzi ACL 2021
The strongest dense-retrieval-specific result. Multi-task training on top of DPR:
- Task A: candidate generation via binary code Hamming similarity
- Task B: rerank top-K candidates using continuous-vector dot product

Index footprint: **65 GB → 2 GB** (~32×) on Natural Questions and TriviaQA, **with no
end-to-end QA accuracy loss**. Top-20 / Top-100 retrieval recall preserved; minor
Top-1 retrieval gap (52.5 → 49.0 NQ) which doesn't propagate to reader EM.

### Thakur, Reimers, Lin — ReNeuIR'23 (arxiv 2205.11498)
Domain-adapted dense retrieval with binary embeddings on BEIR. Findings:
- Naive learning-to-hash on TAS-B loses **up to 14% nDCG@10** zero-shot on BEIR.
- **Domain-adapted BPR** recovers and exceeds baseline: **+11.5% nDCG@10**, with
  **14× CPU speedup** at 32× memory.
- **Domain-adapted JPQ**: +8.2% nDCG@10, 2× CPU speedup, 32× memory.

The single most actionable result for choosing binary recipes when serving
heterogeneous / out-of-domain corpora.

## 4b. ASH — Asymmetric Scalar Hashing (IBM internal paper)

### ASH — Tepper (Elastic) & Willke (IBM WatsonX), 2026 internal pre-print
Stored locally at `research/ash-paper-internal-copy.pdf`. The most relevant
single result for DocUVerse: **Elastic-authored, IBM-co-authored**, evaluated on
modern embedding-model corpora (ada002, gecko, openai-1536/3072, nv-qa-v4,
mpnet, cohere) — i.e. the regime DocUVerse actually serves.

**Core idea.** Learn an orthonormal projection `W ∈ St(d, D)` that reduces
dimensionality `D → d`, then apply **multi-bit scalar quantization** (`b` bits
per reduced dimension, `b ∈ {1, 2, 4}`). **Asymmetric**: queries stay in
float (`Wq`), only DB vectors are quantized.

**Generalizes** prior methods as special cases:
- **ITQ** = ASH with `b=1, d=D` (learned rotation, no dim reduction).
- **RaBitQ** = ASH with `b=1, d=D, W=R` (random rotation).
- The new design space is `d < D` AND `b ≥ 1` simultaneously.

**Headline result.** **Reducing `d` and increasing `b` is better than `b=1` at
full `d`** at the same memory footprint:
- **32× compression** (`b=2, d=D/2`): **+2.3–7.1 points** terminal 10-recall@R
  over RaBitQ; **>12× faster** at RaBitQ's highest-recall point.
- **16× compression** (`b=4, d=D/2`): +2.2–6.8 points over RaBitQ; >4.6× faster.
- vs **PQ FastScan**: **+6.3–22.9 points** terminal recall at 32×;
  +4.3–14.4 at 16×.
- ASH at **`b=1`** matches LeanVec at **`b=4`** (4× more memory).

**Why it's faster than PQ.** ASH's inner loop (b=1) is **masked-load + FMA +
horizontal sum** on AVX-512: latency 8, throughput 0.5 cycles/instr. PQ's
gather is latency 30, throughput 9.75 — ~19× worse throughput. This is the
mechanical reason for the speed gap, not just a constant factor.

**Caveats.** Internal pre-print; AVX-512 needed for headline speeds; needs
`~10·D` training vectors (trivial). Numbers self-reported.

## 5. Engine-specific notes

### Elasticsearch — Better Binary Quantization (Lucene 9.12, 2024)
Native binary scalar quantization built into Lucene's HNSW. Positions itself as a
competitor to PQ for the BBQ ("Big Bag of Quantizers") workflow. ES integration
is most mature for `int8` (`element_type=byte`); 1-bit native support is still
catching up.

### Milvus — Binary Vector docs
First-class `BinaryVector` field type with `HAMMING` and `JACCARD` metrics, BIN_FLAT
and BIN_IVF_FLAT indexes. Cleanest engine for binary-only experiments today.

### LanceDB
Stores binary vectors as `fixed_size_list<uint8>` columns; rescoring is natural in
the columnar SQL layer (already used in `LanceDBHybridEngine`, commits c659c97 and
d4badbd in this repo).

## What was *not* found in the verified literature

- **Matryoshka binary embeddings** as a peer-reviewed paper (Mixedbread blog only).
- Head-to-head **PQ/OPQ vs. binary-quantization+rescore** at equal memory budgets
  on modern dense retrievers.
- **Optimal binary code length** for 384/768-d encoders on text retrieval.
- Concrete **Elasticsearch native 1-bit recipes** (mostly int8 in production).

These are listed as open questions in the deep-research output and are good
candidates for follow-up benchmarking inside DocUVerse.
