# Accuracy / Speed / Memory Comparison

Numbers come from different benchmarks (NQ, TriviaQA, BEIR, MTEB, ImageNet,
NUS-WIDE) and are **not strictly apples-to-apples** — read the row "domain" before
trusting cross-row comparisons. Float32 dense baseline = 1× memory, 1× speed.

## A0. ASH — IBM internal paper (Tepper × Willke, 2026)

Headline iso-footprint comparison from the ASH paper. Datasets: ada002, gecko,
openai-1536, openai-3072, nv-qa-v4, mpnet, cohere (100k–1M scale, 768–3072 dim).
Metric: terminal 10-recall@R.

| Method @ 32× compression | Memory factor | Terminal 10-recall@R | Speed at high-recall point |
|---|---|---|---|
| **ASH `b=2, d=D/2`** | 32× | **baseline (paper SOTA)** | **baseline** |
| RaBitQ (= ASH `b=1, d=D, W=R`) | 32× | **−2.3 to −7.1 points** vs ASH | **>12× slower** at RaBitQ's best point |
| PQ FastScan | 32× | **−6.3 to −22.9 points** vs ASH | slower |

| Method @ 16× compression | Memory factor | Terminal 10-recall@R |
|---|---|---|
| **ASH `b=4, d=D/2`** | 16× | baseline |
| RaBitQ | 16× | −2.2 to −6.8 points; >4.6× slower |
| PQ FastScan | 16× | −4.3 to −14.4 points |

Other relative results from the paper:
- **ASH `b=1` competitive with LeanVec `b=4`** (LeanVec uses 4× more memory).
- Beats LOPQ, EDEN, TurboQuant at iso-footprint across the Pareto frontier.
- ITQ is a special case of ASH (`b=1, d=D`); RaBitQ is a special case
  (`b=1, d=D, W=R`).
- SIMD: ASH `b=1` masked-load = 8 cycles latency / 0.5 throughput vs PQ gather
  = 30 / 9.75 (paper Table 3).

**Confidence:** medium — these are self-reported pre-print numbers, but the
experimental scope is broad (7+ datasets, 6 baselines).

## A. Dense retrieval (text) — most relevant to DocUVerse

| Method | Domain / Benchmark | Memory factor | Search speedup (CPU) | Accuracy (vs. float32 baseline) | Source |
|---|---|---|---|---|---|
| **Float32 dense (DPR / TAS-B / BGE / granite-embedding)** | ODQA + BEIR + MTEB | 1× | 1× | baseline | — |
| **Binary quantization + rescoring** (Sentence-Transformers / Cohere / Mixedbread) | MTEB | **32×** smaller | "large" — popcount Hamming + smaller candidate set; specific factor model-dependent | **~96% NDCG@10** best case (mxbai-embed-large-v1: 54.39 → 52.46), 75–94% on weaker models | [sentence-transformers](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/embedding-quantization/README.md), [Cohere](https://docs.cohere.com/docs/int8-and-binary-embeddings) |
| **int8 quantization** (Cohere uint8 / SBQ) | MTEB | **4×** smaller | moderate (SIMD int8) | ≥99% typically | Cohere docs |
| **BPR — binary DPR** (Yamada et al. ACL'21) | Natural Questions + TriviaQA | **65 GB → 2 GB ≈ 32×** | not separately reported | **No end-to-end QA accuracy loss**; minor Top-1 retrieval gap (52.5 → 49.0 NQ) | [arxiv 2106.00882](https://arxiv.org/abs/2106.00882) |
| **Naive learning-to-hash on TAS-B** (zero-shot) | BEIR (out-of-domain) | 32× | — | **−14% nDCG@10** | [arxiv 2205.11498](https://arxiv.org/abs/2205.11498) |
| **Domain-adapted BPR** (GPL-style) | BEIR | 32× | **14×** | **+11.5% nDCG@10** vs. naive zero-shot baseline | [arxiv 2205.11498](https://arxiv.org/abs/2205.11498) |
| **Domain-adapted JPQ** | BEIR | 32× | 2× | +8.2% nDCG@10 | [arxiv 2205.11498](https://arxiv.org/abs/2205.11498) |

## B. Classical learning-to-hash (image retrieval, mostly)

| Method | Domain | Code length | Accuracy notes | Source |
|---|---|---|---|---|
| **LSH (random hyperplane)** | Generic, unsupervised | Needs more bits than data-dependent methods at equal accuracy | Provable sublinear NN bound | [Gionis-Indyk-Motwani 1999](https://www.cs.princeton.edu/courses/archive/spring13/cos598C/Gionis.pdf) |
| **ITQ** (Gong & Lazebnik 2011) | CIFAR / GIST / ImageNet (unsupervised) | 32 / 64 / 128 / 256 bits | Outperforms LSH and Spectral Hashing as unsupervised baseline | [ITQ](https://slazebni.cs.illinois.edu/publications/ITQ.pdf) |
| **Spherical Hashing** | Same | Same | Stronger geometric locality than ITQ | [Spherical](http://sunglab.kaist.ac.kr/papers/SphericalHashing_TPAMI15.pdf) |
| **DPSH** (deep, pairwise) | CIFAR / NUS-WIDE | 12–48 bits | Joint feature + code learning beats two-stage methods | [arxiv 1511.03855](https://arxiv.org/abs/1511.03855) |
| **HashNet** (continuation `tanh`) | ImageNet, NUS-WIDE | 16–64 bits | Beats DPSH and earlier deep methods | [arxiv 1702.00758](https://arxiv.org/abs/1702.00758) |
| **Greedy Hash** (straight-through `sign`) | CIFAR / NUS-WIDE / ImageNet | 12–64 bits | Simple, competitive with HashNet | [NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/13f3cf8c531952d72e5847c4183e6910-Abstract.html) |
| **CSQ** (central similarity, Hadamard centers) | CIFAR / NUS-WIDE / ImageNet / MS COCO + video | 16 / 32 / 64 bits | **+3–20% mAP** over prior deep SOTA | [arxiv 1908.00347](https://arxiv.org/abs/1908.00347) |

## C. Memory math (dimension d, n = 1M vectors)

| Encoding | Bits/dim | Bytes per d=768 vector | Index size at n=1M, d=768 |
|---|---|---|---|
| float32 | 32 | 3072 | 2.93 GB |
| float16 | 16 | 1536 | 1.46 GB |
| int8 | 8 | 768 | 750 MB |
| **binary (1-bit)** | **1** | **96** | **92 MB** |

## D. Search-cost notes (refuted "fast" claims)

- The widely-quoted *"Hamming distance computes in ~2 CPU cycles"* is **misleading
  and was refuted** by 2/3 verifiers. Per-comparison cost is `O(d / 64)` `popcnt`
  instructions. For d=768 this is ~12 popcnts plus a horizontal sum — fast, but
  not 2 cycles, and SIMD/AVX availability matters.
- Real-world speedup over float32 cosine on the same index size comes from two
  effects: (a) cheaper per-comparison cost, (b) shorter codes fit in cache. The
  14× CPU speedup figure (Thakur et al.) is the strongest empirical anchor for
  end-to-end retrieval, but it is on BEIR-scale corpora (~5M docs); extrapolation
  to 10⁸–10⁹ docs is an open question.

## E. Confidence

- **High confidence**: 32× memory reduction (bit arithmetic); BPR's
  65 GB → 2 GB number; the ~96% best-case MTEB preservation; ITQ's beating LSH
  and Spectral Hashing as an unsupervised baseline.
- **Medium confidence**: cross-row comparisons (different benchmarks); CSQ's
  3–20% mAP gain (self-reported).
- **Low confidence / extrapolation**: image-retrieval methods' relative ranking
  on dense text retrieval; Matryoshka binary specifics.
