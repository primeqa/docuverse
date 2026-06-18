# Embeddings → Bit Hashes: Literature Survey

Research compiled 2026-06-15 for the DocUVerse project. Sources verified by a 3-vote
adversarial deep-research workflow (89 claims extracted, 25 verified, 22 confirmed,
3 refuted). 20 primary/secondary sources covering five angles:

1. Foundational/academic learning-to-hash
2. Deep hashing methods
3. Modern binary quantization in production retrieval
4. Benchmarks and accuracy/speed/memory tradeoffs
5. Vector-engine integration

## Files

- `SUMMARY.md` — per-method narrative summaries
- `COMPARISON.md` — accuracy vs. speed vs. memory comparison table
- `RECOMMENDATIONS.md` — concrete recipes for DocUVerse (Elasticsearch / Milvus / LanceDB)
- `papers/` — one record per paper / source (abstract, key claim, link, citation)

**IBM internal paper:** `papers/16-ASH-Tepper-Willke-IBM-2026.md` covers ASH
(Asymmetric Scalar Hashing) by Tepper (Elastic) × Willke (IBM WatsonX) — a
data-driven asymmetric scalar quantizer combining learned dim-reduction with
multi-bit scalar quantization. Generalizes ITQ and RaBitQ as special cases and
beats PQ/LOPQ/EDEN/TurboQuant/RaBitQ/LeanVec at iso-footprint on modern
embedding models. Highly relevant given DocUVerse's Elasticsearch + granite
combination.

## TL;DR

The field bifurcates into two tracks:

**Classical learning-to-hash track.** LSH (Indyk-Motwani 1998) → ITQ (Gong & Lazebnik
2011) → Spherical Hashing (Heo et al. 2015) → deep hashing (DPSH 2016, HashNet 2017,
Greedy Hash 2018, CSQ 2020). Established the theory: data-independent vs.
data-dependent hashing, Procrustean rotation for quantization-error minimization, and
end-to-end deep hash learning despite the NP-hard discrete `sign()` constraint.
Validated mostly on image retrieval (CIFAR / NUS-WIDE / ImageNet / MS COCO).

**Modern dense-retrieval track.** Train a strong dense encoder, apply 1-bit
binary quantization (32× memory), recover accuracy with continuous-vector rescoring.
This recipe — popularized by Cohere, Sentence-Transformers, and Mixedbread in 2024 —
preserves up to ~96% of MTEB retrieval quality on the best-suited models. For
end-to-end ODQA, BPR (Yamada et al. ACL 2021) shrinks the DPR index from 65 GB to
2 GB on Natural Questions / TriviaQA without QA accuracy loss, using a multi-task
binary-candidate + continuous-rerank objective. On out-of-domain BEIR, naive
learning-to-hash drops nDCG@10 by up to 14% — Thakur et al. (ReNeuIR'23) show
GPL-style domain adaptation recovers and exceeds baseline (+11.5% for BPR with
14× CPU speedup at 32× memory).

## Recommendation for DocUVerse

Default path: **binary quantization with continuous rescoring** on Milvus or LanceDB.
For out-of-domain corpora, layer in **BPR + GPL domain adaptation** before binarizing.
See `RECOMMENDATIONS.md` for engine-specific recipes.
