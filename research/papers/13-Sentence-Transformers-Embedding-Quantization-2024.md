# Sentence-Transformers Embedding Quantization

**Authors:** Aamir Shakir, Tom Aarsen, Nils Reimers (joint UKPLab × Mixedbread)
**Type:** Reference implementation + HuggingFace blog
**Date:** 2024
**Links:**
- https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/embedding-quantization/README.md
- https://huggingface.co/blog/embedding-quantization

## Verified key claims

- Binary quantization converts float32 embedding values to **1 bit per dim** via
  `x > 0`, packed with `np.packbits` — **32× memory reduction**. **3-0 verified.**
- Coupled with a **continuous-vector rescoring** step, preserves **up to ~96.45%**
  of MTEB NDCG@10 on the best-suited models. **3-0 verified.**

Specific MTEB numbers:

| Model | float32 NDCG@10 | binary+rescore NDCG@10 | preservation |
|---|---|---|---|
| mxbai-embed-large-v1 | 54.39 | 52.46 | **96.45%** |
| e5-base-v2 | — | — | ~74% |
| all-MiniLM-L6-v2 | — | — | ~94% |

(Numbers are from the sentence-transformers README; the strong-model figure
of ~96% is widely repeated across Cohere, Mixedbread, Vespa, Qdrant blogs.)

## Reference implementation

```python
from sentence_transformers.quantization import quantize_embeddings

# At ingestion
emb_bin = quantize_embeddings(model.encode(passages), precision="binary")
# emb_bin.shape == (N, d/8) uint8

# Two-stage retrieval
def search(query, K=10, K_prime=100):
    q_float = model.encode(query)
    q_bin = quantize_embeddings(q_float[None], precision="binary")[0]

    # Stage 1: Hamming search returns K' candidates
    candidates = hamming_topk(q_bin, emb_bin, K_prime)

    # Stage 2: rescore with float
    scores = q_float @ emb_float[candidates].T
    return candidates[scores.argsort()[-K:][::-1]]
```

## Why it matters

- The **practitioner's reference implementation**. Drop-in for any
  sentence-transformers encoder.
- Demonstrates the recipe works for non-API encoders (e.g. anything on
  HuggingFace).
- Shows the model-quality dependence: strong encoders preserve ~96%, weaker
  ones drop to 74–94%.

## Refuted sub-claim

A claim that "Hamming distance computes in ~2 CPU cycles" was **refuted (1-2)**.
Real cost is `O(d/64)` `popcnt` instructions per comparison, plus a horizontal
sum. Fast, but not 2 cycles, and SIMD/AVX availability matters in practice.
