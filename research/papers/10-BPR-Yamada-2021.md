# BPR — Binary Passage Retriever

**Authors:** Ikuya Yamada, Akari Asai, Hannaneh Hajishirzi
**Venue:** ACL 2021
**Link:** https://arxiv.org/abs/2106.00882
**Code:** https://github.com/studio-ousia/bpr

## Verified key claims (all 3-0)

- BPR integrates **learning-to-hash into DPR** to represent passages with binary
  codes.
- BPR uses a **multi-task training objective**: binary candidate generation +
  continuous-vector reranking.
- BPR shrinks the passage index from **65 GB to 2 GB (~32×)** on Natural Questions
  and TriviaQA **without QA accuracy loss**.

## Mechanism

DPR encodes a passage `p` with a BERT-based encoder into `h_p ∈ R^d`. BPR adds a
binary head:

```
b_p = sign(W · h_p)         (passage binary code, ∈ {−1, +1}^c)
```

**Two-stage retrieval:**

1. **Candidate generation** — Hamming similarity over binary codes returns the
   top-K' candidates (e.g. K' = 1000).
2. **Reranking** — recompute float dot product `qᵀh_p` for those K', return top-K.

**Multi-task training loss:**

```
L = L_cand(b_q, b_p)   +   L_rerank(h_q, h_p)
```

- `L_cand`: contrastive loss over binary similarity (the "candidate" task).
- `L_rerank`: standard DPR contrastive loss over continuous dot product (the
  "reranking" task).

The binary head is trained alongside the main encoder; gradient flows through
the `sign()` via straight-through estimation.

## Why it matters

- The **strongest result for binary dense retrieval in ODQA**: 32× memory
  reduction with no end-to-end QA loss.
- Top-20 / Top-100 retrieval recall preserved on NQ and TriviaQA.
- Minor Top-1 retrieval gap (52.5 → 49.0 NQ) — but Top-1 is not the metric the
  reader sees; reader EM is preserved.
- Full open-source implementation reproducing the paper.

## Limitations on out-of-domain corpora

Thakur et al. (ReNeuIR'23, see paper 11) show that **naively transferring** BPR to
BEIR loses up to 14% nDCG@10 — but **GPL-style domain adaptation** recovers it
to **+11.5% nDCG@10** with **14× CPU speedup**. So BPR is the right technique;
the work is in the domain-adapted training.

## Citation

```
@inproceedings{yamada2021efficient,
  title={Efficient Passage Retrieval with Hashing for Open-domain Question Answering},
  author={Yamada, Ikuya and Asai, Akari and Hajishirzi, Hannaneh},
  booktitle={ACL},
  year={2021}
}
```
