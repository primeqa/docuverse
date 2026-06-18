# Domain-Adapted Dense Retrieval with Binary Embeddings (Thakur et al.)

**Authors:** Nandan Thakur, Nils Reimers, Jimmy Lin
**Venue:** ReNeuIR 2023 workshop @ SIGIR
**Link:** https://arxiv.org/abs/2205.11498

The most actionable paper for choosing **what to do** when binarizing dense
retrievers for out-of-domain corpora. Authors are the BEIR / sentence-transformers
leads.

## Verified key claims (all 3-0)

- Naive learning-to-hash on a strong dense retriever (TAS-B) loses **up to −14%
  nDCG@10 zero-shot on BEIR**.
- **Domain-adapted BPR** (GPL-style) achieves **+11.5% nDCG@10** improvement,
  **32× memory reduction**, **14× CPU speedup**.
- **Domain-adapted JPQ** achieves **+8.2% nDCG@10**, **32× memory**, **2× CPU
  speedup**.

## Setup

- **Backbone**: TAS-B dense retriever.
- **Domain adaptation**: GPL (Generative Pseudo-Labeling). A T5-style query
  generator produces synthetic queries on the target corpus; cross-encoder scores
  pseudo-labels; the student retriever is fine-tuned with MarginMSE.
- **Quantization heads**:
  - **BPR**: binary code + continuous reranking, trained jointly.
  - **JPQ**: end-to-end joint training of product quantization with the
    retriever.
- **Eval**: BEIR benchmark (heterogeneous corpora — biomedical, finance,
  scientific, news, etc.).

## Why this paper is the practical anchor

Most binary-quantization claims (Cohere, Sentence-Transformers, Mixedbread) come
from MTEB — which is **STS-leaning** and overlaps the encoders' training data.
BEIR is **out-of-domain**. The fact that naive binarization drops 14% on BEIR
is the warning, and the fact that domain-adapted BPR recovers and exceeds is the
prescription.

## Translation to DocUVerse

If your corpus is:

- **In-domain** (similar to encoder training data): naive binary + rescoring is
  fine. Expect ~95% accuracy preserved.
- **Out-of-domain** (legal, medical, code, niche enterprise data): plan for
  GPL-style adaptation before binarizing. Expect 14% drop without it,
  +11.5% gain with it.

## Citation

```
@inproceedings{thakur2023domain,
  title={Domain Adaptation for Dense Retrieval through Self-Supervision by Pseudo-Relevance Labeling},
  author={Thakur, Nandan and Reimers, Nils and Lin, Jimmy},
  booktitle={ReNeuIR Workshop @ SIGIR},
  year={2023}
}
```
