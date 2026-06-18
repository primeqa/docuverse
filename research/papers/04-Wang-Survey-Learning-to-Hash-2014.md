# A Survey on Learning to Hash

**Authors:** Jingdong Wang, Heng Tao Shen, Jingkuan Song, Jianqiu Ji
**Venue:** arxiv 2014 (1408.2927); TPAMI 2018 (extended)
**Links:**
- https://arxiv.org/abs/1408.2927
- https://ieeexplore.ieee.org/document/7915742/

## Verified key claim

Hashing methods divide cleanly into **data-independent** (LSH) and **data-dependent**
(learning-to-hash) families, with learning-to-hash further subdividing into
**pairwise-similarity-preserving**, **multiwise-similarity-preserving**,
**implicit-similarity-preserving**, and **quantization-based** methods. **3-0
verified** against both the 2014 arxiv survey and the 2018 TPAMI extension.

## Refuted claim

A claim that *quantization-based hashing algorithms outperform other L2H
categories across accuracy, speed, and memory* was **refuted 0-3** — the survey
notes tradeoffs, not dominance.

## Why use this survey

The clearest taxonomy of the field. When you need to pick a baseline:

- **Data-independent / LSH**: random projection or random partition methods.
  Use when training data is unavailable or distribution-shift is constant.
- **Pairwise** (DPSH, DSH, DHN): similarity supervision via point pairs.
  The default for supervised hashing.
- **Multiwise** (DTH, ranking-based): triplet or higher-order ordering loss.
  Useful when ranking matters more than binary similarity.
- **Implicit-similarity-preserving** (Spectral Hashing, AGH): no explicit
  supervision; uses graph structure.
- **Quantization-based** (ITQ, OPQ, PQ): minimize reconstruction / quantization
  error directly.

## Use as a guide, not a benchmark

The survey itself doesn't run head-to-head experiments on dense text retrieval —
its value is taxonomic and pedagogical.

## Citation

```
@article{wang2018survey,
  title={A Survey on Learning to Hash},
  author={Wang, Jingdong and Zhang, Ting and Sebe, Nicu and Shen, Heng Tao},
  journal={IEEE TPAMI},
  year={2018}
}
```
