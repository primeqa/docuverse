# A Survey on Deep Hashing Methods

**Authors:** Xiao Luo, Haixin Wang, Daqing Wu, Chong Chen, Minghua Deng, Jianqiang Huang, Xian-Sheng Hua
**Venue:** arxiv 2020 (later published in ACM TKDD)
**Link:** https://arxiv.org/abs/2003.03369

## Verified key claim

Organizes deep hashing into:

- **Deep supervised hashing**:
  - pairwise (e.g. DPSH, DHN)
  - ranking-based (e.g. DTH, DSRH)
  - pointwise (e.g. SDH-style with class probabilities)
  - quantization (e.g. DQN, DPQ)
- **Deep unsupervised hashing**:
  - similarity-reconstruction (e.g. autoencoder-based)
  - pseudo-label (e.g. DistillHash)
  - prediction-free self-supervised (e.g. SSDH)

**3-0 verified.**

## Why use this survey

When picking a deep-hashing baseline, this is the single best navigation map.
It covers:

- The full method tree with a one-paragraph mechanism summary per method.
- Standard image-retrieval benchmarks and what each method achieves on them.
- Open challenges (data efficiency, cross-modal hashing, hash for graph data).

Combined with Wang et al.'s 2014/2018 *Learning to Hash* survey, this gives the
complete picture from classical to deep era.

## Citation

```
@article{luo2020survey,
  title={A Survey on Deep Hashing Methods},
  author={Luo, Xiao and Wang, Haixin and Wu, Daqing and Chen, Chong and Deng, Minghua and Huang, Jianqiang and Hua, Xian-Sheng},
  journal={arXiv:2003.03369},
  year={2020}
}
```
