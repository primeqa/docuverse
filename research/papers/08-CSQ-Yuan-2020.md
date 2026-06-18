# CSQ — Central Similarity Quantization

**Authors:** Li Yuan, Tao Wang, Xiaopeng Zhang, Francis E.H. Tay, Zequn Jie, Wei Liu, Jiashi Feng
**Venue:** CVPR 2020
**Link:** https://arxiv.org/abs/1908.00347

## Verified key claims

- CSQ uses a **global "central similarity" metric** where similar pairs converge to
  a common Hadamard-derived center and dissimilar pairs to different centers in
  Hamming space. **3-0 verified.**
- Reports **3–20% mAP gains** over prior deep-hashing SOTA on CIFAR-10, NUS-WIDE,
  ImageNet, MS COCO and video-retrieval benchmarks. **3-0 verified** (self-reported).

## Mechanism

Most prior deep hashing methods use **pairwise** or **triplet** similarity — local,
quadratic, slow to converge. CSQ replaces this with **global hash centers**:

1. Pre-compute `K` hash centers `{c_1, ..., c_K} ∈ {−1,+1}^c`, one per class.
   Centers are derived from a **Hadamard matrix**, which guarantees pairwise
   Hamming distance `c/2` between centers — maximally separated on the
   hypercube.
2. Loss pulls each sample's binary code toward its class center and (implicitly,
   via Hadamard structure) pushes away from others:
   ```
   L_center = Σ_i BCE(c_{y_i}, b_i)
   ```
   where BCE is binary cross-entropy.
3. Add a quantization regularizer `||b_i − u_i||²` per usual.

## Why it matters

- **Linear in batch size** rather than quadratic in pair count — much faster
  training.
- **Globally consistent** code geometry — Hadamard centers maximize inter-class
  Hamming distance by construction.
- Strongest reported deep-hashing results on standard image-retrieval benchmarks
  at the time of publication.

## Caveats for text retrieval

- All evaluation is on **image / video** benchmarks with class labels.
- Adapting to dense **text** retrieval requires:
  - A class structure (or generated pseudo-classes via clustering).
  - A non-trivial choice of code length `c` (must be a power of 2 for full
    Hadamard, or use partial Hadamard).
- Transfer to BEIR / MTEB has **not been demonstrated** in the verified literature.

## Citation

```
@inproceedings{yuan2020central,
  title={Central Similarity Quantization for Efficient Image and Video Retrieval},
  author={Yuan, Li and Wang, Tao and Zhang, Xiaopeng and Tay, Francis E.H. and Jie, Zequn and Liu, Wei and Feng, Jiashi},
  booktitle={CVPR},
  year={2020}
}
```
