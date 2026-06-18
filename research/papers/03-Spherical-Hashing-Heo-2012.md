# Spherical Hashing

**Authors:** Jae-Pil Heo, Youngwoon Lee, Junfeng He, Shih-Fu Chang, Sung-Eui Yoon
**Venue:** CVPR 2012 (TPAMI 2015 extended)
**Link:** http://sunglab.kaist.ac.kr/papers/SphericalHashing_TPAMI15.pdf

## Mechanism

Replaces hyperplanes (used by LSH and ITQ) with **hyperspheres**. Each hash bit
encodes whether a point lies inside or outside a learned hypersphere, defined by
a center `pᵢ` and radius `tᵢ`:

```
hᵢ(x) = +1 if ||x − pᵢ|| ≤ tᵢ else −1
```

The sphere centers and radii are jointly optimized so that:
- Each bit splits the dataset roughly 50/50 (balance).
- Bits are pairwise independent (low mutual information).

A spherical Hamming distance metric (number of bit flips weighted by sphere
overlap) is proposed for retrieval.

## Why it matters

- **Tighter geometric locality** than hyperplane partitioning. Hyperspheres are
  better matched to the Gaussian-like density of real embedding distributions.
- Reports better retrieval accuracy than ITQ at the same code length on
  GIST-1M / Tiny Images / ImageNet.

## Caveats

- More complex to train than ITQ.
- Has not gained the same ubiquity as ITQ in the literature; usually cited
  alongside ITQ rather than as a default.

## Citation

```
@article{heo2015spherical,
  title={Spherical Hashing: Binary Code Embedding with Hyperspheres},
  author={Heo, Jae-Pil and Lee, Youngwoon and He, Junfeng and Chang, Shih-Fu and Yoon, Sung-Eui},
  journal={IEEE TPAMI},
  year={2015}
}
```
