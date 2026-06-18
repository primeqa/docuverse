# Iterative Quantization (ITQ)

**Authors:** Yunchao Gong, Svetlana Lazebnik
**Venue:** CVPR 2011 (TPAMI 2013 extended)
**Link:** https://slazebni.cs.illinois.edu/publications/ITQ.pdf
**Citations:** ~1880 (canonical unsupervised hashing baseline)

## Verified key claim

ITQ is a **Procrustean approach** combining PCA projection with an optimal
orthogonal rotation that aligns data with the binary hypercube vertices,
minimizing quantization error. **Outperforms LSH and Spectral Hashing as
unsupervised binary-code baselines.** Both claims 3-0 verified.

## Mechanism

Given centered data `X ∈ R^{n×d}` and target code length `c`:

1. **PCA projection**: `V = X · W`, where `W` is the top-`c` eigenvectors of `XᵀX`.
2. **Rotation search**: find orthogonal `R ∈ R^{c×c}` minimizing
   `||sgn(VR) − VR||²_F`. Alternates between:
   - **Update binary codes**: `B = sgn(VR)`
   - **Update rotation**: SVD of `BᵀV = SΩŜᵀ` → `R = ŜSᵀ` (orthogonal Procrustes).

The orthogonal rotation does not change Euclidean distances in the projected
space, but lets the data align with the `2^c` binary hypercube vertices —
minimizing quantization error.

## Why it matters

- **Canonical unsupervised hashing baseline** — every modern method compares to it.
- Beats LSH and Spectral Hashing at short code lengths.
- No labels needed; only PCA + a few SVD iterations.
- Drop-in upgrade over random hyperplane LSH if you have a representative training
  set.

## Variants

- **Supervised ITQ** uses CCA instead of PCA when labels are available.
- The **rotation step** has been reused in many later methods (e.g., Optimized
  Product Quantization).

## Citation

```
@inproceedings{gong2011iterative,
  title={Iterative Quantization: A Procrustean Approach to Learning Binary Codes},
  author={Gong, Yunchao and Lazebnik, Svetlana},
  booktitle={CVPR},
  year={2011}
}
```
