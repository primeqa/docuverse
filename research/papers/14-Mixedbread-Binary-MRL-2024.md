# Mixedbread Binary MRL

**Authors:** Mixedbread AI team (Aamir Shakir et al.)
**Type:** Blog / model release
**Date:** 2024
**Link:** https://www.mixedbread.com/blog/binary-mrl

## Idea

Combine **Matryoshka Representation Learning** (truncatable embedding dimensions —
Kusupati et al. 2022) with **binary quantization**. Lets you tune two knobs
independently:

1. **Code length** (768 → 512 → 256 → 128 → 64 dims) via Matryoshka truncation.
2. **Bits per dim** (32 → 8 → 1) via int8 / binary quantization.

The encoder is **trained jointly** so binary codes at the truncated dimensions
are still high-quality — not just naively chopped from a 768-d float32 embedding.

## Why it matters

- The most flexible recipe today. You can pick a memory budget and let the
  encoder give you the best representation at that budget without retraining.
- Concrete result: on MTEB, `mxbai-embed-large-v1` with binary-MRL preserves the
  bulk of accuracy while reducing memory **64×–128×** (vs. 32× for plain binary,
  combined with 2–4× from MRL truncation).
- Open-source models on HuggingFace.

## Caveat

- **Not peer-reviewed.** Blog post + HF model card only. The numbers are
  self-reported.
- The Matryoshka training strategy itself is from a published paper
  (Kusupati et al. NeurIPS 2022); the binary combination is the new piece.

## Practical use in DocUVerse

If you adopt Mixedbread embeddings (or train your own Matryoshka head on
granite-embedding), the truncate-then-binarize pattern fits cleanly into the
existing pipeline:

```python
# Training time: Matryoshka loss across {64, 128, 256, 512, 768}-d truncations
# Serve time:
emb_full = encoder(text)                       # (768,) float32
emb_trunc = emb_full[:256]                     # MRL truncation
emb_bin   = np.packbits(emb_trunc > 0)         # 32 bytes per vec → 256× from baseline
```

## Citation (informal)

Mixedbread blog, "Binary MRL", 2024. https://www.mixedbread.com/blog/binary-mrl

For the underlying MRL technique:

```
@inproceedings{kusupati2022matryoshka,
  title={Matryoshka Representation Learning},
  author={Kusupati, Aditya and ... },
  booktitle={NeurIPS},
  year={2022}
}
```
