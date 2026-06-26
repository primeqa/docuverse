# NTK-Aware RoPE Scaling

## Background: RoPE (Rotary Position Embeddings)

RoPE encodes position by rotating query/key vectors at each position by an angle proportional to `position * theta`, where:

```
theta_i = base^(-2i/d)    (base = rope_theta, i = dimension index, d = head_dim)
```

For granite_ml_r2 (ModernBERT): `global_rope_theta=150000`, `local_rope_theta=160000`, `head_dim=64`, `max_position_embeddings=8192`.

Beyond the trained range (0-8191), rotation angles become values never seen during training, causing degraded representations.

## Naive Approaches (and their problems)

### Position Interpolation (Linear Scaling)

Divide positions by a scale factor so they fit back into the trained range. E.g., to extend 8K to 16K, use `position / 2`.

**Problem**: Compresses nearby positions closer together, reducing the model's ability to distinguish adjacent tokens. Loses local resolution.

### Direct Extrapolation

Just use positions > 8192 as-is.

**Problem**: High-frequency RoPE dimensions (small `i`) rotate into completely unseen angles. Rapid quality collapse.

## NTK-Aware Scaling

Key insight (bloc97, June 2023): instead of scaling positions, **scale the base** (`rope_theta`). This distributes the extension across all frequency bands in the Fourier space rather than compressing them uniformly.

```python
# To extend context by factor alpha:
new_base = base * alpha^(d / (d - 2))

# Example: extend 8K -> 32K (alpha = 4), base = 150000, d = 64
new_base = 150000 * 4^(64/62) = 150000 * 4.186 = 627,900
```

### Why It Works

RoPE encodes position as a mix of frequencies (like a Fourier decomposition of position). NTK scaling:

- Preserves **high-frequency** components (nearby token distinctions) -- barely changes them
- Stretches **low-frequency** components (long-range position awareness) -- where the extra range is needed

This mirrors how the Neural Tangent Kernel (NTK) theory describes how networks interpolate in different frequency bands, hence the name.

## Variants

### NTK-Aware (Static)

Single base scaling, applied uniformly. Simple but somewhat blunt.

### Dynamic NTK (Code Llama style)

Scale the base dynamically based on actual sequence length at inference time. Only kicks in when `seq_len > max_position_embeddings`:

```python
if seq_len > max_pos:
    alpha = seq_len / max_pos
    new_base = base * alpha^(d / (d - 2))
```

### YaRN (Yet another RoPE extensioN)

Combines NTK scaling with attention temperature correction and per-dimension interpolation. Dimensions split into three groups:

- **Low frequency** -> interpolate (like position interpolation)
- **High frequency** -> don't scale (keep local resolution)
- **Middle** -> blend between the two

YaRN generally gives the best quality for large extensions (4x+).

## Practical Estimates for granite_ml_r2 on NVIDIA L40S (48 GB)

### Context Extension Estimates

| Target Context | Scale alpha | Static NTK Base | Expected Quality     |
|----------------|-------------|-----------------|----------------------|
| 16K            | 2x          | ~310,000        | Good                 |
| 32K            | 4x          | ~628,000        | Decent               |
| 64K            | 8x          | ~1,310,000      | Noticeable degradation |
| 128K+          | 16x+        | --              | Needs fine-tuning    |

### GPU Memory Estimates (Inference, batch_size=1, bf16)

Model weights: ~600 MB. With flash attention, memory is linear in seq_len.

| Seq Length | Est. Memory | Fits L40S? |
|------------|-------------|------------|
| 8,192      | ~2-3 GB     | Yes        |
| 32,768     | ~4-5 GB     | Yes        |
| 131,072    | ~10-15 GB   | Yes        |
| 524,288    | ~35-45 GB   | Borderline |

Without flash attention, global attention layers produce O(seq_len^2) matrices. Max seq_len without FA: ~30,000-40,000 tokens.

### Training Memory at seq_len=8192

| Component                        | Memory    |
|----------------------------------|-----------|
| Weights + Gradients (bf16)       | ~1.2 GB   |
| Optimizer states (fp32 m+v)      | ~2.4 GB   |
| Activations (22 layers, bs=1)    | ~3-5 GB   |
| Overhead                         | ~2 GB     |
| **Total (bs=1)**                 | ~10-12 GB |

Max batch_size at seq_len=8192: ~8-12 (no grad checkpointing), ~20-30+ (with grad checkpointing).

## Implementation

### Modifying rope_theta directly

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("path/to/model")

alpha = 2.0
d = 64  # head_dim
for layer in model.layers:
    layer.self_attn.rotary_emb.base = 150000 * alpha ** (d / (d - 2))
```

### Using transformers rope_scaling config

```json
{
  "rope_scaling": {
    "type": "dynamic",
    "factor": 4.0
  }
}
```

## Caveats

- Without fine-tuning, 2x extension via NTK scaling usually works reasonably. Beyond 4x, quality drops noticeably.
- Short fine-tuning (few hundred steps) on long-context data after applying NTK scaling dramatically improves results at extended lengths.
- For embedding/retrieval models, quality degradation at extended context may matter more than for generative models, since retrieval relies on precise similarity scores. Benchmark retrieval metrics (recall@k, MRR) at extended lengths before deploying.
- ModernBERT's `local_attention=128` window means most layers only attend locally regardless. The global attention layers (every 3rd) are the ones most affected by context extension.

## References

- bloc97, "NTK-Aware Scaled RoPE" (Reddit, June 2023)
- Code Llama paper (Meta, 2023) -- Dynamic NTK scaling
- YaRN paper: "YaRN: Efficient Context Window Extension of Large Language Models" (Peng et al., 2023)
