# Greedy Hash

**Authors:** Shupeng Su, Chao Zhang, Kai Han, Yonghong Tian
**Venue:** NeurIPS 2018
**Link:** https://papers.nips.cc/paper/2018/hash/13f3cf8c531952d72e5847c4183e6910-Abstract.html

## Verified key claims

- Deep hashing optimization is **fundamentally NP-hard** due to the discrete
  `sign()` constraint on outputs. **3-0 verified.**
- Greedy Hash uses `sign()` strictly in the **forward pass**, with the gradient
  copied through unchanged in the **backward pass** — an "identity straight-through
  estimator". **3-0 verified.**

## Mechanism

The cleanest deep-hashing trick. Given continuous output `h = f(x)`:

```
forward:   b = sign(h)            # discrete codes used in loss
backward:  ∂L/∂h := ∂L/∂b         # gradient passes through identically
```

Plus a "code-balancing" regularizer encouraging `Σ_i b_i ≈ 0` per dimension
(each bit is informative, not collapsed).

## Why it matters

- **Trivial to implement** — a single autograd custom op (or `b.detach() + h`
  trick in PyTorch).
- **No continuation schedule** like HashNet; no temperature; no tanh anneal.
- **Eliminates the quantization gap** — the network sees the true binary codes
  during training, not a smoothed proxy.
- Empirically competitive with HashNet and DPSH on CIFAR-10, NUS-WIDE, ImageNet.

## Caveat

- Straight-through is a **heuristic**, not a principled relaxation. Sometimes
  trains less stably than HashNet's continuation. In practice, both work; pick
  Greedy Hash for simplicity, HashNet for theoretical comfort.

## Why it's our recommendation for *training* binary heads in DocUVerse

If a future DocUVerse experiment trains a custom binary embedding head end-to-end
(rather than just packbits-quantizing a frozen encoder), Greedy Hash is the
easiest baseline — drop-in PyTorch:

```python
class StraightThroughSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)
    @staticmethod
    def backward(ctx, g):
        return g  # identity
```

## Citation

```
@inproceedings{su2018greedy,
  title={Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN},
  author={Su, Shupeng and Zhang, Chao and Han, Kai and Tian, Yonghong},
  booktitle={NeurIPS},
  year={2018}
}
```
