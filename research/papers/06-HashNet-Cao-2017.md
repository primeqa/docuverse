# HashNet

**Authors:** Zhangjie Cao, Mingsheng Long, Jianmin Wang, Philip S. Yu
**Venue:** ICCV 2017
**Link:** https://arxiv.org/abs/1702.00758

## Verified key claim

HashNet uses a **continuation method** that starts with a smoothed `tanh(β·x)`
and evolves `β → ∞` over training, converging to `sign()`. This solves the
vanishing-gradient problem of directly optimizing through `sign()`. **3-0 verified.**

## Mechanism

Forward pass at training:

```
h(x) = tanh(β · f(x))   with β increasing per epoch / per stage
```

- Early training: small `β` ⇒ smooth tanh, well-behaved gradients.
- Late training: large `β` ⇒ tanh approaches `sign()`, codes converge to ±1.
- At inference: `b(x) = sign(f(x))`.

Loss is the **weighted maximum likelihood** of pairwise labels — extends the DPSH
loss with a positive-weight `w_{ij}` to address class imbalance (most pairs are
dissimilar).

## Why it matters

- **Principled fix for the discrete-output gradient problem.** Unlike a fixed
  smooth approximation (like just tanh), continuation guarantees the loss
  surface gradually converges to the true objective.
- **Strong image-retrieval results** on ImageNet, NUS-WIDE, MS COCO.

## Limitation

- The β schedule is a hyperparameter requiring tuning.
- Greedy Hash (next year, NeurIPS 2018) achieves comparable accuracy with a
  simpler straight-through estimator.

## Citation

```
@inproceedings{cao2017hashnet,
  title={HashNet: Deep Learning to Hash by Continuation},
  author={Cao, Zhangjie and Long, Mingsheng and Wang, Jianmin and Yu, Philip S.},
  booktitle={ICCV},
  year={2017}
}
```
