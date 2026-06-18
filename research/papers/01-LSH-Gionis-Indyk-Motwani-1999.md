# Locality-Sensitive Hashing (LSH)

**Authors:** Aristides Gionis, Piotr Indyk, Rajeev Motwani
**Venue:** VLDB 1999 (extending Indyk & Motwani STOC 1998)
**Link:** https://www.cs.princeton.edu/courses/archive/spring13/cos598C/Gionis.pdf

## Verified key claim

LSH achieves sublinear approximate-NN query time of `O(d · n^(1/(1+ε)))`, improving
the prior `O(d · n^(1/ε))` bound, by hashing so that collision probability is higher
for nearby points. **3-0 verified.**

## Abstract / mechanism

A **family of hash functions** `H` is `(r₁, r₂, p₁, p₂)`-sensitive if any two
points within distance `r₁` collide with probability ≥ `p₁`, and any two points
beyond distance `r₂` collide with probability ≤ `p₂`, with `p₁ > p₂` and `r₁ < r₂`.

Build `L` independent hash tables, each using `k` concatenated hash functions as
the key. Query: hash the query into all `L` tables, examine the union of buckets.
With appropriate `k` and `L`, returns an `r₂`-approximate near neighbor with
constant probability in sublinear time.

For cosine / inner-product similarity, the canonical instantiation is **random
hyperplane LSH** (Charikar STOC 2002): `h(x) = sgn(w · x)` for `w ~ N(0, I)`.
Hamming distance over the bit-string approximates angular distance.

## Why it matters

- The **theoretical foundation** of all hashing-for-similarity work.
- **Data-independent** — no training data needed.
- Provable sublinear NN query bound.
- Weakness: needs more bits than data-dependent methods at equal accuracy. Modern
  practice uses LSH as a baseline rather than as a deployed solution.

## Refuted sub-claim

A claim that LSH *is* "random projections / bit selections on the Hamming cube"
was refuted (1-2). LSH is a broader family; random hyperplane is one specific
instantiation.

## Citation

```
@inproceedings{gionis1999similarity,
  title={Similarity search in high dimensions via hashing},
  author={Gionis, Aristides and Indyk, Piotr and Motwani, Rajeev},
  booktitle={VLDB},
  year={1999}
}
```
