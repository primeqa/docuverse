# Engine Integration Notes

Production vector DBs and how they support binary vectors. Sources verified
in the deep-research run; specifics may evolve, so re-check release notes
before committing.

## Milvus — first-class binary support

**Source:** https://milvus.io/docs/binary.md

- `BinaryVector` field type.
- Metrics: `HAMMING`, `JACCARD`.
- Indexes: `BIN_FLAT`, `BIN_IVF_FLAT`.
- Milvus 2.4+ adds scalar quantization (8-bit) as a middle ground.

This is the **cleanest engine** for binary-only experiments.

## LanceDB — pragmatic columnar approach

- No native `BinaryVector` type, but `fixed_size_list<uint8>` works perfectly.
- Hamming distance can be done with a custom UDF or computed in Python over
  retrieved candidates.
- The columnar storage is a natural fit for two-stage retrieval: keep `emb_bin`
  and `emb_float` as sibling columns, query on `emb_bin`, JOIN to `emb_float`
  for rescoring.
- Already integrated in DocUVerse as `LanceDBHybridEngine` (commits c659c97,
  d4badbd) — adding a binary-then-rescore path fits the existing composition
  pattern.

## Elasticsearch — int8 mature, 1-bit catching up

**Source:** https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch

- `dense_vector` with `element_type=byte` for **int8** (4× memory) — production
  ready since ES 8.6+.
- Lucene 9.12 / ES 8.16 introduced **Better Binary Quantization (BBQ)**, an
  optimized 1-bit scalar quantization built into HNSW. Newer; production
  hardening still in progress.
- For 1-bit retrieval today: prefer Milvus or LanceDB; revisit ES as BBQ
  matures.

## Qdrant / Weaviate / Pinecone — supported via Cohere integration

These vector DBs offer Cohere-int8 / Cohere-binary integration out of the box.
Less of a fit for DocUVerse (which already wraps Elasticsearch / Milvus /
LanceDB / ChromaDB / FAISS), but worth noting if a comparison is run.

## Refuted sub-claim

A claim that *"Hamming distance over binary embeddings can be computed in
roughly 2 CPU cycles"* was **refuted (1-2)**. Cost is `O(d/64)` `popcnt`
instructions per comparison. For d=768, ~12 popcnts plus a horizontal sum —
fast, but not 2 cycles, and SIMD/AVX-512 availability changes the picture.

## Sources

- Milvus binary vector docs — https://milvus.io/docs/binary.md
- Elastic search-labs BBQ blog — https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch
- Qdrant binary quantization article — https://qdrant.tech/articles/binary-quantization/
