# Cohere int8 & Binary Embeddings (Production API)

**Type:** Production product launch + technical docs
**Launched:** 2024-03-18 (Nils Reimers / Cohere)
**Links:**
- https://docs.cohere.com/docs/int8-and-binary-embeddings
- https://cohere.com/blog/int8-binary-embeddings

## Verified key claim

Cohere Embed natively supports **int8, uint8, binary, and ubinary** embedding
types as a launched product feature for vector-DB memory reduction. **3-0 verified.**

Generally available, integrated with Qdrant, Weaviate, MongoDB Atlas, Pinecone.

## API surface

```python
co.embed(
    texts=[...],
    model="embed-english-v3.0",
    embedding_types=["float", "int8", "binary"],   # any subset
    input_type="search_document",
)
# Returns embeddings.float, embeddings.int8, embeddings.binary, embeddings.ubinary
```

- **int8**: `Q ∈ [−128, 127]` per dim. **4× memory** vs. float32. Calibrated by
  Cohere from the embedding distribution.
- **uint8**: `[0, 255]` per dim. Also 4×.
- **binary**: `b ∈ {−1, +1}^d` packed into bytes. **32× memory**.
- **ubinary**: `b ∈ {0, 1}^d` packed. Also 32×; some vector DBs prefer this form.

Same model, multiple representations returned in **one call** — natural fit for
the two-stage "binary search + float rescore" pattern.

## Why it matters

- Validates the binary-quantization-with-rescoring recipe at production scale.
- Sets the API pattern that other providers (Mixedbread, Voyage) followed.
- Removes the need to roll your own packbits / calibration: the encoder is
  already trained / calibrated for the quantized representation.

## Caveat

Calibration of int8 / binary was done on Cohere's training distribution. Custom
encoders (granite-embedding, BGE, etc.) need their own calibration step or simple
`x > 0` packbits.
