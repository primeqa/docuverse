"""Load both Granite multilingual r2 embedding models and print first 10 values
of the embedding vector for a short sample sentence."""

from sentence_transformers import SentenceTransformer

MODELS = [
    "ibm-granite/granite-embedding-97m-multilingual-r2",
    "ibm-granite/granite-embedding-311m-multilingual-r2",
]

TEXT = "The quick brown fox jumps over the lazy dog."


def main():
    print(f"Text: {TEXT!r}\n")
    for name in MODELS:
        print(f"Loading {name} ...")
        model = SentenceTransformer(name, trust_remote_code=True)
        emb = model.encode(TEXT, normalize_embeddings=True)
        print(f"  dim={emb.shape[-1]}")
        print(f"  first 10: {emb[:10].tolist()}\n")


if __name__ == "__main__":
    main()
