"""
Quick test script for text hashing functionality.
Verifies that text hashing works without dependencies.
"""

import sys
import os

# Test imports
print("Testing imports...")
try:
    from biohash_implementation import BioHash
    print("✓ BioHash imported")
except Exception as e:
    print(f"✗ Failed to import BioHash: {e}")
    sys.exit(1)

try:
    from biohash_text import (
        TextEmbedder,
        TfidfEmbedder,
        BioHashText
    )
    print("✓ Text hashing modules imported")
except Exception as e:
    print(f"✗ Failed to import text modules: {e}")
    sys.exit(1)

# Test TF-IDF embedder
print("\n" + "="*60)
print("Test 1: TF-IDF Embedder")
print("="*60)

try:
    documents = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "Cats and dogs are pets",
        "Parks are nice for walking",
        "The mat is on the floor"
    ]

    # Create and fit embedder
    embedder = TfidfEmbedder(max_features=100, ngram_range=(1, 1))
    embedder.fit(documents)

    print(f"✓ TF-IDF fitted on {len(documents)} documents")
    print(f"  Vocabulary size: {embedder.vector_dim}")

    # Embed documents
    vectors = embedder.embed(documents)
    print(f"✓ Embedded documents: shape {vectors.shape}")

    # Check properties
    assert vectors.shape[0] == len(documents), "Wrong number of vectors"
    assert vectors.shape[1] == embedder.vector_dim, "Wrong vector dimension"

    print("✓ All TF-IDF tests passed")

except Exception as e:
    print(f"✗ TF-IDF test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test BioHashText
print("\n" + "="*60)
print("Test 2: BioHashText")
print("="*60)

try:
    # Create BioHashText
    biohash_text = BioHashText(
        embedder=embedder,
        hash_length=4,
        activity=0.10,
        max_epochs=20,
        device='cpu'
    )

    print(f"✓ BioHashText created")
    print(f"  Hash neurons (m): {biohash_text.num_neurons}")
    print(f"  Active neurons (k): {biohash_text.hash_length}")

    # Train
    print("\n  Training...")
    biohash_text.fit(documents, verbose=False)
    print("✓ Training completed")

    # Generate hash codes
    hash_codes = biohash_text.hash(documents)
    print(f"✓ Generated hash codes: shape {hash_codes.shape}")

    # Check properties
    assert hash_codes.shape[0] == len(documents), "Wrong number of hash codes"
    assert hash_codes.shape[1] == biohash_text.num_neurons, "Wrong hash dimension"

    # Check sparsity
    active = (hash_codes == 1).sum(dim=1)
    print(f"✓ Active neurons per document: {active[0].item()} "
          f"(expected {biohash_text.hash_length})")

    assert (active == biohash_text.hash_length).all(), "Wrong sparsity"

    print("✓ All BioHashText tests passed")

except Exception as e:
    print(f"✗ BioHashText test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test search
print("\n" + "="*60)
print("Test 3: Text Search")
print("="*60)

try:
    # Search for similar documents
    query = "cats and animals"

    print(f"\nQuery: '{query}'")
    print("-" * 60)

    results = biohash_text.search(
        query=query,
        database_texts=documents,
        top_k=3
    )

    print("✓ Search completed")
    print("\nTop 3 results:")
    for rank, text, dist in results:
        print(f"  {rank}. [Hamming={dist}] {text}")

    # Verify results format
    assert len(results) == 3, "Wrong number of results"
    assert all(isinstance(r[0], int) for r in results), "Bad rank format"
    assert all(isinstance(r[1], str) for r in results), "Bad text format"
    assert all(isinstance(r[2], int) for r in results), "Bad distance format"

    print("\n✓ All search tests passed")

except Exception as e:
    print(f"✗ Search test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test DocUVerse integration (basic)
print("\n" + "="*60)
print("Test 4: DocUVerse Integration (Basic)")
print("="*60)

try:
    from biohash_docuverse import DocUVerseBioHash
    import tempfile

    # Create temp TSV file
    tsv_content = """doc_id\ttitle\ttext
1\tFirst Doc\tThis is the first document about cats
2\tSecond Doc\tThis is about dogs and animals
3\tThird Doc\tParks are great for pets
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(tsv_content)
        tsv_path = f.name

    try:
        # Create DocUVerse hash
        doc_hash = DocUVerseBioHash(
            embedding_type='tfidf',
            hash_length=4,
            activity=0.10,
            device='cpu'
        )

        print("✓ DocUVerseBioHash created")

        # Load documents
        doc_hash.load_from_tsv(tsv_path)
        print(f"✓ Loaded {len(doc_hash.documents)} documents")

        # Build index
        doc_hash.build_index(verbose=False)
        print("✓ Index built")

        # Search
        results = doc_hash.search("pets", top_k=2)
        print("✓ Search completed")

        print("\nSearch results:")
        for r in results:
            print(f"  {r['rank']}. [{r['doc_id']}] {r['metadata']['title']}")

        assert len(results) == 2, "Wrong number of results"

        print("\n✓ All DocUVerse integration tests passed")

    finally:
        # Clean up
        os.unlink(tsv_path)

except Exception as e:
    print(f"✗ DocUVerse integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed
print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nText hashing is working correctly.")
print("\nYou can now:")
print("  1. Run demos: python biohash_text.py")
print("  2. Run examples: python example_biohash_benchmark.py")
print("  3. Use with your own data - see README_TEXT_HASHING.md")
