"""
Example: BioHash on DocUVerse Benchmark Data

This script demonstrates using BioHash for document retrieval on
actual benchmark datasets used in DocUVerse.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from biohash_docuverse import DocUVerseBioHash
import time
from typing import List, Dict


def load_benchmark_data(benchmark_dir: str = 'benchmark/clapnq'):
    """
    Load ClapNQ or other benchmark data.

    Args:
        benchmark_dir: Path to benchmark directory

    Returns:
        passages_path, queries_path (if they exist)
    """
    base_path = Path(__file__).parent.parent / benchmark_dir

    passages_path = base_path / 'passages.tsv'
    queries_path = base_path / 'question_dev_answerable.tsv'

    return passages_path, queries_path


def evaluate_retrieval(
    doc_hash: DocUVerseBioHash,
    queries: List[str],
    ground_truth: List[List[str]],
    top_k: int = 100
) -> Dict[str, float]:
    """
    Evaluate retrieval performance.

    Args:
        doc_hash: BioHash index
        queries: List of query strings
        ground_truth: List of lists of relevant doc IDs for each query
        top_k: Number of results to retrieve

    Returns:
        Dict with metrics: recall@k, MRR@k, etc.
    """
    recalls = []
    mrrs = []

    print(f"\nEvaluating on {len(queries)} queries...")

    for i, (query, relevant_docs) in enumerate(zip(queries, ground_truth)):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(queries)}")

        # Search
        results = doc_hash.search(query, top_k=top_k)

        # Get retrieved doc IDs
        retrieved_ids = [r['doc_id'] for r in results]

        # Compute recall@k
        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_ids)
        recall = len(relevant_set & retrieved_set) / len(relevant_set) if relevant_set else 0
        recalls.append(recall)

        # Compute MRR
        mrr = 0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

    return {
        f'recall@{top_k}': np.mean(recalls) * 100,
        f'MRR@{top_k}': np.mean(mrrs) * 100
    }


def demo_with_custom_corpus():
    """Demo: Create and search a custom document corpus."""
    print("="*80)
    print("Example 1: Custom Document Corpus")
    print("="*80)

    # Create a small research paper corpus
    papers = """doc_id\ttitle\ttext
p1\tAttention Is All You Need\tThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
p2\tBERT: Pre-training of Deep Bidirectional Transformers\tWe introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.
p3\tGPT-3: Language Models are Few-Shot Learners\tWe demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters.
p4\tResNet: Deep Residual Learning for Image Recognition\tDeep residual networks are a family of neural networks that use skip connections to jump over some layers. Typical ResNet models are implemented with double or triple layer skips that contain nonlinearities and batch normalization in between.
p5\tImageNet Classification with Deep Convolutional Networks\tWe trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0%.
p6\tMasked Autoencoders Are Scalable Vision Learners\tMasked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs.
p7\tYOLO: You Only Look Once\tWe present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities.
p8\tGAN: Generative Adversarial Networks\tWe propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.
p9\tWord2Vec: Efficient Estimation of Word Representations\tWe propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques.
p10\tGloVe: Global Vectors for Word Representation\tRecent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using vector arithmetic, but the origin of these regularities has remained opaque. We analyze and make explicit the model properties needed for such regularities to emerge.
"""

    import tempfile

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(papers)
        corpus_path = f.name

    try:
        # Create index with different embedding types
        for emb_type in ['tfidf']:  # Can add 'sbert' if installed
            print(f"\n{'='*80}")
            print(f"Building index with {emb_type.upper()} embeddings")
            print(f"{'='*80}")

            config = {}
            if emb_type == 'tfidf':
                config = {'max_features': 500, 'ngram_range': (1, 3)}

            doc_hash = DocUVerseBioHash(
                embedding_type=emb_type,
                embedding_config=config,
                hash_length=16,
                activity=0.05,
                device='cpu'
            )

            # Load and build index
            start = time.time()
            doc_hash.load_from_tsv(corpus_path)
            doc_hash.build_index(verbose=True)
            build_time = time.time() - start

            print(f"\nIndex building time: {build_time:.2f}s")

            # Test queries
            test_queries = [
                "transformer attention mechanisms",
                "computer vision convolutional networks",
                "word embeddings and representations",
                "generative models for images",
                "few-shot learning with large models"
            ]

            print(f"\n{'='*80}")
            print(f"Search Results ({emb_type.upper()})")
            print(f"{'='*80}")

            for query in test_queries:
                print(f"\nQuery: '{query}'")
                print("-" * 80)

                start = time.time()
                results = doc_hash.search(query, top_k=3)
                search_time = time.time() - start

                for result in results:
                    print(f"{result['rank']}. [Hamming={result['hamming_distance']:2d}] "
                          f"{result['metadata']['title']}")

                print(f"   Search time: {search_time*1000:.2f}ms")

    finally:
        os.unlink(corpus_path)


def demo_document_deduplication():
    """Demo: Find duplicate/near-duplicate documents."""
    print("\n" + "="*80)
    print("Example 2: Document Deduplication")
    print("="*80)

    # Create corpus with duplicates
    corpus = """doc_id\ttitle\ttext
1\tOriginal Article\tMachine learning is revolutionizing artificial intelligence research and applications.
2\tDifferent Topic\tQuantum computing promises to solve complex computational problems efficiently.
3\tNear Duplicate\tMachine learning revolutionizes artificial intelligence research and its applications.
4\tExact Duplicate\tMachine learning is revolutionizing artificial intelligence research and applications.
5\tUnrelated\tThe Mediterranean diet includes olive oil, fish, and fresh vegetables.
6\tSimilar Content\tAI research is being revolutionized by advances in machine learning.
7\tAnother Topic\tBlockchain technology enables decentralized transaction verification.
"""

    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(corpus)
        corpus_path = f.name

    try:
        doc_hash = DocUVerseBioHash(
            embedding_type='tfidf',
            hash_length=16,
            activity=0.05,
            device='cpu'
        )

        doc_hash.load_from_tsv(corpus_path)
        doc_hash.build_index(verbose=False)

        # Find duplicates
        print("\nFinding near-duplicates (Hamming distance threshold = 8)...")
        print("-" * 80)

        threshold = 8
        hash_codes = doc_hash.hash_codes

        found_pairs = set()

        for i in range(len(doc_hash.documents)):
            for j in range(i + 1, len(doc_hash.documents)):
                dist = (hash_codes[i] != hash_codes[j]).sum().item()

                if dist <= threshold:
                    pair_id = tuple(sorted([i, j]))
                    if pair_id not in found_pairs:
                        found_pairs.add(pair_id)

                        print(f"\nNear-duplicate pair (Hamming={dist}):")
                        print(f"  [{doc_hash.doc_ids[i]}] {doc_hash.documents[i][:60]}...")
                        print(f"  [{doc_hash.doc_ids[j]}] {doc_hash.documents[j][:60]}...")

        print(f"\nFound {len(found_pairs)} near-duplicate pairs")

    finally:
        os.unlink(corpus_path)


def demo_batch_search():
    """Demo: Efficient batch searching."""
    print("\n" + "="*80)
    print("Example 3: Batch Query Processing")
    print("="*80)

    # Create larger corpus
    print("\nGenerating corpus with 100 documents...")

    topics = [
        ("AI", "artificial intelligence machine learning neural networks deep learning"),
        ("Web", "javascript html css web development frontend backend"),
        ("Data", "data science analytics statistics visualization python"),
        ("Security", "cybersecurity encryption authentication network security"),
        ("Cloud", "cloud computing aws azure containers kubernetes docker")
    ]

    documents = []
    for i in range(100):
        topic_name, keywords = topics[i % len(topics)]
        text = f"Document {i} about {topic_name}. "
        text += " ".join(keywords.split() * 5)  # Repeat keywords
        documents.append(f"{i}\tTopic: {topic_name}\t{text}")

    corpus = "doc_id\ttitle\ttext\n" + "\n".join(documents)

    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(corpus)
        corpus_path = f.name

    try:
        # Build index
        doc_hash = DocUVerseBioHash(
            embedding_type='tfidf',
            hash_length=32,
            activity=0.01,  # Lower activity for larger corpus
            device='cpu'
        )

        doc_hash.load_from_tsv(corpus_path)
        print("\nBuilding index...")
        doc_hash.build_index(verbose=False)

        # Batch queries
        queries = [
            "deep learning and AI",
            "web development",
            "data analytics",
            "cloud computing infrastructure",
            "network security"
        ]

        print(f"\nProcessing {len(queries)} queries...")
        print("-" * 80)

        start = time.time()

        for i, query in enumerate(queries):
            results = doc_hash.search(query, top_k=5)

            print(f"\nQuery {i+1}: '{query}'")
            print(f"  Top result: {results[0]['metadata']['title']} "
                  f"(Hamming={results[0]['hamming_distance']})")

        total_time = time.time() - start

        print(f"\n{'='*80}")
        print(f"Batch processing completed:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg time per query: {total_time/len(queries)*1000:.2f}ms")
        print(f"  Queries per second: {len(queries)/total_time:.1f}")

    finally:
        os.unlink(corpus_path)


def demo_hash_statistics():
    """Demo: Analyze hash code properties."""
    print("\n" + "="*80)
    print("Example 4: Hash Code Statistics")
    print("="*80)

    # Create diverse corpus
    corpus = """doc_id\ttitle\ttext
1\tML\tMachine learning algorithms learn patterns from data to make predictions and decisions.
2\tDL\tDeep learning uses multi-layer neural networks to learn hierarchical representations.
3\tNLP\tNatural language processing enables computers to understand and generate human language.
4\tCV\tComputer vision allows machines to interpret and analyze visual information from images.
5\tRL\tReinforcement learning trains agents to make sequences of decisions through trial and error.
"""

    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(corpus)
        corpus_path = f.name

    try:
        doc_hash = DocUVerseBioHash(
            embedding_type='tfidf',
            hash_length=8,
            activity=0.10,
            device='cpu'
        )

        doc_hash.load_from_tsv(corpus_path)
        doc_hash.build_index(verbose=False)

        hash_codes = doc_hash.hash_codes

        print("\nHash Code Statistics:")
        print("-" * 80)
        print(f"Number of documents: {len(doc_hash.documents)}")
        print(f"Hash dimensions (m): {hash_codes.shape[1]}")
        print(f"Active neurons (k): {doc_hash.hash_length}")
        print(f"Activity level: {doc_hash.activity*100:.1f}%")

        # Verify sparsity
        active_per_doc = (hash_codes == 1).sum(dim=1).float()
        print(f"\nSparsity verification:")
        print(f"  Expected active: {doc_hash.hash_length}")
        print(f"  Actual active (mean): {active_per_doc.mean().item():.1f}")
        print(f"  Actual active (std): {active_per_doc.std().item():.3f}")

        # Pairwise distances
        distances = torch.cdist(hash_codes.float(), hash_codes.float(), p=1) / 2

        print(f"\nPairwise Hamming distances:")
        for i in range(len(doc_hash.documents)):
            for j in range(i + 1, len(doc_hash.documents)):
                dist = distances[i, j].item()
                print(f"  Doc {i+1} ↔ Doc {j+1}: {int(dist):2d}")

        # Compute distance distribution
        upper_tri_dists = distances[torch.triu(torch.ones_like(distances), diagonal=1) == 1]
        print(f"\nDistance distribution:")
        print(f"  Min: {upper_tri_dists.min().item():.1f}")
        print(f"  Max: {upper_tri_dists.max().item():.1f}")
        print(f"  Mean: {upper_tri_dists.mean().item():.1f}")
        print(f"  Median: {upper_tri_dists.median().item():.1f}")

    finally:
        os.unlink(corpus_path)


if __name__ == '__main__':
    print("\n")
    print("█" * 80)
    print("  BioHash for DocUVerse - Practical Examples")
    print("█" * 80)

    # Run all examples
    demo_with_custom_corpus()
    demo_document_deduplication()
    demo_batch_search()
    demo_hash_statistics()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)

    print("\nNext steps:")
    print("  1. Try with your own corpus: doc_hash.load_from_tsv('your_corpus.tsv')")
    print("  2. Experiment with different embeddings: 'tfidf', 'sbert'")
    print("  3. Tune hyperparameters: hash_length, activity")
    print("  4. Save and reuse index: doc_hash.save_index('./my_index')")
