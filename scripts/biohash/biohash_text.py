"""
BioHash for Text: Bio-Inspired Hashing for Text Similarity Search

This module extends BioHash to work with text data using various embeddings:
- TF-IDF (sparse bag-of-words)
- Word2Vec/GloVe (pre-trained word embeddings)
- Sentence-BERT (contextual embeddings)
- Custom embeddings from any model

Supports:
- Document hashing
- Semantic search
- Duplicate detection
- Text clustering
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple
from collections import Counter
import re
from biohash_implementation import BioHash, compute_mAP


class TextEmbedder:
    """Base class for text embedding methods."""

    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim

    def embed(self, texts: List[str]) -> torch.Tensor:
        """Convert texts to vectors."""
        raise NotImplementedError

    def embed_single(self, text: str) -> torch.Tensor:
        """Convert single text to vector."""
        return self.embed([text])[0]


class TfidfEmbedder(TextEmbedder):
    """
    TF-IDF text embedder.
    Uses scikit-learn's TfidfVectorizer.
    """

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            max_df=0.95,
            min_df=1  # Changed from 2 to 1 to handle small test corpora
        )
        self.fitted = False
        super().__init__(vector_dim=max_features)

    def fit(self, texts: List[str]):
        """Fit TF-IDF vocabulary on texts."""
        self.vectorizer.fit(texts)
        self.fitted = True
        self.vector_dim = len(self.vectorizer.get_feature_names_out())
        print(f"TF-IDF vocabulary size: {self.vector_dim}")

    def embed(self, texts: List[str]) -> torch.Tensor:
        """Convert texts to TF-IDF vectors."""
        if not self.fitted:
            raise ValueError("Must call fit() before embed()")

        # Get sparse matrix
        tfidf_matrix = self.vectorizer.transform(texts)

        # Convert to dense tensor
        vectors = torch.FloatTensor(tfidf_matrix.toarray())

        return vectors


class AverageWordEmbedder(TextEmbedder):
    """
    Average word embeddings (Word2Vec, GloVe, FastText, etc.)
    Averages pre-trained word vectors for all words in text.
    """

    def __init__(self, embedding_path: Optional[str] = None, vector_dim: int = 300):
        """
        Args:
            embedding_path: Path to pre-trained embeddings (word2vec/glove format)
            vector_dim: Dimension of word vectors
        """
        super().__init__(vector_dim=vector_dim)
        self.word_vectors = {}

        if embedding_path:
            self.load_embeddings(embedding_path)

    def load_embeddings(self, path: str):
        """Load pre-trained word embeddings."""
        print(f"Loading embeddings from {path}...")
        count = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < self.vector_dim + 1:
                    continue

                word = parts[0]
                try:
                    vector = np.array([float(x) for x in parts[1:self.vector_dim+1]])
                    self.word_vectors[word] = vector
                    count += 1
                except:
                    continue

                if count % 50000 == 0:
                    print(f"  Loaded {count} words...")

        print(f"Loaded {len(self.word_vectors)} word vectors")

    def load_glove(self, glove_path: str):
        """Load GloVe embeddings (same format as load_embeddings)."""
        self.load_embeddings(glove_path)

    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize and clean text."""
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Tokenize
        words = text.split()

        return words

    def embed(self, texts: List[str]) -> torch.Tensor:
        """Convert texts to averaged word vectors."""
        vectors = []

        for text in texts:
            words = self.preprocess_text(text)

            # Get word vectors
            word_vecs = []
            for word in words:
                if word in self.word_vectors:
                    word_vecs.append(self.word_vectors[word])

            # Average word vectors
            if len(word_vecs) > 0:
                avg_vec = np.mean(word_vecs, axis=0)
            else:
                # Default to zero vector if no words found
                avg_vec = np.zeros(self.vector_dim)

            vectors.append(avg_vec)

        return torch.FloatTensor(np.array(vectors))


class SentenceBERTEmbedder(TextEmbedder):
    """
    Sentence-BERT embedder using sentence-transformers.
    Provides state-of-the-art semantic embeddings.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        """
        Args:
            model_name: Name of sentence-transformers model
                       Popular choices:
                       - 'all-MiniLM-L6-v2' (384 dim, fast)
                       - 'all-mpnet-base-v2' (768 dim, best quality)
                       - 'multi-qa-MiniLM-L6-cos-v1' (384 dim, for Q&A)
            device: 'cpu' or 'cuda'
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: "
                            "pip install sentence-transformers")

        self.model = SentenceTransformer(model_name, device=device)
        vector_dim = self.model.get_sentence_embedding_dimension()
        super().__init__(vector_dim=vector_dim)

        print(f"Loaded Sentence-BERT model: {model_name}")
        print(f"Embedding dimension: {vector_dim}")

    def embed(self, texts: List[str]) -> torch.Tensor:
        """Convert texts to SBERT embeddings."""
        # Encode texts
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        return embeddings


class BioHashText:
    """
    BioHash for text similarity search.

    Combines text embeddings with BioHash for efficient semantic search.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        hash_length: int = 16,
        activity: float = 0.05,
        p: float = 2.0,
        delta: float = 0.0,
        learning_rate: float = 0.02,
        max_epochs: int = 100,
        batch_size: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize BioHash for text.

        Args:
            embedder: TextEmbedder instance for converting text to vectors
            hash_length: Number of active neurons (k)
            activity: Activity level (k/m), typically 0.01-0.10 for text
            Other args: Same as BioHash
        """
        self.embedder = embedder
        self.hash_length = hash_length
        self.activity = activity
        self.device = device

        # Compute number of neurons from activity level
        self.num_neurons = int(hash_length / activity)

        # Initialize BioHash
        self.biohash = BioHash(
            input_dim=embedder.vector_dim,
            num_neurons=self.num_neurons,
            hash_length=hash_length,
            p=p,
            delta=delta,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device
        )

        print(f"BioHashText initialized:")
        print(f"  Embedding dim: {embedder.vector_dim}")
        print(f"  Hash neurons (m): {self.num_neurons}")
        print(f"  Active neurons (k): {hash_length}")
        print(f"  Activity: {activity*100:.1f}%")

    def fit(self, texts: List[str], verbose: bool = True):
        """
        Train BioHash on text data.

        Args:
            texts: List of text strings
            verbose: Whether to print progress
        """
        if verbose:
            print(f"\nEmbedding {len(texts)} texts...")

        # Embed texts
        embeddings = self.embedder.embed(texts).to(self.device)

        if verbose:
            print(f"Embeddings shape: {embeddings.shape}")
            print("\nTraining BioHash...")

        # Train BioHash
        self.biohash.fit(embeddings, verbose=verbose)

    def hash(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Generate hash codes for texts.

        Args:
            texts: Single text string or list of texts

        Returns:
            Hash codes (n × m) with k active neurons per text
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        # Embed texts
        embeddings = self.embedder.embed(texts).to(self.device)

        # Generate hash codes
        hash_codes = self.biohash.hash(embeddings)

        if single:
            return hash_codes[0]

        return hash_codes

    def search(
        self,
        query: str,
        database_texts: List[str],
        database_hashes: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> List[Tuple[int, str, int]]:
        """
        Search for similar texts.

        Args:
            query: Query text
            database_texts: List of database texts
            database_hashes: Pre-computed hash codes (optional)
            top_k: Number of results to return

        Returns:
            List of (rank, text, hamming_distance) tuples
        """
        # Hash query
        query_hash = self.hash(query).unsqueeze(0)

        # Hash database if not provided
        if database_hashes is None:
            database_hashes = self.hash(database_texts)

        # Compute Hamming distances
        distances = torch.cdist(query_hash.float(), database_hashes.float(), p=1) / 2
        distances = distances.squeeze()

        # Get top-k
        top_k_distances, top_k_indices = torch.topk(
            distances, min(top_k, len(database_texts)), largest=False
        )

        # Format results
        results = []
        for rank, (idx, dist) in enumerate(zip(top_k_indices, top_k_distances)):
            results.append((
                rank + 1,
                database_texts[idx.item()],
                int(dist.item())
            ))

        return results

    def save(self, filepath: str):
        """Save model (BioHash weights only, embedder saved separately)."""
        self.biohash.save(filepath)

    def load(self, filepath: str):
        """Load model."""
        self.biohash.load(filepath)


def demo_tfidf():
    """Demo using TF-IDF embeddings."""
    print("="*70)
    print("Demo 1: BioHash + TF-IDF for Text Search")
    print("="*70)

    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret images.",
        "Reinforcement learning trains agents through rewards.",
        "Supervised learning uses labeled training data.",
        "Unsupervised learning finds patterns without labels.",
        "The cat sat on the mat in the sunny room.",
        "Dogs are loyal companions and great pets.",
        "Birds can fly high in the blue sky.",
        "Fish swim in the ocean and rivers.",
        "Python is a popular programming language.",
        "JavaScript runs in web browsers.",
        "Java is used for enterprise applications.",
        "C++ provides low-level memory control.",
    ]

    print(f"\nCorpus: {len(documents)} documents")

    # Create TF-IDF embedder
    embedder = TfidfEmbedder(max_features=1000, ngram_range=(1, 2))
    embedder.fit(documents)

    # Create BioHash for text
    biohash_text = BioHashText(
        embedder=embedder,
        hash_length=8,
        activity=0.10,  # 10% for small corpus
        device='cpu'
    )

    # Train
    biohash_text.fit(documents, verbose=True)

    # Generate hash codes
    print("\nGenerating hash codes for all documents...")
    hash_codes = biohash_text.hash(documents)

    print(f"Hash codes shape: {hash_codes.shape}")
    print(f"Active neurons per document: {(hash_codes == 1).sum(dim=1)[0].item()}")

    # Search examples
    queries = [
        "deep neural networks and AI",
        "pets and animals",
        "programming languages"
    ]

    print("\n" + "="*70)
    print("Search Results")
    print("="*70)

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)

        results = biohash_text.search(query, documents, hash_codes, top_k=3)

        for rank, text, dist in results:
            print(f"{rank}. [Hamming={dist:2d}] {text}")


def demo_sentence_bert():
    """Demo using Sentence-BERT embeddings."""
    print("\n" + "="*70)
    print("Demo 2: BioHash + Sentence-BERT for Semantic Search")
    print("="*70)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Sentence-transformers not installed. Skipping this demo.")
        print("Install with: pip install sentence-transformers")
        return

    # Sample documents (more semantic variety)
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps above an idle hound.",
        "Python is an excellent programming language for beginners.",
        "Learning to code in Python is great for newcomers to programming.",
        "The Eiffel Tower is located in Paris, France.",
        "Paris, the capital of France, is home to the famous Eiffel Tower.",
        "Artificial intelligence is transforming modern technology.",
        "AI is revolutionizing the tech industry.",
        "Climate change poses serious environmental challenges.",
        "Global warming threatens our planet's ecosystems.",
        "The stock market experienced significant volatility today.",
        "Financial markets saw dramatic fluctuations this morning.",
        "Quantum computing could revolutionize cryptography.",
        "The new restaurant serves delicious Italian cuisine.",
        "I enjoy reading science fiction novels in my free time.",
    ]

    print(f"\nCorpus: {len(documents)} documents")

    # Create Sentence-BERT embedder
    embedder = SentenceBERTEmbedder(
        model_name='all-MiniLM-L6-v2',  # Fast, good quality
        device='cpu'
    )

    # Create BioHash for text
    biohash_text = BioHashText(
        embedder=embedder,
        hash_length=16,
        activity=0.05,  # 5% activity
        device='cpu'
    )

    # Train
    biohash_text.fit(documents, verbose=True)

    # Generate hash codes
    print("\nGenerating hash codes for all documents...")
    hash_codes = biohash_text.hash(documents)

    # Search examples (test semantic similarity)
    queries = [
        "A speedy fox jumping",  # Should match "quick brown fox"
        "Learn Python programming",  # Should match Python docs
        "Paris landmarks",  # Should match Eiffel Tower
        "AI technology",  # Should match AI/tech docs
    ]

    print("\n" + "="*70)
    print("Semantic Search Results")
    print("="*70)

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)

        results = biohash_text.search(query, documents, hash_codes, top_k=3)

        for rank, text, dist in results:
            print(f"{rank}. [Hamming={dist:2d}] {text[:70]}...")


def demo_duplicate_detection():
    """Demo: Detect near-duplicate documents."""
    print("\n" + "="*70)
    print("Demo 3: Near-Duplicate Detection")
    print("="*70)

    documents = [
        "Machine learning is a field of computer science.",
        "Machine learning is a field of computer science.",  # Exact duplicate
        "Machine learning is a branch of computer science.",  # Near duplicate
        "Deep learning is a subset of machine learning.",
        "Natural language processing uses machine learning.",
        "Computer vision applies deep learning techniques.",
        "Reinforcement learning trains through trial and error.",
        "Supervised learning requires labeled data.",
        "Unsupervised learning finds hidden patterns.",
        "Machine learning is an area within computer science.",  # Near duplicate of #1
    ]

    print(f"\nCorpus: {len(documents)} documents")

    # Use TF-IDF for speed
    embedder = TfidfEmbedder(max_features=500)
    embedder.fit(documents)

    biohash_text = BioHashText(
        embedder=embedder,
        hash_length=16,
        activity=0.05,
        device='cpu'
    )

    biohash_text.fit(documents, verbose=False)

    # Hash all documents
    hash_codes = biohash_text.hash(documents)

    # Find duplicates (documents with low Hamming distance)
    print("\n" + "-"*70)
    print("Duplicate Detection (Hamming distance threshold = 5)")
    print("-"*70)

    threshold = 5
    found_pairs = set()

    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            dist = (hash_codes[i] != hash_codes[j]).sum().item()

            if dist <= threshold:
                pair_id = tuple(sorted([i, j]))
                if pair_id not in found_pairs:
                    found_pairs.add(pair_id)

                    print(f"\nNear-duplicate found (Hamming distance = {dist}):")
                    print(f"  Doc {i}: {documents[i]}")
                    print(f"  Doc {j}: {documents[j]}")


def demo_news_clustering():
    """Demo: Cluster news articles by topic."""
    print("\n" + "="*70)
    print("Demo 4: News Article Clustering")
    print("="*70)

    # Sample news headlines
    articles = [
        # Tech news
        "Apple announces new iPhone with improved camera",
        "Google releases latest Android update",
        "Microsoft acquires gaming company for billions",
        "Tesla stock rises after earnings report",
        # Sports news
        "Lakers win championship in overtime thriller",
        "Olympic athletes break world records",
        "Soccer team advances to finals",
        "Tennis star retires after long career",
        # Politics news
        "President announces new economic policy",
        "Congress debates infrastructure bill",
        "Election results show close race",
        "International summit addresses climate change",
        # Health news
        "New study shows benefits of exercise",
        "Vaccine research makes breakthrough",
        "Healthy eating habits reduce disease risk",
        "Medical technology improves patient care",
    ]

    labels = ['tech'] * 4 + ['sports'] * 4 + ['politics'] * 4 + ['health'] * 4

    print(f"\nArticles: {len(articles)}")
    print(f"Categories: {set(labels)}")

    # Use TF-IDF
    embedder = TfidfEmbedder(max_features=500)
    embedder.fit(articles)

    biohash_text = BioHashText(
        embedder=embedder,
        hash_length=8,
        activity=0.10,
        device='cpu'
    )

    biohash_text.fit(articles, verbose=False)

    # Hash all articles
    hash_codes = biohash_text.hash(articles)

    # Compute pairwise distances
    distances = torch.cdist(hash_codes.float(), hash_codes.float(), p=1) / 2

    # Compute within-category vs between-category distances
    within_distances = []
    between_distances = []

    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            dist = distances[i, j].item()

            if labels[i] == labels[j]:
                within_distances.append(dist)
            else:
                between_distances.append(dist)

    print(f"\nClustering Quality:")
    print(f"  Avg within-category distance: {np.mean(within_distances):.2f}")
    print(f"  Avg between-category distance: {np.mean(between_distances):.2f}")
    print(f"  Separation ratio: {np.mean(between_distances) / np.mean(within_distances):.2f}")

    # Show example: find similar articles to tech article
    print("\n" + "-"*70)
    print("Example: Articles similar to 'Apple announces new iPhone'")
    print("-"*70)

    query_idx = 0
    query_text = articles[query_idx]

    similar_indices = torch.argsort(distances[query_idx])[:5]

    for rank, idx in enumerate(similar_indices):
        if idx == query_idx:
            continue

        dist = distances[query_idx, idx].item()
        print(f"{rank}. [Hamming={int(dist):2d}] [{labels[idx]:8s}] {articles[idx]}")


if __name__ == '__main__':
    print("\n")
    print("█" * 70)
    print("  BioHash for Text - Demos")
    print("  Bio-Inspired Hashing for Text Similarity Search")
    print("█" * 70)

    # Run demos
    demo_tfidf()

    # Try Sentence-BERT if available
    try:
        demo_sentence_bert()
    except Exception as e:
        print(f"\nSkipping Sentence-BERT demo: {e}")

    demo_duplicate_detection()
    demo_news_clustering()

    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)
