"""
BioHash integration with DocUVerse framework.

Hash documents from DocUVerse corpora for fast similarity search.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import pickle
import json
from pathlib import Path

from biohash_text import BioHashText, TfidfEmbedder, SentenceBERTEmbedder, AverageWordEmbedder


class DocUVerseBioHash:
    """
    BioHash for DocUVerse document search.

    Integrates BioHash with DocUVerse corpora for efficient document retrieval.
    """

    def __init__(
        self,
        embedding_type: str = 'tfidf',
        embedding_config: Optional[Dict] = None,
        hash_length: int = 32,
        activity: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize DocUVerse BioHash.

        Args:
            embedding_type: Type of embedder ('tfidf', 'sbert', 'word2vec')
            embedding_config: Config dict for embedder
            hash_length: Number of active hash neurons
            activity: Activity level (k/m)
            device: 'cpu' or 'cuda'
        """
        self.embedding_type = embedding_type
        self.hash_length = hash_length
        self.activity = activity
        self.device = device

        # Initialize embedder
        if embedding_config is None:
            embedding_config = {}

        self.embedder = self._create_embedder(embedding_type, embedding_config)
        self.biohash_text = None

        # Document storage
        self.documents = []
        self.doc_ids = []
        self.metadata = []
        self.hash_codes = None

    def _create_embedder(self, embedding_type: str, config: Dict):
        """Create embedder based on type."""
        if embedding_type == 'tfidf':
            return TfidfEmbedder(
                max_features=config.get('max_features', 5000),
                ngram_range=config.get('ngram_range', (1, 2))
            )

        elif embedding_type == 'sbert':
            return SentenceBERTEmbedder(
                model_name=config.get('model_name', 'all-MiniLM-L6-v2'),
                device=self.device
            )

        elif embedding_type == 'word2vec':
            return AverageWordEmbedder(
                embedding_path=config.get('embedding_path'),
                vector_dim=config.get('vector_dim', 300)
            )

        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

    def load_from_tsv(
        self,
        tsv_path: str,
        text_column: str = 'text',
        id_column: str = 'doc_id',
        title_column: Optional[str] = 'title',
        max_docs: Optional[int] = None
    ):
        """
        Load documents from TSV file (DocUVerse format).

        Args:
            tsv_path: Path to TSV file
            text_column: Column name for document text
            id_column: Column name for document ID
            title_column: Column name for title (optional)
            max_docs: Maximum number of documents to load
        """
        import csv

        print(f"Loading documents from {tsv_path}...")

        self.documents = []
        self.doc_ids = []
        self.metadata = []

        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for i, row in enumerate(reader):
                if max_docs and i >= max_docs:
                    break

                # Get text (combine title and text if available)
                text = row.get(text_column, '')

                if title_column and title_column in row:
                    title = row.get(title_column, '')
                    if title:
                        text = f"{title}. {text}"

                self.documents.append(text)
                self.doc_ids.append(row.get(id_column, str(i)))

                # Store metadata
                self.metadata.append({
                    'title': row.get(title_column, ''),
                    'doc_id': row.get(id_column, str(i))
                })

                if (i + 1) % 1000 == 0:
                    print(f"  Loaded {i + 1} documents...")

        print(f"Loaded {len(self.documents)} documents")

    def load_from_jsonl(
        self,
        jsonl_path: str,
        text_field: str = 'text',
        id_field: str = 'id',
        title_field: Optional[str] = 'title',
        max_docs: Optional[int] = None
    ):
        """Load documents from JSONL file."""
        print(f"Loading documents from {jsonl_path}...")

        self.documents = []
        self.doc_ids = []
        self.metadata = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_docs and i >= max_docs:
                    break

                doc = json.loads(line)

                # Get text
                text = doc.get(text_field, '')

                if title_field and title_field in doc:
                    title = doc.get(title_field, '')
                    if title:
                        text = f"{title}. {text}"

                self.documents.append(text)
                self.doc_ids.append(doc.get(id_field, str(i)))
                self.metadata.append(doc)

                if (i + 1) % 1000 == 0:
                    print(f"  Loaded {i + 1} documents...")

        print(f"Loaded {len(self.documents)} documents")

    def build_index(self, verbose: bool = True):
        """Build BioHash index for loaded documents."""
        if len(self.documents) == 0:
            raise ValueError("No documents loaded. Call load_from_tsv() or load_from_jsonl() first.")

        print("\nBuilding BioHash index...")

        # Fit embedder if needed (for TF-IDF)
        if self.embedding_type == 'tfidf':
            self.embedder.fit(self.documents)

        # Create BioHashText
        self.biohash_text = BioHashText(
            embedder=self.embedder,
            hash_length=self.hash_length,
            activity=self.activity,
            device=self.device
        )

        # Train BioHash
        self.biohash_text.fit(self.documents, verbose=verbose)

        # Generate hash codes for all documents
        print("\nGenerating hash codes for all documents...")
        self.hash_codes = self.biohash_text.hash(self.documents)

        print(f"Index built: {len(self.documents)} documents, "
              f"{self.hash_codes.shape[1]} hash neurons")

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search for documents similar to query.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of result dicts with keys: rank, doc_id, text, score, metadata
        """
        if self.biohash_text is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Hash query
        query_hash = self.biohash_text.hash(query).unsqueeze(0)

        # Compute Hamming distances
        distances = torch.cdist(query_hash.float(), self.hash_codes.float(), p=1) / 2
        distances = distances.squeeze()

        # Get top-k
        top_k_distances, top_k_indices = torch.topk(
            distances, min(top_k, len(self.documents)), largest=False
        )

        # Format results
        results = []
        for rank, (idx, dist) in enumerate(zip(top_k_indices, top_k_distances)):
            idx = idx.item()
            results.append({
                'rank': rank + 1,
                'doc_id': self.doc_ids[idx],
                'text': self.documents[idx],
                'hamming_distance': int(dist.item()),
                'metadata': self.metadata[idx]
            })

        return results

    def save_index(self, save_dir: str):
        """Save BioHash index to directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving index to {save_dir}...")

        # Save BioHash model
        self.biohash_text.save(save_dir / 'biohash_model.pt')

        # Save hash codes
        torch.save(self.hash_codes, save_dir / 'hash_codes.pt')

        # Save documents and metadata
        with open(save_dir / 'documents.pkl', 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_ids': self.doc_ids,
                'metadata': self.metadata
            }, f)

        # Save embedder (if TF-IDF)
        if self.embedding_type == 'tfidf':
            with open(save_dir / 'embedder.pkl', 'wb') as f:
                pickle.dump(self.embedder, f)

        # Save config
        config = {
            'embedding_type': self.embedding_type,
            'hash_length': self.hash_length,
            'activity': self.activity,
            'num_documents': len(self.documents)
        }

        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print("Index saved successfully")

    def load_index(self, load_dir: str):
        """Load BioHash index from directory."""
        load_dir = Path(load_dir)

        print(f"Loading index from {load_dir}...")

        # Load config
        with open(load_dir / 'config.json', 'r') as f:
            config = json.load(f)

        self.embedding_type = config['embedding_type']
        self.hash_length = config['hash_length']
        self.activity = config['activity']

        # Load embedder
        if self.embedding_type == 'tfidf':
            with open(load_dir / 'embedder.pkl', 'rb') as f:
                self.embedder = pickle.load(f)
        else:
            # Re-create embedder (for SBERT, etc.)
            self.embedder = self._create_embedder(self.embedding_type, {})

        # Load documents and metadata
        with open(load_dir / 'documents.pkl', 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.doc_ids = data['doc_ids']
            self.metadata = data['metadata']

        # Load hash codes
        self.hash_codes = torch.load(load_dir / 'hash_codes.pt')

        # Create BioHashText and load model
        self.biohash_text = BioHashText(
            embedder=self.embedder,
            hash_length=self.hash_length,
            activity=self.activity,
            device=self.device
        )

        self.biohash_text.load(load_dir / 'biohash_model.pt')

        print(f"Index loaded: {len(self.documents)} documents")


def demo_docuverse_integration():
    """Demo: Use BioHash with DocUVerse-style data."""
    print("="*70)
    print("Demo: BioHash + DocUVerse Integration")
    print("="*70)

    # Create sample corpus in DocUVerse TSV format
    sample_corpus = """doc_id\ttitle\ttext
1\tMachine Learning Basics\tMachine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.
2\tDeep Learning Introduction\tDeep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.
3\tNatural Language Processing\tNatural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.
4\tComputer Vision Overview\tComputer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.
5\tReinforcement Learning Guide\tReinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms.
6\tPython Programming\tPython is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.
7\tJavaScript Development\tJavaScript, often abbreviated as JS, is a programming language that conforms to the ECMAScript specification. JavaScript is high-level, often just-in-time compiled, and multi-paradigm.
8\tData Science Career\tData science is an inter-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data, and apply knowledge and actionable insights from data across a broad range of application domains.
9\tCloud Computing Basics\tCloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user. The term is generally used to describe data centers available to many users over the Internet.
10\tCybersecurity Fundamentals\tCybersecurity is the practice of protecting systems, networks, and programs from digital attacks. These cyberattacks are usually aimed at accessing, changing, or destroying sensitive information; extorting money from users; or interrupting normal business processes.
"""

    # Save to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(sample_corpus)
        corpus_path = f.name

    try:
        # Create DocUVerse BioHash
        doc_hash = DocUVerseBioHash(
            embedding_type='tfidf',
            embedding_config={'max_features': 1000, 'ngram_range': (1, 2)},
            hash_length=16,
            activity=0.05,
            device='cpu'
        )

        # Load documents
        doc_hash.load_from_tsv(corpus_path)

        # Build index
        doc_hash.build_index(verbose=True)

        # Search examples
        queries = [
            "neural networks and deep learning",
            "programming languages",
            "protecting computer systems"
        ]

        print("\n" + "="*70)
        print("Search Results")
        print("="*70)

        for query in queries:
            print(f"\nQuery: '{query}'")
            print("-" * 70)

            results = doc_hash.search(query, top_k=3)

            for result in results:
                print(f"\n{result['rank']}. [Hamming={result['hamming_distance']:2d}] "
                      f"{result['metadata']['title']}")
                print(f"   {result['text'][:100]}...")

        # Save index
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\n\nTesting save/load...")
            doc_hash.save_index(tmpdir)

            # Load index
            doc_hash_loaded = DocUVerseBioHash(device='cpu')
            doc_hash_loaded.load_index(tmpdir)

            print("Index reloaded successfully!")

            # Test search on loaded index
            results = doc_hash_loaded.search("machine learning", top_k=2)
            print(f"\nTest search on loaded index:")
            for result in results:
                print(f"  {result['rank']}. {result['metadata']['title']}")

    finally:
        # Clean up temp file
        import os
        os.unlink(corpus_path)


if __name__ == '__main__':
    demo_docuverse_integration()

    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)
    print("\nUsage example:")
    print("""
# Create index
doc_hash = DocUVerseBioHash(embedding_type='tfidf')
doc_hash.load_from_tsv('path/to/corpus.tsv')
doc_hash.build_index()

# Search
results = doc_hash.search("your query here", top_k=10)
for r in results:
    print(f"{r['rank']}. {r['metadata']['title']}")

# Save index
doc_hash.save_index('./my_index')

# Load later
doc_hash = DocUVerseBioHash()
doc_hash.load_index('./my_index')
""")
