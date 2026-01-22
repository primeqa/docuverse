import os
import json
import logging
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm

from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.engines.search_engine import SearchEngine
from docuverse.utils import get_param, _trim_json
from docuverse.utils.timer import timer
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction


class FAISSEngine(RetrievalEngine):
    """
    FAISS retrieval engine for document retrieval using the FAISS vector database.

    This class implements vector-based retrieval functionality using FAISS (Facebook AI
    Similarity Search) as the underlying indexing and search engine.
    """
    # Class constants
    TEXT_HEADER = "text"
    TITLE_HEADER = "title"
    ID_HEADER = "id"
    DEFAULT_INDEX_TYPE = "Flat"  # Options: "Flat", "IVFFlat", "HNSW"

    def __init__(self, config_params, **kwargs):
        """
        Initialize the FAISS retriever with configuration parameters.

        Args:
            config_params: Configuration parameters for the retriever
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_params, **kwargs)

        # Model attributes
        self.model = None
        self.hidden_dim = None

        # FAISS index attributes
        self.index = None
        self.id_map = []  # Maps FAISS internal IDs to document IDs
        self.metadata_store = {}  # Stores metadata for each document

        # Configuration
        self.load_model_config(config_params)
        self.text_header = self.TEXT_HEADER
        self.title_header = self.TITLE_HEADER
        self.id_header = self.ID_HEADER
        self.extra_fields = get_param(self.config.data_template, "extra_fields", [])
        self.index_type = get_param(config_params, "index_type", self.DEFAULT_INDEX_TYPE)
        self.persist_directory = get_param(self.config, "project_dir", "/tmp")

        # Initialize components
        self.init_model(**kwargs)

        # Handle batch size configuration
        if self.config.ingestion_batch_size == 40:
            self.config.ingestion_batch_size = self.config.bulk_batch

        self.init_client()

    # ===== Initialization Methods =====

    def init_client(self) -> None:
        """Initialize the FAISS index storage directory."""
        os.makedirs(os.path.join(self.persist_directory, "faiss_data"), exist_ok=True)

    def init_model(self, **kwargs) -> None:
        """Initialize the embedding model."""
        self.model = DenseEmbeddingFunction(
            self.config.model_name,
            attn_implementation=self.config.attn_implementation
        )
        self.hidden_dim = len(self.model.encode(['text'], show_progress_bar=False)[0])

    # ===== Index Management Methods =====

    def _get_index_path(self, index_name: str) -> str:
        """Get the file path for storing the FAISS index."""
        return os.path.join(self.persist_directory, "faiss_data", f"{index_name}.index")

    def _get_metadata_path(self, index_name: str) -> str:
        """Get the file path for storing the metadata."""
        return os.path.join(self.persist_directory, "faiss_data", f"{index_name}_metadata.pkl")

    def has_index(self, index_name: str) -> bool:
        """
        Check if an index exists.

        Args:
            index_name: Name of the index to check

        Returns:
            bool: True if the index exists, False otherwise
        """
        index_path = self._get_index_path(index_name)
        metadata_path = self._get_metadata_path(index_name)
        return os.path.exists(index_path) and os.path.exists(metadata_path)

    def check_client(self):
        pass

    def create_index(self, index_name: Optional[str] = None, **kwargs) -> None:
        """
        Create a new FAISS index.

        Args:
            index_name: Name of the index to create
            **kwargs: Additional parameters for index creation
        """
        if index_name is None:
            index_name = self.config.index_name

        try:
            # Create the appropriate FAISS index based on type
            if self.index_type == "Flat":
                # L2 distance (can be changed to METRIC_INNER_PRODUCT for cosine)
                # self.index = faiss.IndexFlatL2(self.hidden_dim)
                self.index = faiss.IndexFlatIP(self.hidden_dim)
            elif self.index_type == "IVFFlat":
                # Inverted file index with flat quantizer
                nlist = get_param(kwargs, "nlist", 100)  # number of clusters
                # quantizer = faiss.IndexFlatL2(self.hidden_dim)
                quantizer = faiss.IndexFlatIP(self.hidden_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.hidden_dim, nlist)
            elif self.index_type == "HNSW":
                # Hierarchical Navigable Small World graph
                m = get_param(kwargs, "hnsw_m", 32)  # number of connections
                self.index = faiss.IndexHNSWFlat(self.hidden_dim, m)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")

            # Initialize metadata storage
            self.id_map = []
            self.metadata_store = {}

            logging.info(f"Created FAISS index: {index_name} (type: {self.index_type})")
        except Exception as e:
            logging.error(f"Error creating FAISS index: {e}")
            raise

    def load_or_create_index(self) -> None:
        """
        Load an existing index or create it if it doesn't exist.
        """
        if self.index is None:
            if not self.has_index(self.config.index_name):
                self.create_index(self.config.index_name)
            else:
                self.load_index()

    def load_index(self, index_name: Optional[str] = None) -> None:
        """
        Load an existing index from disk.

        Args:
            index_name: Name of the index to load

        Raises:
            ValueError: If the specified index does not exist.
        """
        if index_name is None:
            index_name = self.config.index_name

        if self.index is None:
            if not self.has_index(index_name):
                raise ValueError(f"Index {index_name} does not exist. Please ingest data first.")

            try:
                # Load FAISS index
                index_path = self._get_index_path(index_name)
                self.index = faiss.read_index(index_path)

                # Load metadata
                metadata_path = self._get_metadata_path(index_name)
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.id_map = metadata['id_map']
                    self.metadata_store = metadata['metadata_store']

                logging.info(f"Loaded FAISS index: {index_name} with {self.index.ntotal} vectors")
            except Exception as e:
                logging.error(f"Error loading FAISS index: {e}")
                raise

    def save_index(self, index_name: Optional[str] = None) -> None:
        """
        Save the current index to disk.

        Args:
            index_name: Name of the index to save
        """
        if index_name is None:
            index_name = self.config.index_name

        try:
            # Save FAISS index
            index_path = self._get_index_path(index_name)
            faiss.write_index(self.index, index_path)

            # Save metadata
            metadata_path = self._get_metadata_path(index_name)
            metadata = {
                'id_map': self.id_map,
                'metadata_store': self.metadata_store
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            logging.info(f"Saved FAISS index: {index_name}")
        except Exception as e:
            logging.error(f"Error saving FAISS index: {e}")
            raise

    def delete_index(self, index_name: str, **kwargs) -> None:
        """
        Delete a FAISS index.

        Args:
            index_name: Name of the index to delete
            **kwargs: Additional parameters
        """
        try:
            index_path = self._get_index_path(index_name)
            metadata_path = self._get_metadata_path(index_name)

            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            self.index = None
            self.id_map = []
            self.metadata_store = {}

            logging.info(f"Deleted FAISS index: {index_name}")
        except Exception as e:
            logging.error(f"Error deleting FAISS index: {e}")

    # ===== Data Ingestion Methods =====

    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs) -> bool:
        """
        Ingest a search corpus into the FAISS index.

        This method ingests a collection of documents, encoding the content and
        storing it into the FAISS index in batches for efficiency.

        Args:
            corpus: The collection of documents to be ingested
            update: Indicates whether to update or overwrite existing data
            **kwargs: Additional optional arguments

        Returns:
            bool: True if the ingestion was successful
        """
        self.check_client()
        fmt = "\n=== {:30} ==="

        still_create_index = self.create_update_index(fmt=fmt, do_update=update)
        if not still_create_index:
            return None

        self.load_or_create_index()

        tm = timer(f"Faiss::ingest")
        tm.add_timing("data_analysis")

        corpus_size = len(corpus)

        # For IVF indexes, we need to train before adding vectors
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            logging.info("Training IVF index...")
            # Sample vectors for training (use up to 10k for training)
            train_size = min(10000, corpus_size)
            train_corpus = corpus[:train_size]
            train_texts = [
                _trim_json(get_param(doc, self.text_header, ""), max_string_len=self.config.max_text_size)
                for doc in train_corpus
            ]
            train_texts = [t for t in train_texts if t]
            train_embeddings = self.model.encode(train_texts, show_progress_bar=True)
            self.index.train(np.array(train_embeddings).astype('float32'))
            logging.info("Training complete")

        # Setup progress bars
        tq = tqdm(desc="Creating data", total=corpus_size, leave=True)
        tq1 = tqdm(desc="  * Encoding data", total=corpus_size, leave=False)
        tq2 = tqdm(desc="  * Ingesting data", total=corpus_size, leave=False)

        # Process in batches
        for i in range(0, corpus_size, self.ingestion_batch_size):
            last = min(i + self.ingestion_batch_size, corpus_size)
            data = self._create_data(corpus[i:last], tq_instance=tq1)
            tm.add_timing("encoding_data")

            self._insert_data(data, tq_instance=tq2)
            tm.add_timing("data_insertion")

            tq.update(last-i)

        # Save the index after ingestion
        self.save_index()

        logging.info(f"Ingested {corpus_size} documents into FAISS index {self.config.index_name}")
        return True

    def _create_data(self, corpus: List[Dict[str, Any]], tq_instance=None, **kwargs) -> Tuple[List, np.ndarray, List, List]:
        """
        Create data for ingestion from corpus documents.

        Args:
            corpus: List of documents to process
            tq_instance: Optional tqdm instance for progress tracking
            **kwargs: Additional parameters

        Returns:
            Tuple containing documents, embeddings, metadata, and IDs
        """
        documents = []
        metadatas = []
        ids = []

        # Process each document
        for doc_id, doc in enumerate(corpus):
            text = _trim_json(
                get_param(doc, self.text_header, ""),
                max_string_len=self.config.max_text_size
            )

            if not text:
                continue

            # Extract metadata
            metadata = {
                k: (json.dumps(_trim_json(v)) if isinstance(v, dict) else _trim_json(v))
                for k, v in doc.items()
                if k in self.extra_fields or k == "title"
            }
            metadata['text'] = text  # Store the text in metadata for retrieval

            # Add document ID
            doc_id_str = get_param(doc, self.id_header, f"doc_{doc_id}")

            documents.append(text)
            metadatas.append(metadata)
            ids.append(doc_id_str)

        # Generate embeddings
        embeddings = self.model.encode(
            documents,
            show_progress_bar=False,
            _batch_size=len(documents)
        )

        if tq_instance:
            tq_instance.update(len(documents))

        return documents, np.array(embeddings).astype('float32'), metadatas, ids

    def _insert_data(self, data: Tuple[List, np.ndarray, List, List], tq_instance=None) -> None:
        """
        Insert data into the FAISS index.

        Args:
            data: Tuple containing documents, embeddings, metadata, and IDs
            tq_instance: Optional tqdm instance for progress tracking
        """
        documents, embeddings, metadatas, ids = data

        # Add vectors to the index
        start_idx = self.index.ntotal
        self.index.add(embeddings)

        # Update ID mapping and metadata store
        for i, (doc_id, metadata) in enumerate(zip(ids, metadatas)):
            internal_id = start_idx + i
            self.id_map.append(doc_id)
            self.metadata_store[doc_id] = metadata

        if tq_instance:
            tq_instance.update(len(documents))

    # ===== Search Methods =====

    def search(self, query: SearchQueries.Query, **kwargs) -> SearchResult:
        """
        Search for documents matching a single query.

        Args:
            query: Query to search for
            **kwargs: Additional parameters

        Returns:
            SearchResult: Search result containing matched passages
        """
        tm = timer("ingest_and_test::search")
        self.load_index()
        tm.add_timing("load_index")
        query_text = query.text

        # Generate query embedding
        embedding = self.model.encode([query_text], show_progress_bar=False)[0]
        query_vector = np.array([embedding]).astype('float32')
        tm.add_timing("embedding")

        # Perform the search
        k = self.config.top_k
        distances, indices = self.index.search(query_vector, k)
        tm.add_timing("faiss_search")

        # Process results
        retrieved_passages = []
        for idx, (distance, faiss_idx) in enumerate(zip(distances[0], indices[0])):
            if faiss_idx == -1:  # FAISS returns -1 for empty results
                continue

            # Get document ID and metadata
            doc_id = self.id_map[faiss_idx]
            metadata = self.metadata_store.get(doc_id, {})

            # Extract text from metadata
            text = metadata.get('text', '')

            # Convert L2 distance to similarity score (inverse distance)
            # For L2 distance: smaller is better, so we use 1/(1+distance)
            score = 1.0 / (1.0 + distance)

            # Create a passage dict
            passage = {
                "id": doc_id,
                "text": text,
                "score": float(score),
                **{k: v for k, v in metadata.items() if k != 'text'}
            }
            retrieved_passages.append(passage)

        res = SearchResult(query, retrieved_passages)
        tm.add_timing("result_construction")
        return res

    # ===== Utility Methods =====

    def info(self) -> Dict[str, Any]:
        """
        Get information about the retriever.

        Returns:
            dict: Information about the retriever
        """
        info = {
            "retriever_type": "FAISS",
            "index_type": self.index_type,
            "model": getattr(self.config, 'model_name', 'default'),
            "index_name": self.config.index_name,
            "dimension": self.hidden_dim
        }

        if self.index is not None:
            info["document_count"] = self.index.ntotal
            info["is_trained"] = self.index.is_trained if hasattr(self.index, 'is_trained') else True

        return info
