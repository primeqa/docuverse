import os
import json
import logging
import chromadb
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm

from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import get_param, _trim_json
from docuverse.utils.timer import timer
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction


class ChromaDBEngine(RetrievalEngine):
    """
    ChromaDB retrieval engine for document retrieval using the ChromaDB vector database.
    
    This class implements vector-based retrieval functionality using ChromaDB as the
    underlying storage and search engine.
    """
    # Class constants
    TEXT_HEADER = "text"
    TITLE_HEADER = "title"
    ID_HEADER = "id"
    DEFAULT_SIMILARITY = "cosine"
    
    def __init__(self, config_params, **kwargs):
        """
        Initialize the ChromaDB retriever with configuration parameters.
        
        Args:
            config_params: Configuration parameters for the retriever
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_params, **kwargs)
        
        # Model attributes
        self.model = None
        self.hidden_dim = None
        
        # Client attributes
        self.client = None
        self.collection = None
        self.embedding_function = None
        
        # Configuration
        self.load_model_config(config_params)
        self.text_header = self.TEXT_HEADER
        self.title_header = self.TITLE_HEADER
        self.id_header = self.ID_HEADER
        self.extra_fields = get_param(self.config.data_template, "extra_fields", [])
        
        # Initialize components
        self.init_model(**kwargs)
        
        # Handle batch size configuration
        if self.config.ingestion_batch_size == 40:
            self.config.ingestion_batch_size = self.config.bulk_batch
            
        self.init_client()

    # ===== Initialization Methods =====
    
    def init_client(self) -> None:
        """Initialize the ChromaDB client with persistence."""
        persist_directory = os.path.join(get_param(self.config, "project_dir", "/tmp"), "chromadb_data")
        self.client = chromadb.PersistentClient(path=persist_directory)

    def init_model(self, **kwargs) -> None:
        """Initialize the ChromaDB embedding model."""
        self.model = DenseEmbeddingFunction(
            self.config.model_name, 
            attn_implementation=self.config.attn_implementation
        )
        self.hidden_dim = len(self.model.encode(['text'], show_progress_bar=False)[0])

    # ===== Index Management Methods =====
    
    def has_index(self, index_name: str) -> bool:
        """
        Check if an index exists.
        
        Args:
            index_name: Name of the index to check
            
        Returns:
            bool: True if the index exists, False otherwise
        """
        try:
            collections = self.client.list_collections()
            return any(collection.name == index_name for collection in collections)
        except Exception as e:
            logging.error(f"Error checking if index exists: {e}")
            return False

    def create_index(self, index_name: Optional[str] = None, **kwargs) -> None:
        """
        Create a new ChromaDB collection.
        
        Args:
            index_name: Name of the collection to create
            **kwargs: Additional parameters for collection creation
        """
        if index_name is None:
            index_name = self.config.index_name
            
        try:
            self.collection = self.client.create_collection(
                name=index_name,
                embedding_function=None,
                metadata={"hnsw:space": self.DEFAULT_SIMILARITY}
            )
            logging.info(f"Created collection: {index_name}")
        except Exception as e:
            logging.error(f"Error creating collection: {e}")

    def load_or_create_index(self) -> None:
        """
        Load an existing index or create it if it doesn't exist.
        """
        if not self.collection:
            if not self.has_index(self.config.index_name):
                self.create_index(self.config.index_name)
            else:
                self.load_index()

    def load_index(self, index_name: Optional[str] = None) -> None:
        """
        Loads an existing index from the client and sets it as the current collection.
        
        This method checks if the specified index exists in the client. If it does not,
        an exception is raised to indicate that the data needs to be ingested first.
        
        Args:
            index_name: Name of the index to load
            
        Raises:
            ValueError: If the specified index does not exist in the client.
        """
        if index_name is None:
            index_name = self.config.index_name
            
        if not self.collection:
            if not self.has_index(index_name):
                raise ValueError(f"Collection {index_name} does not exist. Please ingest data first.")
            else:
                self.collection = self.client.get_collection(
                    name=index_name,
                    embedding_function=None
                )

    def delete_index(self, index_name: str, **kwargs) -> None:
        """
        Delete a ChromaDB collection.
        
        Args:
            index_name: Name of the collection to delete
            **kwargs: Additional parameters
        """
        try:
            self.client.delete_collection(name=index_name)
            logging.info(f"Deleted collection: {index_name}")
        except Exception as e:
            logging.error(f"Error deleting collection: {e}")

    # ===== Data Ingestion Methods =====
    
    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs) -> bool:
        """
        Handles the ingestion of a search corpus into a dataset.
        
        This method ingests a collection of documents, encoding the content and 
        storing it into the system in batches for efficiency. The method also 
        provides progress tracking and timing for each stage of the ingestion 
        process.
        
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

        tm = timer("ingest_and_test::ingest")
        tm.add_timing("data_analysis")
        
        corpus_size = len(corpus)
        
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
            tm.add_timing("data_chroming")
            
            tq.update(last-i)

        logging.info(f"Ingested {corpus_size} documents into collection {self.config.index_name}")
        return True

    def _create_data(self, corpus: List[Dict[str, Any]], tq_instance=None, **kwargs) -> Tuple[List, List, List, List]:
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
            
        return documents, embeddings, metadatas, ids

    def _insert_data(self, data: Tuple[List, List, List, List], tq_instance=None) -> None:
        """
        Insert data into the ChromaDB collection.
        
        Args:
            data: Tuple containing documents, embeddings, metadata, and IDs
            tq_instance: Optional tqdm instance for progress tracking
        """
        self.collection.upsert(
            documents=data[0], 
            embeddings=data[1], 
            metadatas=data[2], 
            ids=data[3]
        )
        
        if tq_instance:
            tq_instance.update(len(data[0]))

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
        
        # Apply filters if specified
        filter_condition = None
        if self.config.filters and self.config.filter_on:
            filter_key = self.config.filter_on
            filter_value = self.config.filters
            filter_condition = {filter_key: filter_value}
        tm.add_timing("filter_condition")
        # Generate query embedding
        embedding = self.model.encode([query_text], show_progress_bar=False)[0]
        tm.add_timing("embedding")
        
        # Perform the search
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=20*self.config.top_k,  # We need to extend the top_k for chromadb, or will give suboptimal results.
            where=filter_condition
        )
        tm.add_timing("chromadb_search")
        # Process results
        retrieved_passages = []
        if results and 'documents' in results and results['documents']:
            for idx, doc in enumerate(results['documents'][0][:self.config.top_k]):
                metadata = results['metadatas'][0][idx] if 'metadatas' in results and results['metadatas'] else {}
                doc_id = results['ids'][0][idx] if 'ids' in results and results['ids'] else f"doc_{idx}"
                distance = results['distances'][0][idx] if 'distances' in results and results['distances'] else 0.0
                
                # Create a passage dict
                passage = {
                    "id": doc_id,
                    "text": doc,
                    "score": 1.0 - distance,  # Convert distance to similarity score
                    **metadata
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
            "retriever_type": "ChromaDB",
            "model": getattr(self.config, 'model_name', 'default'),
            "index_name": self.config.index_name
        }
        
        if self.collection:
            try:
                count = self.collection.count()
                info["document_count"] = count
            except Exception:
                pass
                
        return info
