import os
from docuverse.engines.retrieval.chromadb import chromadb_engine
import json

from tqdm import tqdm

from docuverse.engines.retrieval.retrieval_engine import RetrievalEngine
from docuverse.engines.search_corpus import SearchCorpus
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import get_param, _trim_json
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction
from docuverse.utils.timer import timer


class ChromaDBEngine(RetrievalEngine):
    """
    Chromadb_retriever class for document retrieval using the ChromaDB vector database.

    This class implements vector-based retrieval functionality using ChromaDB as the
    underlying storage and search engine.
    """

    def __init__(self, config_params, **kwargs):
        """
        Initialize the ChromaDB retriever with configuration parameters.

        Args:
            config_params: Configuration parameters for the retriever
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_params, **kwargs)
        self.model = None
        self.hidden_dim = None
        self.load_model_config(config_params)
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.text_header = "text" # get_param(self.config.data_header_format, "text_header", "text")
        self.title_header = "title" # get_param(self.config.data_header_format, "title_header", "title")
        self.id_header = "id" # get_param(self.config.data_header_format, "id_header", "id")
        self.extra_fields = get_param(self.config.data_template, "extra_fields", [])
        self.init_model(**kwargs)

        if self.config.ingestion_batch_size == 40:
            self.config.ingestion_batch_size = self.config.bulk_batch

        self.init_client()

    def init_client(self):
        """Initialize the ChromaDB client."""
        persist_directory = os.path.join(get_param(self.config, "project_dir", "/tmp"), "chromadb_data")
        self.client = chromadb.PersistentClient(path=persist_directory)

    def init_model(self, **kwargs):
        """Initialize the ChromaDB embedding model."""
        self.model = DenseEmbeddingFunction(self.config.model_name, attn_implementation=self.config.attn_implementation)
        self.hidden_dim = len(self.model.encode(['text'], show_progress_bar=False)[0])


    def has_index(self, index_name):
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
            print(f"Error checking if index exists: {e}")
            return False

    def create_index(self, index_name: str|None=None, **kwargs):
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
                metadata={"hnsw:space": "cosine"}  # Using cosine similarity by default
            )
            print(f"Created collection: {index_name}")
        except Exception as e:
            print(f"Error creating collection: {e}")

    def load_or_create_index(self):
        if not self.collection:
            if not self.has_index(self.config.index_name):
                self.create_index(self.config.index_name)
            else:
                self.load_index()

    def load_index(self, index_name: str|None=None):
        """
        Loads an existing index from the client and sets it as the current collection.

        This method checks if the specified index exists in the client. If it does not,
        an exception is raised to indicate that the data needs to be ingested first. If
        the index exists, it retrieves the collection associated with the given index
        name and sets it as the current collection.

        Raises:
            ValueError: If the specified index does not exist in the client.

        """
        if index_name is None:
            index_name = self.config.index_name
        if not self.collection:
            if not self.has_index(self.config.index_name):
                raise ValueError(f"Collection {self.config.index_name} does not exist. Please ingest data first.")
            else:
                self.collection = self.client.get_collection(
                    name=self.config.index_name,
                    embedding_function=None
                )

    def delete_index(self, index_name, **kwargs):
        """
        Delete a ChromaDB collection.

        Args:
            index_name: Name of the collection to delete
            **kwargs: Additional parameters
        """
        try:
            self.client.delete_collection(name=index_name)
            print(f"Deleted collection: {index_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")

    def _check_index_creation_and_get_text(self, corpus, update=False):
        self.check_client()
        texts = [row['text'] for row in corpus]
        return texts

    def ingest(self, corpus: SearchCorpus, update: bool = False, **kwargs):
        """Handles the ingestion of a search corpus into a dataset.

        This method ingests a collection of documents, encoding the content and
        storing it into the system in batches for efficiency. The method also
        provides progress tracking and timing for each stage of the ingestion
        process.

        Args:
            corpus (SearchCorpus): The collection of documents to be ingested.
            update (bool): Indicates whether to update or overwrite existing data.
            **kwargs: Additional optional arguments.

        Returns:
            bool: True if the ingestion was successful.
        """
        self.check_client()

        self.load_or_create_index()

        tm = timer("ingest_and_test::ingest")

        tm.add_timing("data_analysis")
        corpus_size = len(corpus)

        tq = tqdm(desc="Creating data", total=corpus_size, leave=True)
        tq1 = tqdm(desc="  * Encoding data", total=corpus_size, leave=False)
        tq2 = tqdm(desc="  * Ingesting data", total=corpus_size, leave=False)

        ingestion_batch = self.ingestion_batch_size

        for i in range(0, corpus_size, ingestion_batch):
            last = min(i+ingestion_batch, corpus_size)
            data = self._create_data(corpus[i:last], tq_instance=tq1)
            tm.add_timing("encoding_data")
            self._insert_data(data, tq_instance=tq2)
            tm.add_timing("data_chroming")
            tq.update(last-i)

        print(f"Ingested {corpus_size} documents into collection {self.config.index_name}")
        return True

    def _create_data(self, corpus, tq_instance=None, **kwargs):
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        for doc_id, doc in enumerate(corpus):
            text = _trim_json(get_param(doc, self.text_header, ""), max_string_len=self.config.max_text_size)
            if not text:
                continue

            # Extract metadata (all fields except text_field)
            metadata = _trim_json({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in doc.items() if k in self.extra_fields or k=="title"})

            # Add document ID
            doc_id_str = get_param(doc, self.id_header, f"doc_{doc_id}")

            documents.append(text)
            metadatas.append(metadata)
            ids.append(doc_id_str)
        embeddings = self.model.encode(documents, show_progress_bar=False, _batch_size=len(documents))
        if tq_instance:
            tq_instance.update(len(documents))
        return documents, embeddings, metadatas, ids

    def _insert_data(self, data, tq_instance):
        self.collection.add(documents=data[0], embeddings=data[1], metadatas=data[2], ids=data[3])
        if tq_instance:
            tq_instance.update(len(data[0]))

    def search(self, query: SearchQueries.Query, **kwargs) -> SearchResult:
        """
        Search for documents matching a single query.

        Args:
            query: Query to search for

        Returns:
            SearchResult: Search result
        """
        self.load_index()

        query_text = query.text

        # Apply filters if specified
        filter_condition = None
        if self.config.filters and self.config.filter_on:
            filter_key = self.config.filter_on
            filter_value = self.config.filters
            filter_condition = {filter_key: filter_value}

        embedding = self.model.encode([query_text], show_progress_bar=False)[0]

        # Perform the search
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=self.config.top_k,
            where=filter_condition
        )

        # Process results
        retrieved_passages = []
        if results and 'documents' in results and results['documents']:
            for idx, doc in enumerate(results['documents'][0]):
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

        return SearchResult(query, retrieved_passages)

    def info(self):
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

