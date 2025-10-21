import os
import json
import threading
import time
import socket
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager
import logging

from pymilvus import MilvusClient, Collection, connections, utility
from pymilvus.exceptions import MilvusException
import requests

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")


@dataclass
class SearchRequest:
    collection_name: str
    query_vector: List[float]
    limit: int = 10
    output_fields: List[str] = None
    search_params: Dict[str, Any] = None


@dataclass
class SearchResponse:
    results: List[Dict[str, Any]]
    status: str = "success"
    error: Optional[str] = None


class MilvusServerInstance:
    """Singleton server instance that manages the Milvus database and serves API requests"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str, host: str = "127.0.0.1", port: int = 8765):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, db_path: str, host: str = "127.0.0.1", port: int = 8765):
        if self._initialized:
            return
            
        self.db_path = db_path
        self.host = host
        self.port = port
        self.client = None
        self.app = None
        self.server_thread = None
        self.running = False
        
        # Ensure Flask is available for server mode
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask is required for server mode. Install with: pip install flask")
            
        self._initialize_client()
        self._setup_flask_app()
        self._initialized = True
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_client(self):
        """Initialize the Milvus client with file-based database"""
        try:
            # Ensure the database directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Initialize file-based Milvus client
            self.client = MilvusClient(uri=self.db_path)
            self.logger.info(f"Milvus client initialized with database at: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Milvus client: {e}")
            raise
    
    def _setup_flask_app(self):
        """Setup Flask application with API endpoints"""
        self.app = Flask(__name__)
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy", "db_path": self.db_path})
        
        @self.app.route('/collections', methods=['GET'])
        def list_collections():
            try:
                collections = self.client.list_collections()
                return jsonify({"collections": collections, "status": "success"})
            except Exception as e:
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route('/collections/<collection_name>/exists', methods=['GET'])
        def collection_exists(collection_name):
            try:
                exists = self.client.has_collection(collection_name)
                return jsonify({"exists": exists, "status": "success"})
            except Exception as e:
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route('/search', methods=['POST'])
        def search():
            try:
                data = request.get_json()
                
                # Validate required fields
                required_fields = ['collection_name', 'query_vector']
                for field in required_fields:
                    if field not in data:
                        return jsonify({"error": f"Missing required field: {field}", "status": "error"}), 400
                
                search_req = SearchRequest(
                    collection_name=data['collection_name'],
                    query_vector=data['query_vector'],
                    limit=data.get('limit', 10),
                    output_fields=data.get('output_fields', ["*"]),
                    search_params=data.get('search_params', {})
                )
                
                # Perform search
                results = self.client.search(
                    collection_name=search_req.collection_name,
                    data=[search_req.query_vector],
                    limit=search_req.limit,
                    output_fields=search_req.output_fields,
                    search_params=search_req.search_params
                )
                
                # Convert results to serializable format
                search_results = []
                for hit in results[0]:
                    hit_dict = {
                        'id': hit.get('id'),
                        'distance': hit.get('distance'),
                        'entity': hit.entity.to_dict() if hasattr(hit, 'entity') else {}
                    }
                    # Add output fields to the result
                    for field in search_req.output_fields:
                        if hasattr(hit, field):
                            hit_dict[field] = getattr(hit, field)
                        elif field in hit:
                            hit_dict[field] = hit[field]
                    search_results.append(hit_dict)
                
                response = SearchResponse(results=search_results)
                return jsonify(response.__dict__)
                
            except Exception as e:
                self.logger.error(f"Search error: {e}")
                response = SearchResponse(results=[], status="error", error=str(e))
                return jsonify(response.__dict__), 500
        
        @self.app.route('/collections/<collection_name>/stats', methods=['GET'])
        def collection_stats(collection_name):
            try:
                stats = self.client.get_collection_stats(collection_name)
                return jsonify({"stats": stats, "status": "success"})
            except Exception as e:
                return jsonify({"error": str(e), "status": "error"}), 500
    
    def start_server(self, threaded: bool = True):
        """Start the Flask server"""
        if self.running:
            self.logger.info(f"Server already running on {self.host}:{self.port}")
            return
        
        def run_server():
            try:
                self.logger.info(f"Starting Milvus server on {self.host}:{self.port}")
                self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
            except Exception as e:
                self.logger.error(f"Server error: {e}")
        
        if threaded:
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            # Wait a bit for the server to start
            time.sleep(2)
            
            # Check if server is actually running
            if self._is_server_running():
                self.running = True
                self.logger.info(f"Server successfully started on {self.host}:{self.port}")
            else:
                raise RuntimeError("Failed to start server")
        else:
            run_server()
    
    def _is_server_running(self) -> bool:
        """Check if the server is running"""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def stop_server(self):
        """Stop the server (Note: Flask doesn't have a built-in way to stop gracefully)"""
        self.running = False
        self.logger.info("Server stop requested")


class MilvusFileServer:
    """Main Milvus Server class that handles both server and client functionality"""
    
    def __init__(self, db_path: str, host: str = "127.0.0.1", port: int = 8765, 
                 use_api: bool = None, start_server: bool = True):
        """
        Initialize MilvusServer
        
        Args:
            db_path: Path to the file-based Milvus database
            host: Host address for API server
            port: Port for API server
            use_api: If True, use API client. If False, use direct file access. If None, auto-detect
            start_server: Whether to start the API server automatically
        """
        self.db_path = db_path
        self.host = host
        self.port = port
        self.use_api = use_api
        self.server_instance = None
        self.client = None
        self.api_client = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect mode if not specified
        if self.use_api is None:
            self.use_api = self._should_use_api()
        
        if self.use_api:
            self._initialize_api_client(start_server)
        else:
            self._initialize_direct_client()
    
    def _should_use_api(self) -> bool:
        """Determine whether to use API mode based on server availability"""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _initialize_direct_client(self):
        """Initialize direct file-based Milvus client"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.client = MilvusClient(uri=self.db_path)
            self.logger.info(f"Direct Milvus client initialized: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize direct client: {e}")
            raise
    
    def _initialize_api_client(self, start_server: bool = True):
        """Initialize API client, starting server if needed"""
        # Check if server is already running
        if not self._is_server_running():
            if start_server:
                self.logger.info("Server not running, starting new instance...")
                self.server_instance = MilvusServerInstance(self.db_path, self.host, self.port)
                self.server_instance.start_server(threaded=True)
            else:
                raise RuntimeError(f"No server running on {self.host}:{self.port} and start_server=False")
        else:
            self.logger.info(f"Using existing server on {self.host}:{self.port}")
        
        self.api_client = MilvusAPIClient(self.host, self.port)
    
    def _is_server_running(self) -> bool:
        """Check if the API server is running"""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        if self.use_api:
            return self.api_client.has_collection(collection_name)
        else:
            return self.client.has_collection(collection_name)
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        if self.use_api:
            return self.api_client.list_collections()
        else:
            return self.client.list_collections()
    
    def search(self, collection_name: str, query_vector: List[float], 
               limit: int = 10, output_fields: List[str] = None, 
               search_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform vector search
        
        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            output_fields: Fields to include in results
            search_params: Additional search parameters
            
        Returns:
            List of search results
        """
        if output_fields is None:
            output_fields = ["*"]
        if search_params is None:
            search_params = {}
            
        if self.use_api:
            return self.api_client.search(collection_name, query_vector, limit, 
                                        output_fields, search_params)
        else:
            try:
                results = self.client.search(
                    collection_name=collection_name,
                    data=[query_vector],
                    limit=limit,
                    output_fields=output_fields,
                    search_params=search_params
                )
                
                # Convert to consistent format
                search_results = []
                for hit in results[0]:
                    hit_dict = {
                        'id': hit.get('id'),
                        'distance': hit.get('distance'),
                    }
                    # Add output fields
                    for field in output_fields:
                        if field in hit:
                            hit_dict[field] = hit[field]
                    search_results.append(hit_dict)
                
                return search_results
                
            except Exception as e:
                self.logger.error(f"Search error: {e}")
                raise
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics"""
        if self.use_api:
            return self.api_client.get_collection_stats(collection_name)
        else:
            return self.client.get_collection_stats(collection_name)
    
    def close(self):
        """Close the client connection"""
        if self.client:
            # MilvusClient doesn't have explicit close method
            self.client = None
        if self.server_instance:
            self.server_instance.stop_server()


class MilvusAPIClient:
    """API client for communicating with MilvusServer"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.base_url = f"http://{host}:{port}"
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request to the API server"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise RuntimeError(f"API request failed: {e}")
    
    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists"""
        response = self._make_request('GET', f'/collections/{collection_name}/exists')
        return response.get('exists', False)
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        response = self._make_request('GET', '/collections')
        return response.get('collections', [])
    
    def search(self, collection_name: str, query_vector: List[float], 
               limit: int = 10, output_fields: List[str] = None,
               search_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform vector search via API"""
        data = {
            'collection_name': collection_name,
            'query_vector': query_vector,
            'limit': limit,
            'output_fields': output_fields or ["*"],
            'search_params': search_params or {}
        }
        
        response = self._make_request('POST', '/search', data)
        
        if response.get('status') == 'error':
            raise RuntimeError(response.get('error', 'Unknown search error'))
        
        return response.get('results', [])
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics"""
        response = self._make_request('GET', f'/collections/{collection_name}/stats')
        return response.get('stats', {})


# Convenience functions
def create_milvus_server(db_path: str, host: str = "127.0.0.1", port: int = 8765, 
                        use_api: bool = None, start_server: bool = True) -> MilvusFileServer:
    """
    Create a MilvusServer instance
    
    Args:
        db_path: Path to the Milvus database file
        host: API server host
        port: API server port  
        use_api: Whether to use API mode (auto-detect if None)
        start_server: Whether to start server automatically
        
    Returns:
        MilvusServer instance
    """
    return MilvusFileServer(db_path, host, port, use_api, start_server)


# Example usage and testing
if __name__ == "__main__":
    # Example: First instance (will start server)
    print("Creating first instance (server mode)...")
    server1 = create_milvus_server("./test_db/milvus.db", use_api=False, start_server=True)
    
    print("Collections:", server1.list_collections())
    
    # Example: Second instance (will use API)
    print("\nCreating second instance (client mode)...")  
    server2 = create_milvus_server("./test_db/milvus.db", use_api=True, start_server=False)
    
    print("Collections via API:", server2.list_collections())
    
    # Cleanup
    server1.close()
    server2.close()
