import os
import json
import threading
import time
import socket
import hashlib
import fcntl
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager
import logging
from pathlib import Path

import numpy
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
    output_fields: Optional[List[str]] = None
    search_params: Optional[Dict[str, Any]] = None


@dataclass
class SearchResponse:
    results: List[Dict[str, Any]]
    status: str = "success"
    error: Optional[str] = None


class Utils:
    @staticmethod
    def extract_hits(results, output_fields: List[str] = None) -> List[Dict]:
        search_results = []
        for hit in results:
            hit_dict = {
                'id': hit.get('id'),
                'distance': hit.get('distance')
            }
            # Add output fields to the result
            if output_fields:
                vals = hit['entity']
                for field in output_fields:
                    if field in vals:
                        hit_dict[field] = vals[field]
            search_results.append(hit_dict)

        return search_results


@dataclass
class ServerInfo:
    """Information about a running server"""
    host: str
    port: int
    pid: int
    db_path: str
    start_time: float
    machine_id: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'port': self.port,
            'pid': self.pid,
            'db_path': self.db_path,
            'start_time': self.start_time,
            'machine_id': self.machine_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServerInfo':
        return cls(**data)


class ServerRegistry:
    """Manages server registration and discovery across machines"""
    
    def __init__(self, db_path: str, registry_dir: str = None):
        self.db_path = os.path.abspath(db_path)
        
        # Create a unique identifier for this database
        # self.db_hash = hashlib.md5(self.db_path.encode()).hexdigest()[:8]
        self.db_hash = hashlib.sha256(self.db_path.encode()).hexdigest()[:16]

        # Default registry directory
        if registry_dir is None:
            registry_dir = os.path.join(os.path.dirname(self.db_path), '.milvus_servers')
        
        self.registry_dir = os.path.abspath(registry_dir)
        os.makedirs(self.registry_dir, exist_ok=True)
        
        # Registry file for this specific database
        self.registry_file = os.path.join(self.registry_dir, f"server_{self.db_hash}.json")
        self.lock_file = f"{self.registry_file}.lock"
        
        self.logger = logging.getLogger(__name__)
    
    def _get_machine_id(self) -> str:
        """Get a unique identifier for this machine"""
        try:
            # Try to get machine ID from various sources
            if os.path.exists('/etc/machine-id'):
                with open('/etc/machine-id', 'r') as f:
                    return f.read().strip()
            elif os.path.exists('/var/lib/dbus/machine-id'):
                with open('/var/lib/dbus/machine-id', 'r') as f:
                    return f.read().strip()
            else:
                # Fallback: use hostname + MAC address
                import uuid
                mac = hex(uuid.getnode())
                hostname = socket.gethostname()
                return hashlib.md5(f"{hostname}:{mac}".encode()).hexdigest()
        except:
            # Last resort: just use hostname
            return socket.gethostname()
    
    @contextmanager
    def _file_lock(self):
        """File-based locking for cross-process synchronization"""
        lock_fd = None
        try:
            lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except OSError as e:  # Be more specific
                    self.logger.warning(f"Failed to release file lock: {e}")
    
    def register_server(self, host: str, port: int, pid: int) -> ServerInfo:
        """Register a server instance"""
        server_info = ServerInfo(
            host=host,
            port=port,
            pid=pid,
            db_path=self.db_path,
            start_time=time.time(),
            machine_id=self._get_machine_id()
        )
        
        with self._file_lock():
            with open(self.registry_file, 'w') as f:
                json.dump(server_info.to_dict(), f, indent=2)
        
        self.logger.info(f"Registered server {host}:{port} for database {self.db_path}")
        return server_info
    
    def get_server_info(self) -> Optional[ServerInfo]:
        """Get information about the registered server"""
        if not os.path.exists(self.registry_file):
            return None
        
        try:
            with self._file_lock():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                return ServerInfo.from_dict(data)
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            # Registry file is corrupted or missing, clean it up
            self._cleanup_registry()
            return None
    
    def is_server_alive(self, server_info: ServerInfo) -> bool:
        """Check if a registered server is still alive"""
        if server_info is None:
            return False
        
        try:
            # First, check if it's on the same machine and process is alive
            if server_info.machine_id == self._get_machine_id():
                try:
                    os.kill(server_info.pid, 0)  # Check if process exists
                except (OSError, ProcessLookupError):
                    return False
            
            # Then check if the server responds to HTTP requests
            response = requests.get(
                f"http://{server_info.host}:{server_info.port}/health", 
                timeout=5
            )
            return response.status_code == 200
            
        except (requests.RequestException, requests.Timeout, ConnectionError) as e:
            self.logger.debug(f"Server health check failed: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"Unexpected error in server health check: {e}")
            return False
    
    def cleanup_if_dead(self) -> bool:
        """Clean up registry if server is dead, returns True if cleaned up"""
        server_info = self.get_server_info()
        if server_info and not self.is_server_alive(server_info):
            self._cleanup_registry()
            return True
        return False
    
    def _cleanup_registry(self):
        """Remove the registry file"""
        try:
            if os.path.exists(self.registry_file):
                os.remove(self.registry_file)
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except:
            pass
    
    def unregister_server(self):
        """Unregister the current server"""
        self._cleanup_registry()


class MilvusServerInstance:
    """Singleton server instance that manages the Milvus database and serves API requests"""
    
    _instances = {}  # One instance per database path
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str, host: str = "0.0.0.0", port: int = 8765):
        db_path = os.path.abspath(db_path)
        
        with cls._lock:
            if db_path not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[db_path] = instance
            return cls._instances[db_path]
    
    def __init__(self, db_path: str, host: str = "0.0.0.0", port: int = 8765):
        # Add double-check locking for thread safety
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        with self.__class__._lock:  # Add lock here too
            if hasattr(self, '_initialized') and self._initialized:
                return
            
            self.db_path = os.path.abspath(db_path)
            self.host = host
            self.port = port
            self.client = None
            self.app = None
            self.server_thread = None
            self.running = False
            logging.getLogger('werkzeug').setLevel(logging.ERROR)

        logging.basicConfig(level=logging.WARN)
        self.logger = logging.getLogger(__name__)

        # Server registry for cross-machine discovery
        self.registry = ServerRegistry(self.db_path)
        
        # Ensure Flask is available for server mode
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask is required for server mode. Install with: pip install flask")
            
        self._initialize_client()
        self._setup_flask_app()
        self._initialized = True

    def _get_external_ip(self) -> str:
        """Get the external IP address of this machine"""
        if self.host != "0.0.0.0":
            return self.host
            
        try:
            # Try to get external IP by connecting to a remote service
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"
    
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
        self.app.logger.setLevel(logging.ERROR)

        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                "status": "healthy", 
                "db_path": self.db_path,
                "host": self._get_external_ip(),
                "port": self.port,
                "pid": os.getpid()
            })
        
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
                search_results = Utils.extract_hits(results[0], search_req.output_fields)
                
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
        """Start the Flask server and register it"""
        if self.running:
            self.logger.warning(f"Server already running on {self.host}:{self.port}")
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
            
            # Check if server is actually running and register it
            if self._is_server_running():
                # Register the server with external IP
                external_ip = self._get_external_ip()
                self.registry.register_server(external_ip, self.port, os.getpid())
                self.running = True
                self.logger.info(f"Server successfully started on {external_ip}:{self.port}")
            else:
                raise RuntimeError("Failed to start server")
        else:
            run_server()
    
    def _is_server_running(self) -> bool:
        """Check if the server is running"""
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def stop_server(self):
        """Stop the server and unregister it"""
        self.running = False
        self.registry.unregister_server()
        self.logger.info("Server stop requested and unregistered")


class MilvusServer:
    """Main Milvus Server class that handles both server and client functionality"""
    
    def __init__(self, db_path: str, host: str = "0.0.0.0", port: int = 8765, 
                 use_api: bool = None, start_server: bool = True, 
                 registry_dir: str = None):
        """
        Initialize MilvusServer with cross-machine support
        
        Args:
            db_path: Path to the file-based Milvus database
            host: Host address for API server (use "0.0.0.0" to bind to all interfaces)
            port: Port for API server
            use_api: If True, use API client. If False, use direct file access. If None, auto-detect
            start_server: Whether to start the API server automatically
            registry_dir: Directory for server registry files (defaults to {db_dir}/.milvus_servers)
        """
        self.db_path = os.path.abspath(db_path)
        self.host = host
        self.port = port
        self.use_api = use_api
        self.server_instance = None
        self.client = None
        self.api_client = None
        
        # Server registry for cross-machine discovery
        self.registry = ServerRegistry(self.db_path, registry_dir)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Clean up dead servers first
        self.registry.cleanup_if_dead()
        
        # Auto-detect mode if not specified
        if self.use_api is None:
            self.use_api = self._should_use_api()
        
        if self.use_api:
            self._initialize_api_client(start_server)
        else:
            self._initialize_direct_client()

    def _should_use_api(self) -> bool:
        """Determine whether to use API mode based on server availability"""
        # Check registry first
        server_info = self.registry.get_server_info()
        if server_info and self.registry.is_server_alive(server_info):
            self.logger.info(f"Found existing server at {server_info.host}:{server_info.port}")
            # Update host/port to use the registered server
            self.host = server_info.host
            self.port = server_info.port
            return True
        
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
        server_info = self.registry.get_server_info()

        # Check if we have a registered server and it's alive
        if server_info and self.registry.is_server_alive(server_info):
            self.host = server_info.host
            self.port = server_info.port
            self.logger.info(f"Using existing server at {server_info.host}:{server_info.port}")
        else:
            if start_server:
                self.logger.warning("No running server found, starting new instance...")
                self.server_instance = MilvusServerInstance(self.db_path, self.host, self.port)
                self.server_instance.start_server(threaded=True)
                
                # Update host/port with the registered server info
                server_info = self.registry.get_server_info()
                if server_info:
                    self.host = server_info.host
                    self.port = server_info.port
                self.logger.info(f"Starting server {server_info} at {server_info.host}:{server_info.port}")
            else:
                raise RuntimeError(f"No server running for database {self.db_path} and start_server=False")
        
        self.api_client = MilvusAPIClient(self.host, self.port)
    
    def get_server_info(self) -> Optional[ServerInfo]:
        """Get information about the current server"""
        return self.registry.get_server_info()
    
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
    
    def search(self, collection_name: str, query_vector: List[float]|numpy.ndarray,
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
        if isinstance(query_vector, numpy.ndarray):
            query_vector = query_vector.tolist()
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
                search_results = Utils.extract_hits(results[0], output_fields)

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
    """API client for communicating with MilvusServer across machines"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.base_url = f"http://{host}:{port}"
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, timeout: int = 30) -> Dict:
        """Make HTTP request to the API server"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            headers = {'Content-Type': 'application/json'}
        
        # Log the request for debugging
            self.logger.info(f"Making {method} request to {url}")
            if data:
                self.logger.info(f"Request data keys: {list(data.keys())}")

            if method.upper() == 'GET':
                response = requests.get(url, timeout=timeout, headers=headers)
            elif method.upper() == 'POST':
                # Validate data before sending
                if data is not None:
                    self._validate_request_data(data, endpoint)
                response = requests.post(url, json=data, timeout=timeout, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
        
            # Check if response is successful
            if response.status_code == 500:
                # Try to parse as JSON first
                error_detail = "Unknown server error"
                try:
                    json_response = response.json()
                    if isinstance(json_response, dict):
                        error_detail = json_response.get('error', 'Unknown server error')
                    else:
                        error_detail = str(json_response)
                except (ValueError, requests.exceptions.JSONDecodeError):
                    # If response is not JSON, safely extract text content
                    raw_text = response.text if response.text else "No error details available"
                    # Clean up HTML content if present
                    if '<html>' in raw_text.lower() or '<!doctype' in raw_text.lower():
                        # It's likely an HTML error page, extract useful info
                        if '<title>' in raw_text and '</title>' in raw_text:
                            start = raw_text.find('<title>') + 7
                            end = raw_text.find('</title>')
                            error_detail = f"HTML Error Page: {raw_text[start:end]}"
                        else:
                            error_detail = "HTML Error Page (details unavailable)"
                    else:
                        # Truncate long text responses and sanitize
                        error_detail = raw_text[:500].strip()
                        # Remove any potentially problematic characters
                        error_detail = ''.join(char for char in error_detail if ord(char) < 127 and char.isprintable())
                        if not error_detail:
                            error_detail = "Unreadable server error response"
                
                self.logger.error(f"Server error (500): {error_detail}")
                raise RuntimeError(f"Server internal error: {error_detail}")
        
            response.raise_for_status()
        
            # Validate response is JSON
            try:
                return response.json()
            except (ValueError, requests.exceptions.JSONDecodeError) as e:
                self.logger.error(f"Invalid JSON response: {response.text[:200]}")
                raise RuntimeError(f"Invalid server response format: {str(e)}")
            
        except requests.ConnectionError as e:
            self.logger.error(f"Connection failed to {url}: {e}")
            raise RuntimeError(f"Cannot connect to server at {url}: {e}")
        except requests.Timeout as e:
            self.logger.error(f"Request timeout for {url}: {e}")
            raise RuntimeError(f"Request timeout after {timeout}s: {e}")
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise RuntimeError(f"API request failed: {e}")

    def _validate_request_data(self, data: Dict, endpoint: str) -> None:
        """Validate request data before sending to server"""
        if endpoint == '/search':
            required_fields = ['collection_name', 'query_vector', 'limit']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field '{field}' in search request")
        
            # Validate query_vector format
            query_vector = data.get('query_vector')
            if not isinstance(query_vector, list) or not query_vector:
                raise ValueError("query_vector must be a non-empty list")
        
            # Validate all vector elements are numbers
            if not all(isinstance(x, (int, float)) for x in query_vector):
                raise ValueError("query_vector must contain only numeric values")
        
            # Validate limit
            limit = data.get('limit')
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
        
            # Validate output_fields if provided
            output_fields = data.get('output_fields')
            if output_fields is not None and not isinstance(output_fields, list):
                raise ValueError("output_fields must be a list")
    
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
        
        response = self._make_request('POST', '/search', data, timeout=60)
        
        if response.get('status') == 'error':
            raise RuntimeError(response.get('error', 'Unknown search error'))
        
        return response.get('results', [])
    
    def search(self, collection_name: str, query_vector: List[float], 
               limit: int = 10, output_fields: List[str] = None,
               search_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform vector search via API"""
        
        # Validate inputs before making request
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        
        if not isinstance(query_vector, list) or not query_vector:
            raise ValueError("query_vector must be a non-empty list")
        
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        
        data = {
            'collection_name': collection_name,
            'query_vector': query_vector,
            'limit': limit,
            'output_fields': output_fields or ["*"],
            'search_params': search_params or {}
        }
    
        try:
            response = self._make_request('POST', '/search', data, timeout=60)
        
            if response.get('status') == 'error':
                error_msg = response.get('error', 'Unknown search error')
                self.logger.error(f"Search failed: {error_msg}")
                raise RuntimeError(f"Search failed: {error_msg}")
        
            results = response.get('results', [])
            self.logger.debug(f"Search completed successfully, found {len(results)} results")
            return results
        
        except Exception as e:
            self.logger.error(f"Search request failed: {str(e)}")
            # Re-raise with more context
            raise RuntimeError(f"Search failed for collection '{collection_name}': {str(e)}")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics"""
        response = self._make_request('GET', f'/collections/{collection_name}/stats')
        return response.get('stats', {})


# Convenience functions
def create_milvus_server(db_path: str, host: str = "0.0.0.0", port: int = 8765, 
                        use_api: bool = None, start_server: bool = True,
                        registry_dir: str = None) -> MilvusServer:
    """
    Create a MilvusServer instance with cross-machine support
    
    Args:
        db_path: Path to the Milvus database file
        host: API server host (use "0.0.0.0" to bind to all interfaces)
        port: API server port  
        use_api: Whether to use API mode (auto-detect if None)
        start_server: Whether to start server automatically
        registry_dir: Directory for server registry files
        
    Returns:
        MilvusServer instance
    """
    return MilvusServer(db_path, host, port, use_api, start_server, registry_dir)


def list_running_servers(registry_dir: str = None) -> List[Dict[str, Any]]:
    """
    List all running Milvus servers
    
    Args:
        registry_dir: Directory to search for server registry files
        
    Returns:
        List of server information dictionaries
    """
    if registry_dir is None:
        registry_dir = os.path.join(os.getcwd(), '.milvus_servers')
    
    if not os.path.exists(registry_dir):
        return []
    
    servers = []
    for filename in os.listdir(registry_dir):
        if filename.startswith('server_') and filename.endswith('.json'):
            try:
                filepath = os.path.join(registry_dir, filename)
                with open(filepath, 'r') as f:
                    server_data = json.load(f)
                
                # Validate server is still alive
                registry = ServerRegistry(server_data['db_path'], registry_dir)
                server_info = ServerInfo.from_dict(server_data)
                if registry.is_server_alive(server_info):
                    servers.append(server_data)
                else:
                    # Clean up dead server
                    registry._cleanup_registry()
            except:
                continue
    
    return servers


# Example usage and testing
if __name__ == "__main__":
    # Example: First instance (will start server)
    print("Creating first instance (server mode)...")
    server1 = create_milvus_server("./test_db/milvus.db", host="0.0.0.0", port=8765, use_api=True)
    
    print("Server info:", server1.get_server_info())
    print("Collections:", server1.list_collections())
    
    # Example: Second instance on same or different machine (will use API)
    print("\nCreating second instance (client mode)...")  
    server2 = create_milvus_server("./test_db/milvus.db", use_api=True)
    
    print("Collections via API:", server2.list_collections())
    
    # List all running servers
    print("\nRunning servers:")
    for server_info in list_running_servers("./test_db/.milvus_servers"):
        print(f"  {server_info['host']}:{server_info['port']} - {server_info['db_path']}")
    
    # Cleanup
    server1.close()
    server2.close()
