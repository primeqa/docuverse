import os
import json
import threading
import time
import socket
import hashlib
import fcntl
import random
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager
import logging

import numpy
from pymilvus import MilvusClient


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
    pid: int
    db_path: str
    start_time: float
    socket_path: str = None  # Unix domain socket path
    host: str = None  # TCP host (legacy)
    port: int = None  # TCP port (legacy)
    machine_id: str = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'port': self.port,
            'pid': self.pid,
            'db_path': self.db_path,
            'start_time': self.start_time,
            'machine_id': self.machine_id,
            'socket_path': self.socket_path
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
        self.lock_fd = None
        self._acquire_lock()

    def _get_machine_id(self) -> str:
        """Get a unique identifier for this machine"""
        try:
            # Try to get machine ID from various sources
            # if os.path.exists('/etc/machine-id'):
            #     with open('/etc/machine-id', 'r') as f:
            #         return f.read().strip()
            # elif os.path.exists('/var/lib/dbus/machine-id'):
            #     with open('/var/lib/dbus/machine-id', 'r') as f:
            #         return f.read().strip()
            # else:
                # Fallback: use hostname + MAC address
            import uuid
            mac = hex(uuid.getnode())
            hostname = socket.gethostname()
            return hashlib.md5(f"{hostname}:{mac}".encode()).hexdigest()
        except:
            # Last resort: just use hostname
            return socket.gethostname()

    @contextmanager
    def _acquire_lock(self):
        """Acquire file lock as context manager"""
        try:
            self.lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
            yield
        except OSError as e:
            self.logger.error(f"Failed to acquire file lock: {e}")
            raise
        finally:
            if self.lock_fd is not None:
                try:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                    os.close(self.lock_fd)
                except OSError as e:
                    self.logger.warning(f"Failed to release file lock: {e}")

    def __del__(self):
        """Release file lock on destruction"""
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
            except OSError as e:
                self.logger.warning(f"Failed to release file lock: {e}")

    def register_server(self, pid: int, socket_path: str = None, host: str = None, port: int = None) -> ServerInfo:
        """Register a server instance (supports both Unix socket and TCP)"""
        server_info = ServerInfo(
            pid=pid,
            db_path=self.db_path,
            start_time=time.time(),
            machine_id=self._get_machine_id(),
            socket_path=socket_path,
            host=host,
            port=port
        )

        with open(self.registry_file, 'w') as f:
            json.dump(server_info.to_dict(), f, indent=2)

        if socket_path:
            self.logger.info(f"Registered server at {socket_path} for database {self.db_path}")
        else:
            self.logger.info(f"Registered server {host}:{port} for database {self.db_path}")
        return server_info
    
    def get_server_info(self) -> Optional[ServerInfo]:
        """Get information about the registered server"""
        if not os.path.exists(self.registry_file):
            return None
        
        try:
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
            is_same_machine = server_info.machine_id == self._get_machine_id()

            # For same-machine servers, first check if process exists
            if is_same_machine:
                try:
                    os.kill(server_info.pid, 0)  # Check if process exists
                except (OSError, ProcessLookupError):
                    # Process doesn't exist locally, server is dead
                    return False

            # Now verify server responds (for both local and remote servers)
            if server_info.socket_path:
                # Unix socket: check if socket file exists and is connectable
                if not os.path.exists(server_info.socket_path):
                    return False
                try:
                    import socket as sock
                    s = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
                    s.settimeout(2)
                    s.connect(server_info.socket_path)
                    s.close()
                    return True
                except:
                    return False
            else:
                # TCP: use gRPC health check (works for both local and remote)
                from src.retrievers.protos.milvus_grpc_client import MilvusGRPCClient
                try:
                    client = MilvusGRPCClient(server_info.host, server_info.port)
                    result = client.health_check()
                    client.close()
                    return result
                except Exception as e:
                    # If this is a remote server, log more details about the failure
                    if not is_same_machine:
                        self.logger.debug(f"Remote server health check failed for {server_info.host}:{server_info.port}: {e}")
                    return False

        except Exception as e:
            self.logger.debug(f"Server health check failed: {e}")
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
    """Singleton server instance that manages the Milvus database and serves API requests.

    Use create_instance() static method to create new instances. Direct instantiation
    is not allowed to ensure proper singleton pattern implementation.
    """

    _instance = None
    _lock_file = None
    _lock_fd = None

    @staticmethod
    def create_instance(db_path: str, socket_path: str = None, host: str = "0.0.0.0", port: int = 8765):
        """Create a new MilvusServerInstance with thread-safety

        Args:
            db_path: Path to Milvus database file
            socket_path: Unix domain socket path (if None, uses TCP with host:port)
            host: Host address to bind server to (for TCP mode)
            port: Port to run server on (for TCP mode)

        Returns:
            New or existing MilvusServerInstance

        Raises:
            RuntimeError: If server is already running
        """
        # Create lock file path
        lock_dir = os.path.dirname(os.path.abspath(db_path))
        lock_file = os.path.join(lock_dir, ".milvus_instance.lock")
        MilvusServerInstance._lock_file = lock_file

        # Acquire file lock
        try:
            MilvusServerInstance._lock_fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY | os.O_EXCL)

            print(f"Process {os.getpid()} locking Milvus instance")
            fcntl.flock(MilvusServerInstance._lock_fd, fcntl.LOCK_EX)

            if not MilvusServerInstance._instance:
                MilvusServerInstance._instance = MilvusServerInstance(db_path, socket_path, host, port)

            print(f"Process {os.getpid()} finished creating Milvus instance (server not started yet).", flush=True)
            return MilvusServerInstance._instance

        except FileExistsError:
            try:
                MilvusServerInstance._attempt_lock_cleanup(lock_file)
                return None
            except OSError:
                return None


        except OSError as e:
            raise RuntimeError(f"Failed to acquire instance lock: {e}")
        finally:
            if MilvusServerInstance._lock_fd is not None:
                fcntl.flock(MilvusServerInstance._lock_fd, fcntl.LOCK_UN)
                os.close(MilvusServerInstance._lock_fd)
                MilvusServerInstance._lock_fd = None

    @staticmethod
    def _attempt_lock_cleanup(lock_file: str):
        if os.path.exists(lock_file):
            file_age = time.time() - os.path.getmtime(lock_file)
            if file_age > 10:  # Lock file older than 10 seconds
                try:
                    os.remove(lock_file)
                    print(f"Removed stale lock file (age: {file_age:.1f}s)")
                    return True
                except OSError:
                    pass  # Ignore errors during cleanup
        return False

    @staticmethod
    def remove_creation_lock():
        """Remove the MilvusServerInstance lock"""
        try:
            os.remove(MilvusServerInstance._lock_file)
        except OSError:
            pass

    @staticmethod
    def creation_in_progress():
        if MilvusServerInstance._lock_fd is not None:
            MilvusServerInstance._attempt_lock_cleanup(MilvusServerInstance._lock_file)
            return os.path.exists(MilvusServerInstance._lock_file)
        else:
            return False

    def __new__(cls, db_path: str, socket_path: str = None, host: str = "0.0.0.0", port: int = 8765):
        db_path = os.path.abspath(db_path)
        instance = super().__new__(cls)
        instance._initialized = False
        instance.registry = ServerRegistry(db_path)

        # Check if server already exists
        server_info = instance.registry.get_server_info()
        if server_info and instance.registry.is_server_alive(server_info):
            location = server_info.socket_path if server_info.socket_path else f"{server_info.host}:{server_info.port}"
            raise RuntimeError(f"Server already running at {location}")

        return instance

    def __init__(self, db_path: str, socket_path: str = None, host: str = "0.0.0.0", port: int = 8765):
        """Initialize MilvusServerInstance (private constructor)"""
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing MilvusServerInstance")

        self.db_path = os.path.abspath(db_path)
        self.socket_path = socket_path
        self.host = host
        self.port = port
        self.client = None
        self.grpc_server = None
        self.server_thread = None
        self.running = False
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.basicConfig(level=logging.INFO)

        # Server registry for cross-machine discovery
        self.registry = ServerRegistry(self.db_path)

        # Initialize under registry lock
        with self.registry._acquire_lock():
            if hasattr(self, '_initialized') and self._initialized:
                return
            self._initialize_client()
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

    def start_server(self, threaded: bool = True):
        """Start the gRPC server and register it"""
        if self.running:
            location = self.socket_path if self.socket_path else f"{self.host}:{self.port}"
            self.logger.warning(f"Server already running on {location}")
            return

        def run_server():
            try:
                # Use gRPC server for fast binary protocol
                from src.retrievers.protos.milvus_grpc_server import serve_grpc

                if self.socket_path:
                    self.logger.info(f"Starting Milvus gRPC server on Unix socket: {self.socket_path}")
                else:
                    self.logger.info(f"Starting Milvus gRPC server on {self.host}:{self.port}")

                self.grpc_server = serve_grpc(self.client, self.host, self.port, socket_path=self.socket_path)

                # Wait forever (blocking)
                self.grpc_server.wait_for_termination()
            except Exception as e:
                self.logger.error(f"Server error: {e}")

        if threaded:
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()

            # Wait a bit for the server to start
            time.sleep(2)

            # Check if server is actually running and register it
            if self._is_server_running():
                if self.socket_path:
                    self.registry.register_server(pid=os.getpid(), socket_path=self.socket_path)
                    self.running = True
                    self.logger.info(f"Server successfully started on Unix socket {self.socket_path}")
                else:
                    external_ip = self._get_external_ip()
                    self.registry.register_server(pid=os.getpid(), host=external_ip, port=self.port)
                    self.running = True
                    self.logger.info(f"Server successfully started on {external_ip}:{self.port}")
            else:
                raise RuntimeError("Failed to start server")
        else:
            run_server()
    
    def _is_server_running(self) -> bool:
        """Check if the server is running"""
        try:
            # Use gRPC health check
            from src.retrievers.protos.milvus_grpc_client import MilvusGRPCClient
            client = MilvusGRPCClient("localhost", self.port)
            result = client.health_check()
            client.close()
            return result
        except:
            return False
    
    def stop_server(self):
        """Stop the server and unregister it"""
        self.running = False
        self.registry.unregister_server()
        # Clean up Unix socket file if it exists
        if self.socket_path and os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except:
                pass
        self.logger.info("Server stop requested and unregistered")


class MilvusServer:
    """Main Milvus Server class that handles both server and client functionality"""
    
    def __init__(self, db_path: str, socket_path: str = None, host: str = "0.0.0.0", port: int = 8765,
                 use_api: bool = None, start_server: bool = True,
                 registry_dir: str = None):
        """
        Initialize MilvusServer with cross-machine support

        Args:
            db_path: Path to the file-based Milvus database
            socket_path: Unix domain socket path (if provided, uses Unix sockets instead of TCP)
            host: Host address for API server (use "0.0.0.0" to bind to all interfaces)
            port: Port for API server
            use_api: If True, use API client. If False, use direct file access. If None, auto-detect
            start_server: Whether to start the API server automatically
            registry_dir: Directory for server registry files (defaults to {db_dir}/.milvus_servers)
        """
        self.db_path = os.path.abspath(db_path)
        self.socket_path = socket_path
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
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)

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
            if server_info.socket_path:
                self.logger.info(f"Found existing server at {server_info.socket_path}")
                self.socket_path = server_info.socket_path
            else:
                self.logger.info(f"Found existing server at {server_info.host}:{server_info.port}")
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
        num_tries = 4
        trial = 1
        pid = os.getpid()

        while trial <= num_tries:
            """Initialize API client, starting server if needed"""
            nap_time = random.randint(10, 500) / 1000
            self.logger.info(f"Process id {pid} sleeping for {nap_time} seconds.")
            time.sleep(nap_time)  # Sleep 10-200ms
            existing_server_info = self.registry.get_server_info()

            # Check if we have a registered server and it's alive
            if existing_server_info and self.registry.is_server_alive(existing_server_info):
                if existing_server_info.socket_path:
                    self.socket_path = existing_server_info.socket_path
                    self.logger.info(f"Using existing server at {existing_server_info.socket_path}")
                else:
                    self.host = existing_server_info.host
                    self.port = existing_server_info.port
                    self.logger.info(f"Using existing server at {existing_server_info.host}:{existing_server_info.port}")
                break
            else:
                if start_server:
                    if MilvusServerInstance.creation_in_progress():
                        self.logger.warning(
                            f"Process: {pid}, trial {trial}: Creation in progress...")
                        time.sleep(2)
                        continue
                    else:
                        self.logger.warning(
                            f"Process: {pid}, trial {trial}: No running server found, starting new instance...")
                    try:
                        self.server_instance = MilvusServerInstance.create_instance(self.db_path, self.socket_path, self.host, self.port)
                        if self.server_instance is None:
                            trial += 1 # initialization in progress
                            time.sleep(2)
                            continue
                        self.server_instance.start_server(threaded=True)
                        # Update socket_path/host/port with the registered server info
                        existing_server_info = self.registry.get_server_info()
                        if existing_server_info:
                            if existing_server_info.socket_path:
                                self.socket_path = existing_server_info.socket_path
                                self.logger.info(f"Starting server {existing_server_info} at {existing_server_info.socket_path}")
                            else:
                                self.host = existing_server_info.host
                                self.port = existing_server_info.port
                                self.logger.info(f"Starting server {existing_server_info} at {existing_server_info.host}:{existing_server_info.port}")
                        # MilvusServerInstance.remove_creation_lock()
                        break
                    except Exception as e:
                        self.logger.warning(f"Received an error: {e} in process {pid}, "
                                            f"this is the try number {trial}, {num_tries-trial+1} remaining.")
                        time.sleep(1)
                        trial += 1
                        if trial > num_tries:
                            raise RuntimeError(f"Failed {trial} times: no server running for database {self.db_path} and start_server=False")
                else:
                    if trial >= num_tries:
                        raise RuntimeError(f"No server running for database {self.db_path} and start_server=False")

        self.api_client = MilvusAPIClient(socket_path=self.socket_path, host=self.host, port=self.port)
    
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
    
    def search(self, collection_name: str,
               query_vector: List[float]|numpy.ndarray = None,
               data: List[float]|numpy.ndarray = None,
               limit: int = 10, output_fields: List[str] = None, 
               search_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform vector search
        
        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            data: alternative name for query_vector
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
        if data is not None:
            query_vector = data
        if isinstance(query_vector, numpy.ndarray):
            query_vector = query_vector.tolist()
        if self.use_api:
            return self.api_client.search(collection_name=collection_name, query_vector=query_vector, limit=limit,
                                        output_fields=output_fields, search_params=search_params)
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
    """gRPC client for communicating with MilvusServer across machines"""

    def __init__(self, socket_path: str = None, host: str = "127.0.0.1", port: int = 8765):
        from src.retrievers.protos.milvus_grpc_client import MilvusGRPCClient

        self.socket_path = socket_path
        self.host = host
        self.port = port

        # gRPC client (supports both Unix socket and TCP)
        self.grpc_client = MilvusGRPCClient(host=host, port=port, socket_path=socket_path)
        self.logger = logging.getLogger(__name__)

    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists"""
        return self.grpc_client.has_collection(collection_name)

    def list_collections(self) -> List[str]:
        """List all collections"""
        return self.grpc_client.list_collections()

    def search(self, collection_name: str,
               data: List[float]|None=None,
               query_vector: List[float] = None,
               limit: int = 10, output_fields: List[str] = None,
               search_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform vector search via gRPC"""

        # Handle query_vector vs data parameter
        if query_vector is not None:
            data = query_vector

        if data is None:
            raise ValueError("Either 'data' or 'query_vector' must be provided")

        # Call gRPC client with error handling for missing collections
        try:
            return self.grpc_client.search(
                collection_name=collection_name,
                query_vector=data,
                limit=limit,
                output_fields=output_fields or ["*"]
            )
        except Exception as e:
            # Check if this is a collection not found error
            error_msg = str(e)
            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                # Get available collections
                try:
                    available_collections = self.grpc_client.list_collections()
                    # ANSI color codes: green for collection names
                    colored_collections = ', '.join([f"\033[92m{col}\033[0m" for col in available_collections]) if available_collections else 'none'
                    self.logger.error(f"Collection '{collection_name}' not found. Available collections: {colored_collections}")
                except:
                    pass  # If we can't list collections, just re-raise original error
            raise

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics"""
        # Note: This is not implemented in the gRPC service yet
        raise NotImplementedError("Collection stats not yet supported via gRPC")

    def close(self):
        """Close the gRPC client"""
        self.grpc_client.close()


# Convenience functions
def create_milvus_server(db_path: str,
                         socket_path: str = None,
                         host: str = "0.0.0.0",
                         port: int = 8765,
                         use_api: bool = None,
                         start_server: bool = True,
                         registry_dir: str = None) -> MilvusServer:
    """
    Create a MilvusServer instance with cross-machine support

    Args:
        db_path: Path to the Milvus database file
        socket_path: Unix domain socket path (if provided, uses Unix sockets for faster IPC)
        host: API server host (use "0.0.0.0" to bind to all interfaces, for TCP mode)
        port: API server port (for TCP mode)
        use_api: Whether to use API mode (auto-detect if None)
        start_server: Whether to start server automatically
        registry_dir: Directory for server registry files

    Returns:
        MilvusServer instance
    """
    return MilvusServer(db_path, socket_path, host, port, use_api, start_server, registry_dir)


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
    
    print("Server info:", server1.get_server_inao())
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
