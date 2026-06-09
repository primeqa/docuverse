from .lancedb import LanceDBEngine as _LanceDBBase
from .lancedb_dense import LanceDBDenseEngine
from .lancedb_bm25 import LanceDBBM25Engine

# Public name `LanceDBEngine` resolves to the dense subclass for back-compat.
LanceDBEngine = LanceDBDenseEngine

__all__ = ['LanceDBEngine', 'LanceDBDenseEngine', 'LanceDBBM25Engine']
