from .lancedb import LanceDBEngine as _LanceDBBase
from .lancedb_dense import LanceDBDenseEngine

# Public name `LanceDBEngine` resolves to the dense subclass for back-compat.
LanceDBEngine = LanceDBDenseEngine

__all__ = ['LanceDBEngine', 'LanceDBDenseEngine']
