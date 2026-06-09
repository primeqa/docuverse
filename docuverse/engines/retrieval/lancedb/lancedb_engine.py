"""Back-compat shim. The dense engine has moved to ``lancedb_dense``.

External code that did ``from docuverse.engines.retrieval.lancedb.lancedb_engine
import LanceDBEngine`` still works — the name now resolves to the dense
subclass.
"""
from docuverse.engines.retrieval.lancedb.lancedb_dense import LanceDBDenseEngine

LanceDBEngine = LanceDBDenseEngine

__all__ = ["LanceDBEngine"]
