#!/usr/bin/env python3
"""DEPRECATED: Use db2db_copy.py --todb lancedb instead.

This script is kept for backwards compatibility. It delegates to
db2db_copy.py with --todb lancedb.
"""
import sys
import warnings


def main():
    warnings.warn(
        "milvus_copy_to_lancedb.py is deprecated. "
        "Use: db2db_copy.py --todb lancedb",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from scripts.milvus_utils.db2db_copy import main as _main
    except (ImportError, ModuleNotFoundError):
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "db2db_copy",
            os.path.join(os.path.dirname(__file__), "db2db_copy.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _main = mod.main
    return _main(["--todb", "lancedb", "--fromdb", "milvus"] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
