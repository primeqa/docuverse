#!/usr/bin/env python3
"""
Replace the vector index in a Milvus collection with an HNSW index.

Supports both file-based (.db) and server-based Milvus connections.
Note: Milvus Lite (.db files) only supports FLAT, IVF_FLAT, and AUTOINDEX.
HNSW requires a Milvus server. The script auto-detects the connection type
and adjusts available index types accordingly.

Usage:
    # On a Milvus server (supports HNSW):
    python milvus_create_hnsw_index.py --server localhost -c my_collection
    python milvus_create_hnsw_index.py --server localhost:19530 -c my_collection -M 64

    # On a .db file (HNSW not supported, will offer IVF_FLAT instead):
    python milvus_create_hnsw_index.py experiments/sap/sap.db --index-type IVF_FLAT

Examples:
    # HNSW on a local Milvus server
    python milvus_create_hnsw_index.py --server localhost \\
        -c wiki_en_milvus_dense_granite149m_1024_100 -M 128 --ef-construction 256

    # IVF_FLAT on a .db file
    python milvus_create_hnsw_index.py experiments/sap/sap.db \\
        --index-type IVF_FLAT --nlist 1024

    # Dry-run to see current index info
    python milvus_create_hnsw_index.py --server localhost -c my_collection --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from pymilvus import MilvusClient
except ImportError:
    print("Error: pymilvus is not installed. Install with: pip install 'pymilvus[model]'",
          file=sys.stderr)
    sys.exit(1)

# Index types supported by Milvus Lite (file-based .db)
FILE_INDEX_TYPES = {"FLAT", "IVF_FLAT", "AUTOINDEX"}
# Index types that require a Milvus server
SERVER_INDEX_TYPES = FILE_INDEX_TYPES | {"HNSW", "IVF_SQ8", "IVF_PQ", "RHNSW_FLAT"}


def connect(args):
    """Create a MilvusClient from CLI args. Returns (client, is_file_mode)."""
    if args.server:
        server = args.server
        # Support named servers from config/milvus_servers.json
        servers_file = Path(__file__).resolve().parents[2] / "config" / "milvus_servers.json"
        if servers_file.exists():
            with open(servers_file) as f:
                named_servers = json.load(f)
            if server in named_servers:
                s = named_servers[server]
                server = f"{s['host']}:{s.get('port', 19530)}"
        if ":" not in server:
            server = f"{server}:19530"
        uri = f"http://{server}"
        print(f"Connecting to Milvus server: {uri}")
        return MilvusClient(uri=uri), False
    else:
        db_path = Path(args.db_file)
        if not db_path.exists():
            print(f"Error: file not found: {args.db_file}", file=sys.stderr)
            sys.exit(1)
        print(f"Opening database: {db_path.resolve()}")
        return MilvusClient(uri=str(db_path)), True


def resolve_collection(client, collection_name):
    """Discover or validate the collection name."""
    collections = client.list_collections()
    if not collections:
        print("Error: no collections found.", file=sys.stderr)
        sys.exit(1)

    if collection_name:
        if collection_name not in collections:
            print(f"Error: collection '{collection_name}' not found. "
                  f"Available: {collections}", file=sys.stderr)
            sys.exit(1)
        return collection_name
    elif len(collections) == 1:
        return collections[0]
    else:
        print(f"Multiple collections found: {collections}")
        print("Please specify one with --collection / -c", file=sys.stderr)
        sys.exit(1)


def find_vector_field(client, collection_name, field_name=None):
    """Find the dense vector field in a collection schema."""
    desc = client.describe_collection(collection_name)
    vector_fields = []
    for f in desc.get("fields", []):
        dtype = f.get("type", 0)
        # FLOAT_VECTOR=101, FLOAT16_VECTOR=102, BFLOAT16_VECTOR=103
        if isinstance(dtype, int) and dtype in (101, 102, 103):
            vector_fields.append(f)
        elif "VECTOR" in str(dtype) and "SPARSE" not in str(dtype):
            vector_fields.append(f)

    if not vector_fields:
        print(f"Error: no dense vector fields found in collection '{collection_name}'",
              file=sys.stderr)
        sys.exit(1)

    if field_name:
        matches = [f for f in vector_fields if f["name"] == field_name]
        if not matches:
            names = [f["name"] for f in vector_fields]
            print(f"Error: field '{field_name}' not found. Vector fields: {names}",
                  file=sys.stderr)
            sys.exit(1)
        return matches[0]

    if len(vector_fields) > 1:
        names = [f["name"] for f in vector_fields]
        print(f"Multiple vector fields found: {names}. "
              f"Please specify one with --field", file=sys.stderr)
        sys.exit(1)

    return vector_fields[0]


def get_current_index(client, collection_name, field_name):
    """Get the current index info for a field, if any."""
    try:
        indexes = client.list_indexes(collection_name)
        for idx_name in indexes:
            idx = client.describe_index(collection_name, index_name=idx_name)
            if idx.get("field_name") == field_name:
                return idx
    except Exception:
        pass
    return None


def build_index_params(index_type, metric, args):
    """Build the index params dict for the requested index type."""
    params = {
        "index_type": index_type,
        "metric_type": metric,
    }
    if index_type == "HNSW":
        params["params"] = {
            "M": args.M,
            "efConstruction": args.ef_construction,
        }
    elif index_type == "IVF_FLAT":
        params["params"] = {
            "nlist": args.nlist,
        }
    # FLAT and AUTOINDEX need no extra params
    return params


def format_index_info(idx):
    """Format index info dict as a readable string."""
    if not idx:
        return "none"
    parts = [idx.get("index_type", "?"), f"metric={idx.get('metric_type', '?')}"]
    p = idx.get("params", {})
    if isinstance(p, dict):
        for k, v in p.items():
            if k not in ("index_type", "metric_type"):
                parts.append(f"{k}={v}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Create/replace a vector index on a Milvus collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # HNSW on a Milvus server
  python milvus_create_hnsw_index.py --server localhost -c my_collection -M 128

  # IVF_FLAT on a .db file
  python milvus_create_hnsw_index.py data.db --index-type IVF_FLAT --nlist 1024

  # Dry-run (show current index, don't change anything)
  python milvus_create_hnsw_index.py --server localhost -c my_collection --dry-run
        """,
    )
    # Connection: either --server or a db_file
    parser.add_argument("db_file", nargs="?", default=None,
                        help="Path to Milvus .db file")
    parser.add_argument("--server", "-s", default=None,
                        help="Milvus server (host, host:port, or name from milvus_servers.json)")

    # Collection / field
    parser.add_argument("--collection", "-c", default=None,
                        help="Collection name (auto-detected if only one exists)")
    parser.add_argument("--field", "-f", default=None,
                        help="Vector field name (auto-detected if only one exists)")

    # Index parameters
    parser.add_argument("--index-type", "-t", default="HNSW",
                        help="Index type: HNSW, IVF_FLAT, FLAT, AUTOINDEX (default: HNSW)")
    parser.add_argument("-M", type=int, default=128,
                        help="HNSW M parameter — max connections per node (default: 128)")
    parser.add_argument("--ef-construction", type=int, default=128,
                        help="HNSW efConstruction — search width during build (default: 128)")
    parser.add_argument("--nlist", type=int, default=1024,
                        help="IVF_FLAT nlist parameter — number of clusters (default: 1024)")
    parser.add_argument("--metric", default="IP", choices=["IP", "L2", "COSINE"],
                        help="Distance metric (default: IP)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    args = parser.parse_args()

    if args.server is None and args.db_file is None:
        parser.error("either db_file or --server is required")

    # Connect
    client, is_file_mode = connect(args)

    # Validate index type for connection mode
    index_type = args.index_type.upper()
    allowed = FILE_INDEX_TYPES if is_file_mode else SERVER_INDEX_TYPES
    if index_type not in allowed:
        if is_file_mode:
            print(f"\nError: Milvus Lite (.db files) only supports: {sorted(FILE_INDEX_TYPES)}")
            print(f"  HNSW requires a Milvus server. Use --server to connect to one,")
            print(f"  or use --index-type IVF_FLAT for an approximate index on .db files.")
        else:
            print(f"Error: unsupported index type '{index_type}'. "
                  f"Supported: {sorted(allowed)}", file=sys.stderr)
        client.close()
        return 1

    # Resolve collection
    collection_name = resolve_collection(client, args.collection)

    # Find the vector field
    vector_field = find_vector_field(client, collection_name, args.field)
    field_name = vector_field["name"]
    dim = vector_field.get("params", {}).get("dim", "?")

    # Get row count
    stats = client.get_collection_stats(collection_name)
    row_count = stats.get("row_count", 0)

    # Get current index
    current_idx = get_current_index(client, collection_name, field_name)

    print(f"\nCollection:    {collection_name}  ({row_count:,} rows)")
    print(f"Field:         {field_name}  (dim={dim})")
    print(f"Current index: {format_index_info(current_idx)}")

    new_params = build_index_params(index_type, args.metric, args)
    print(f"New index:     {format_index_info(new_params)}")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        client.close()
        return 0

    # Drop existing index
    if current_idx:
        idx_name = current_idx.get("index_name", field_name)
        print(f"\nDropping existing index '{idx_name}'...")
        client.drop_index(collection_name, index_name=idx_name)

    # Release collection before creating a new index
    try:
        client.release_collection(collection_name)
    except Exception:
        pass

    # Create new index
    print(f"Creating {index_type} index on '{field_name}'...")
    t0 = time.time()

    index_params = client.prepare_index_params()
    index_params.add_index(field_name=field_name, **new_params)
    client.create_index(collection_name, index_params=index_params)

    elapsed = time.time() - t0
    print(f"Index created in {elapsed:.1f}s")

    # Load collection to make it queryable
    print("Loading collection...")
    client.load_collection(collection_name)

    # Verify
    new_idx = get_current_index(client, collection_name, field_name)
    if new_idx:
        print(f"\nVerified: {format_index_info(new_idx)}, state={new_idx.get('state', '?')}")
    else:
        print("\nWarning: could not verify new index.", file=sys.stderr)

    client.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
