#!/usr/bin/env python3
"""
Copy a collection from a Milvus Lite .db file to a Milvus server.

Reads the schema and all data (including embeddings) from a local .db file
and inserts it into a Milvus server, optionally creating an HNSW index.
No re-computation of embeddings is needed.

Usage:
    python milvus_copy_to_server.py <database_file> --server <host>
    python milvus_copy_to_server.py <database_file> -s localhost -c <collection>
    python milvus_copy_to_server.py <database_file> -s localhost --dry-run

Examples:
    python milvus_copy_to_server.py experiments/sap/sap.db -s localhost

    python milvus_copy_to_server.py experiments/wikipedia_en/wiki-en.db \\
        -s localhost --batch-size 5000 -M 64 --ef-construction 256

    python milvus_copy_to_server.py experiments/sap/sap.db -s localhost \\
        --target-collection sap_hnsw --drop-existing
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from pymilvus import (
        MilvusClient, FieldSchema, CollectionSchema, DataType,
    )
except ImportError:
    print("Error: pymilvus is not installed. Install with: pip install 'pymilvus[model]'",
          file=sys.stderr)
    sys.exit(1)

from tqdm.auto import tqdm

# Map integer type codes from describe_collection to DataType enum
_INT_TO_DTYPE = {
    1: DataType.BOOL,
    2: DataType.INT8,
    3: DataType.INT16,
    4: DataType.INT32,
    5: DataType.INT64,
    10: DataType.FLOAT,
    11: DataType.DOUBLE,
    20: DataType.STRING,
    21: DataType.VARCHAR,
    22: DataType.ARRAY,
    23: DataType.JSON,
    100: DataType.BINARY_VECTOR,
    101: DataType.FLOAT_VECTOR,
    102: DataType.FLOAT16_VECTOR,
    103: DataType.BFLOAT16_VECTOR,
    104: DataType.SPARSE_FLOAT_VECTOR,
}

_DTYPE_NAMES = {
    5: "INT64", 21: "VARCHAR", 101: "FLOAT_VECTOR", 102: "FLOAT16_VECTOR",
    103: "BFLOAT16_VECTOR", 104: "SPARSE_FLOAT_VECTOR",
}


def resolve_server(server_str):
    """Resolve a server string to host:port URI."""
    servers_file = Path(__file__).resolve().parents[2] / "config" / "milvus_servers.json"
    if servers_file.exists():
        with open(servers_file) as f:
            named_servers = json.load(f)
        if server_str in named_servers:
            s = named_servers[server_str]
            return f"http://{s['host']}:{s.get('port', 19530)}"
    if ":" not in server_str:
        server_str = f"{server_str}:19530"
    return f"http://{server_str}"


def resolve_collection(client, collection_name):
    """Discover or validate the collection name."""
    collections = client.list_collections()
    if not collections:
        print("Error: no collections found in source.", file=sys.stderr)
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


def build_schema_from_description(desc):
    """Build a CollectionSchema from describe_collection() output.

    Returns (schema, fields_info) where fields_info is a list of dicts
    with parsed metadata for each field.
    """
    fields = []
    fields_info = []
    for f in desc.get("fields", []):
        name = f["name"]
        type_int = f["type"]
        dtype = _INT_TO_DTYPE.get(type_int)
        if dtype is None:
            print(f"Warning: unknown field type {type_int} for field '{name}', skipping",
                  file=sys.stderr)
            continue

        is_primary = f.get("is_primary", False)
        auto_id = f.get("auto_id", False)
        params = f.get("params", {})

        kwargs = {}
        if "dim" in params:
            kwargs["dim"] = int(params["dim"])
        if "max_length" in params:
            kwargs["max_length"] = int(params["max_length"])

        field_schema = FieldSchema(
            name=name, dtype=dtype,
            is_primary=is_primary, auto_id=auto_id,
            **kwargs,
        )
        fields.append(field_schema)

        info = {
            "name": name,
            "type_int": type_int,
            "type_name": _DTYPE_NAMES.get(type_int, str(dtype)),
            "is_primary": is_primary,
            "auto_id": auto_id,
        }
        info.update(kwargs)
        fields_info.append(info)

    collection_name = desc.get("collection_name", "")
    schema = CollectionSchema(fields, description=desc.get("description", ""))
    return schema, fields_info


def find_vector_fields(fields_info):
    """Return list of vector field names from fields_info."""
    vector_types = {101, 102, 103, 104}
    return [f["name"] for f in fields_info if f["type_int"] in vector_types]


def build_index_params(client, fields_info, index_type, metric, M, ef_construction):
    """Build index params for the target collection."""
    index_params = client.prepare_index_params()
    vector_fields = find_vector_fields(fields_info)
    for vf in vector_fields:
        if index_type == "HNSW":
            index_params.add_index(
                field_name=vf, index_type="HNSW", metric_type=metric,
                params={"M": M, "efConstruction": ef_construction},
            )
        elif index_type == "IVF_FLAT":
            index_params.add_index(
                field_name=vf, index_type="IVF_FLAT", metric_type=metric,
                params={"nlist": 1024},
            )
        else:
            index_params.add_index(
                field_name=vf, index_type=index_type, metric_type=metric,
            )
    return index_params


def get_output_fields(fields_info):
    """Get the list of field names to read from source (skip auto-id primary key)."""
    return [f["name"] for f in fields_info if not f.get("auto_id", False)]


def insert_with_retry(target, tgt_collection, batch, max_retries=5):
    """Insert a batch with exponential backoff on transient failures."""
    for attempt in range(max_retries):
        try:
            target.insert(collection_name=tgt_collection, data=batch)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"\n  Insert failed ({e}), retrying in {wait}s "
                  f"(attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)


def copy_data(source, target, src_collection, tgt_collection,
              output_fields, skip_fields, batch_size, total_rows, skip_rows=0):
    """Copy all rows from source to target collection using query_iterator."""
    print("Initializing source iterator (this may take a while for large databases)...",
          flush=True)
    t_iter = time.time()
    iterator = source.query_iterator(
        collection_name=src_collection,
        batch_size=batch_size,
        output_fields=output_fields,
    )
    print(f"Iterator ready in {time.time() - t_iter:.1f}s. Reading first batch...", flush=True)

    copied = 0
    skipped = 0

    # Skip rows that are already in the target (for --resume)
    if skip_rows > 0:
        skip_pbar = tqdm(total=skip_rows, desc="Skipping rows", unit="rows")
        while skipped < skip_rows:
            batch = iterator.next()
            if not batch:
                break
            skipped += len(batch)
            skip_pbar.update(len(batch))
        skip_pbar.close()

    pbar = tqdm(total=total_rows - skip_rows, desc="Copying rows", unit="rows")
    while True:
        batch = iterator.next()
        if not batch:
            break
        if copied == 0:
            print(f"First batch received ({len(batch)} rows). Copying...", flush=True)
        # Strip auto-id / primary key fields that Milvus always returns
        if skip_fields:
            batch = [{k: v for k, v in row.items() if k not in skip_fields}
                     for row in batch]
        insert_with_retry(target, tgt_collection, batch)
        copied += len(batch)
        pbar.update(len(batch))

    pbar.close()
    iterator.close()
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Copy a collection from a Milvus .db file to a Milvus server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python milvus_copy_to_server.py experiments/sap/sap.db -s localhost
  python milvus_copy_to_server.py data.db -s localhost -c my_collection --batch-size 5000
  python milvus_copy_to_server.py data.db -s localhost --target-collection new_name
  python milvus_copy_to_server.py data.db -s localhost --dry-run
        """,
    )
    parser.add_argument("db_file", help="Path to source Milvus .db file")
    parser.add_argument("--server", "-s", required=True,
                        help="Target Milvus server (host, host:port, or name from milvus_servers.json)")
    parser.add_argument("--collection", "-c", default=None,
                        help="Source collection name (auto-detected if only one)")
    parser.add_argument("--target-collection", default=None,
                        help="Target collection name (defaults to source collection name)")
    parser.add_argument("--batch-size", "-b", type=int, default=1000,
                        help="Rows per batch for read/insert (default: 1000)")
    parser.add_argument("--index-type", "-t", default="HNSW",
                        help="Index type on target: HNSW, IVF_FLAT, FLAT (default: HNSW)")
    parser.add_argument("-M", type=int, default=128,
                        help="HNSW M parameter (default: 128)")
    parser.add_argument("--ef-construction", type=int, default=128,
                        help="HNSW efConstruction parameter (default: 128)")
    parser.add_argument("--metric", default="IP", choices=["IP", "L2", "COSINE"],
                        help="Distance metric (default: IP)")
    parser.add_argument("--drop-existing", action="store_true",
                        help="Drop target collection if it already exists")
    parser.add_argument("--no-index", action="store_true",
                        help="Copy data without indexing, then build the index once at the end. "
                             "Much faster for large datasets.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous interrupted copy (skips rows already in target)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without copying")
    args = parser.parse_args()

    # Validate source
    db_path = Path(args.db_file)
    if not db_path.exists():
        print(f"Error: file not found: {args.db_file}", file=sys.stderr)
        return 1

    # Connect to source
    print(f"Source:  {db_path.resolve()}")
    source = MilvusClient(uri=str(db_path))

    # Connect to target
    target_uri = resolve_server(args.server)
    print(f"Target:  {target_uri}")
    target = MilvusClient(uri=target_uri)

    # Resolve source collection
    src_collection = resolve_collection(source, args.collection)
    tgt_collection = args.target_collection or src_collection

    # Get source schema and stats
    desc = source.describe_collection(src_collection)
    schema, fields_info = build_schema_from_description(desc)
    stats = source.get_collection_stats(src_collection)
    total_rows = stats.get("row_count", 0)

    output_fields = get_output_fields(fields_info)
    vector_fields = find_vector_fields(fields_info)

    # Display plan
    print(f"\nSource collection:  {src_collection}  ({total_rows:,} rows)")
    print(f"Target collection:  {tgt_collection}")
    print(f"Index type:         {args.index_type}")
    if args.index_type == "HNSW":
        print(f"  M={args.M}, efConstruction={args.ef_construction}, metric={args.metric}")
    print(f"Batch size:         {args.batch_size:,}")
    print(f"\nSchema ({len(fields_info)} fields):")
    for f in fields_info:
        details = []
        if f.get("is_primary"):
            details.append("PRIMARY KEY")
        if f.get("auto_id"):
            details.append("auto_id (skip in copy)")
        if "dim" in f:
            details.append(f"dim={f['dim']}")
        if "max_length" in f:
            details.append(f"max_length={f['max_length']}")
        detail_str = ", ".join(details) if details else ""
        print(f"  {f['name']:<25} {f['type_name']:<22} {detail_str}")

    print(f"\nFields to copy: {output_fields}")
    print(f"Vector fields:  {vector_fields}")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        source.close()
        target.close()
        return 0

    # Handle target collection: resume, drop, or create new
    already_copied = 0
    target_exists = target.has_collection(tgt_collection)

    if target_exists and args.resume:
        target_stats = target.get_collection_stats(tgt_collection)
        already_copied = target_stats.get("row_count", 0)
        print(f"\nResuming: target '{tgt_collection}' has {already_copied:,} rows, "
              f"skipping those from source.")
    elif target_exists and args.drop_existing:
        print(f"\nDropping existing target collection '{tgt_collection}'...")
        target.drop_collection(tgt_collection)
        target_exists = False
    elif target_exists:
        print(f"\nError: target collection '{tgt_collection}' already exists. "
              f"Use --drop-existing to overwrite, --resume to continue, "
              f"or --target-collection for a different name.",
              file=sys.stderr)
        source.close()
        target.close()
        return 1

    if not target_exists:
        print(f"\nCreating target collection '{tgt_collection}'...")
        if args.no_index:
            # Create without index — data inserts will be much faster
            target.create_collection(
                collection_name=tgt_collection,
                schema=schema,
                consistency_level="Eventually",
            )
            print("Collection created without vector index (will build after copy).")
        else:
            index_params = build_index_params(
                target, fields_info, args.index_type, args.metric,
                args.M, args.ef_construction,
            )
            target.create_collection(
                collection_name=tgt_collection,
                schema=schema,
                index_params=index_params,
                consistency_level="Eventually",
            )
            print(f"Collection created with {args.index_type} index.")

    # Copy data
    remaining = total_rows - already_copied
    if remaining <= 0:
        print(f"\nAll {total_rows:,} rows already in target. Nothing to copy.")
        source.close()
        target.close()
        return 0

    print(f"\nCopying {remaining:,} rows" +
          (f" (skipping first {already_copied:,})..." if already_copied else "..."))
    t0 = time.time()
    # Fields to strip from query results (auto-id PKs are always returned by Milvus)
    skip_fields = {f["name"] for f in fields_info if f.get("auto_id")}
    copied = copy_data(source, target, src_collection, tgt_collection,
                       output_fields, skip_fields, args.batch_size,
                       total_rows, skip_rows=already_copied)
    elapsed = time.time() - t0
    rate = copied / elapsed if elapsed > 0 else 0
    print(f"Copied {copied:,} rows in {elapsed:.1f}s ({rate:,.0f} rows/s)")

    # Build index after copy if --no-index was used
    if args.no_index:
        print(f"\nBuilding {args.index_type} index (this may take a while)...")
        t_idx = time.time()
        index_params = build_index_params(
            target, fields_info, args.index_type, args.metric,
            args.M, args.ef_construction,
        )
        target.create_index(
            collection_name=tgt_collection,
            index_params=index_params,
        )
        idx_elapsed = time.time() - t_idx
        print(f"Index built in {idx_elapsed:.1f}s")

    # Load collection
    print("Loading collection...")
    target.load_collection(tgt_collection)

    # Verify
    target_stats = target.get_collection_stats(tgt_collection)
    target_rows = target_stats.get("row_count", 0)
    print(f"\nVerification:")
    print(f"  Source rows:  {total_rows:,}")
    print(f"  Target rows:  {target_rows:,}")
    if target_rows == total_rows:
        print("  Status: OK")
    else:
        print("  Status: row count mismatch (target may still be indexing; "
              "counts should converge shortly)")

    source.close()
    target.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
