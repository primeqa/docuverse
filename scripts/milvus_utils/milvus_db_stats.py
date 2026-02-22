#!/usr/bin/env python3
"""
Display statistics for a Milvus Lite database file (.db).

Shows database overview, per-collection schema/index/row details,
and optionally sample data rows.

Usage:
    python milvus_db_stats.py <database_file>
    python milvus_db_stats.py <database_file> --collection <name>
    python milvus_db_stats.py <database_file> --sample 5
    python milvus_db_stats.py <database_file> --json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def format_size(size_bytes):
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


_DTYPE_MAP = {
    0: "NONE", 1: "BOOL", 2: "INT8", 3: "INT16", 4: "INT32", 5: "INT64",
    10: "FLOAT", 11: "DOUBLE", 20: "STRING", 21: "VARCHAR", 22: "ARRAY",
    23: "JSON",
    100: "BINARY_VECTOR", 101: "FLOAT_VECTOR", 102: "FLOAT16_VECTOR",
    103: "BFLOAT16_VECTOR", 104: "SPARSE_FLOAT_VECTOR",
}


def dtype_name(dtype):
    """Extract a readable name from a pymilvus DataType enum or int."""
    if isinstance(dtype, int):
        return _DTYPE_MAP.get(dtype, f"UNKNOWN({dtype})")
    s = str(dtype)
    # e.g. <DataType.FLOAT_VECTOR: 101> -> "FLOAT_VECTOR"
    if "." in s and ":" in s:
        return s.split(".")[1].split(":")[0]
    return s


def get_collection_info(client, collection_name):
    """Gather schema, index, and row-count info for a collection."""
    info = {"name": collection_name}

    # Row count
    try:
        stats = client.get_collection_stats(collection_name)
        info["row_count"] = stats.get("row_count", 0)
    except Exception as e:
        info["row_count"] = f"error: {e}"

    # Schema via describe_collection
    try:
        desc = client.describe_collection(collection_name)
        fields = []
        for f in desc.get("fields", []):
            field_info = {
                "name": f.get("name", ""),
                "type": dtype_name(f.get("type", "")),
                "is_primary": f.get("is_primary", False),
                "auto_id": f.get("auto_id", False),
            }
            params = f.get("params", {})
            if isinstance(params, dict):
                if "dim" in params:
                    field_info["dim"] = params["dim"]
                if "max_length" in params:
                    field_info["max_length"] = params["max_length"]
            fields.append(field_info)
        info["fields"] = fields
        info["description"] = desc.get("description", "")
        info["auto_id"] = desc.get("auto_id", False)
        info["consistency_level"] = str(desc.get("consistency_level", ""))
    except Exception as e:
        info["schema_error"] = str(e)

    # Index info
    try:
        indexes = client.list_indexes(collection_name)
        index_details = []
        for idx_name in indexes:
            try:
                idx_desc = client.describe_index(collection_name, index_name=idx_name)
                index_details.append(idx_desc)
            except Exception:
                index_details.append({"index_name": idx_name, "error": "could not describe"})
        info["indexes"] = index_details
    except Exception as e:
        info["index_error"] = str(e)

    return info


def print_collection_info(info, sample_rows=None):
    """Pretty-print collection information."""
    name = info["name"]
    row_count = info.get("row_count", "?")
    print(f"\n  Collection: {name}")
    print(f"  Rows: {row_count:,}" if isinstance(row_count, int) else f"  Rows: {row_count}")
    if info.get("description"):
        print(f"  Description: {info['description']}")
    if info.get("consistency_level"):
        print(f"  Consistency: {info['consistency_level']}")

    # Schema
    fields = info.get("fields", [])
    if fields:
        print(f"\n  Schema ({len(fields)} fields):")
        print(f"    {'Name':<25} {'Type':<22} {'Details'}")
        print(f"    {'-'*25} {'-'*22} {'-'*30}")
        for f in fields:
            details = []
            if f.get("is_primary"):
                details.append("PRIMARY KEY")
            if f.get("auto_id"):
                details.append("auto_id")
            if f.get("dim"):
                details.append(f"dim={f['dim']}")
            if f.get("max_length"):
                details.append(f"max_length={f['max_length']}")
            detail_str = ", ".join(details)
            print(f"    {f['name']:<25} {f['type']:<22} {detail_str}")
    elif "schema_error" in info:
        print(f"  Schema error: {info['schema_error']}")

    # Indexes
    indexes = info.get("indexes", [])
    if indexes:
        print(f"\n  Indexes ({len(indexes)}):")
        for idx in indexes:
            if isinstance(idx, dict):
                idx_name = idx.get("index_name", "?")
                idx_type = idx.get("index_type", "?")
                metric = idx.get("metric_type", "?")
                field = idx.get("field_name", "?")
                params = {k: v for k, v in idx.items()
                          if k not in ("index_name", "index_type", "metric_type",
                                       "field_name", "total_rows", "indexed_rows",
                                       "pending_index_rows", "state", "error")}
                print(f"    - {idx_name}: type={idx_type}, metric={metric}, field={field}")
                if params:
                    print(f"      params: {params}")
                state = idx.get("state", "")
                total = idx.get("total_rows", "")
                indexed = idx.get("indexed_rows", "")
                if state:
                    print(f"      state={state}, indexed_rows={indexed}/{total}")
            else:
                print(f"    - {idx}")
    elif "index_error" in info:
        print(f"  Index error: {info['index_error']}")

    # Sample data
    if sample_rows:
        print(f"\n  Sample data ({len(sample_rows)} rows):")
        for i, row in enumerate(sample_rows):
            print(f"    Row {i}:")
            for key, val in row.items():
                display = str(val)
                if isinstance(val, str) and len(val) > 80:
                    display = val[:80] + "..."
                elif isinstance(val, list) and len(val) > 5:
                    display = f"[{val[0]:.4f}, {val[1]:.4f}, ... ] (len={len(val)})"
                elif isinstance(val, dict):
                    n = len(val)
                    display = f"{{sparse vector, {n} non-zero entries}}"
                print(f"      {key}: {display}")


def get_sample_data(client, collection_name, n, fields):
    """Fetch n sample rows from a collection."""
    # Only request non-vector fields by default, plus vector dims
    output_fields = []
    vector_fields = []
    for f in fields:
        ftype = f.get("type", "")
        if "VECTOR" in ftype:
            vector_fields.append(f["name"])
        output_fields.append(f["name"])

    try:
        rows = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=output_fields,
            limit=n,
        )
        return rows
    except Exception as e:
        print(f"    (Could not fetch sample data: {e})")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Display statistics for a Milvus Lite database file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python milvus_db_stats.py milvus_demo.db
  python milvus_db_stats.py milvus_demo.db --collection my_collection
  python milvus_db_stats.py milvus_demo.db --sample 3
  python milvus_db_stats.py milvus_demo.db --json
        """,
    )
    parser.add_argument("db_file", help="Path to Milvus .db file")
    parser.add_argument("--collection", "-c", help="Show stats for a specific collection only")
    parser.add_argument("--sample", "-s", type=int, default=0,
                        help="Number of sample rows to display per collection (default: 0)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON instead of formatted text")
    args = parser.parse_args()

    db_path = Path(args.db_file)
    if not db_path.exists():
        print(f"Error: file not found: {args.db_file}", file=sys.stderr)
        return 1

    try:
        from pymilvus import MilvusClient
    except ImportError:
        print("Error: pymilvus is not installed. Install with: pip install 'pymilvus[model]'",
              file=sys.stderr)
        return 1

    file_size = os.path.getsize(db_path)

    client = MilvusClient(uri=str(db_path))
    try:
        collections = client.list_collections()
    except Exception as e:
        print(f"Error listing collections: {e}", file=sys.stderr)
        client.close()
        return 1

    if args.collection:
        if args.collection not in collections:
            print(f"Error: collection '{args.collection}' not found. "
                  f"Available: {collections}", file=sys.stderr)
            client.close()
            return 1
        collections = [args.collection]

    # Gather info
    all_info = []
    for cname in sorted(collections):
        cinfo = get_collection_info(client, cname)
        if args.sample > 0:
            cinfo["sample_data"] = get_sample_data(
                client, cname, args.sample, cinfo.get("fields", [])
            )
        all_info.append(cinfo)

    # JSON output
    if args.json:
        output = {
            "db_file": str(db_path.resolve()),
            "file_size_bytes": file_size,
            "file_size_human": format_size(file_size),
            "num_collections": len(all_info),
            "collections": all_info,
        }
        print(json.dumps(output, indent=2, default=str))
        client.close()
        return 0

    # Formatted output
    print("=" * 70)
    print("  Milvus Database Statistics")
    print("=" * 70)
    print(f"  File: {db_path.resolve()}")
    print(f"  Size: {format_size(file_size)} ({file_size:,} bytes)")
    print(f"  Collections: {len(all_info)}")

    total_rows = 0
    for cinfo in all_info:
        sample_rows = cinfo.pop("sample_data", None) if args.sample > 0 else None
        print_collection_info(cinfo, sample_rows=sample_rows)
        rc = cinfo.get("row_count", 0)
        if isinstance(rc, int):
            total_rows += rc

    if len(all_info) > 1:
        print(f"\n  {'Total rows across all collections:':<40} {total_rows:,}")

    print("\n" + "=" * 70)

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
