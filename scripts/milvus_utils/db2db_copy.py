#!/usr/bin/env python3
"""Copy data between vector database formats: Milvus, LanceDB, FAISS.

Reads all data (including embeddings) from a source database and writes it to
a target format. No re-computation of embeddings is needed.

Supported formats:
  milvus   - Milvus Lite .db files (read only)
  lancedb  - LanceDB directory with PyArrow tables
  faiss    - FAISS .index file + metadata pickle

The source and target formats are auto-detected from the paths, or can be
specified explicitly with --fromdb and --todb.

Usage:
    db2db-copy <input> -o <output> [--fromdb TYPE] [--todb TYPE]

Examples:
    # Milvus -> LanceDB (auto-detected)
    db2db-copy data.db -o /data/lance_db

    # Milvus -> FAISS
    db2db-copy data.db --todb faiss -o /data/faiss_out

    # LanceDB -> FAISS
    db2db-copy /data/lance_db --todb faiss -o /data/faiss_out

    # FAISS -> LanceDB
    db2db-copy /data/faiss_out --todb lancedb -o /data/lance_db

    # Explicit types
    db2db-copy data.db --fromdb milvus --todb faiss -o /data/faiss_out

    # Dry-run
    db2db-copy data.db -o /data/lance_db --dry-run
"""

import argparse
import glob as globmod
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
except ImportError:
    print("Error: pyarrow is required. Install with: pip install pyarrow",
          file=sys.stderr)
    sys.exit(1)

from tqdm.auto import tqdm

_DB_TYPES = ("milvus", "lancedb", "faiss")

# ======================================================================
# Constants / Milvus type mappings
# ======================================================================

_VECTOR_TYPES = {100, 101, 102, 103, 104}

_DTYPE_NAMES = {
    1: "BOOL", 2: "INT8", 3: "INT16", 4: "INT32", 5: "INT64",
    10: "FLOAT", 11: "DOUBLE", 20: "STRING", 21: "VARCHAR",
    22: "ARRAY", 23: "JSON",
    100: "BINARY_VECTOR", 101: "FLOAT_VECTOR", 102: "FLOAT16_VECTOR",
    103: "BFLOAT16_VECTOR", 104: "SPARSE_FLOAT_VECTOR",
}

_MILVUS_TO_ARROW = {
    1: pa.bool_(),
    2: pa.int8(),
    3: pa.int16(),
    4: pa.int32(),
    5: pa.int64(),
    10: pa.float32(),
    11: pa.float64(),
    20: pa.utf8(),
    21: pa.utf8(),
}

# ======================================================================
# Auto-detection
# ======================================================================


def detect_db_type(path):
    """Auto-detect database type from a file/directory path.

    Returns 'milvus', 'lancedb', 'faiss', or None.
    """
    p = Path(path)

    # Milvus: single .db file
    if p.is_file() and p.suffix == ".db":
        return "milvus"

    if p.is_dir():
        # LanceDB: directory with .lance files or _versions/
        lance_markers = (list(p.glob("**/*.lance"))[:1]
                         + list(p.glob("**/_versions"))[:1])
        if lance_markers:
            return "lancedb"
        if p.name == "lancedb_data" or (p / "lancedb_data").is_dir():
            return "lancedb"

        # FAISS: directory with faiss_data/ subdir, or is faiss_data/ itself
        if (p / "faiss_data").is_dir():
            return "faiss"
        if p.name == "faiss_data" and globmod.glob(os.path.join(str(p), "*.index")):
            return "faiss"
        # Also detect if the dir itself contains .index files
        if globmod.glob(os.path.join(str(p), "*.index")):
            return "faiss"

    return None


# ======================================================================
# Unified fields_info format
# ======================================================================


def make_fields_info(scalar_names, vector_field_name, vector_dim):
    """Build a fields_info list (same format as parse_fields) from simple names.

    Used by LanceDB and FAISS readers to produce a uniform schema description.
    """
    fields = []
    for name in scalar_names:
        fields.append({
            "name": name,
            "type_int": 21,  # VARCHAR
            "type_name": "VARCHAR",
            "is_primary": (name == "id"),
            "auto_id": False,
            "is_vector": False,
        })
    fields.append({
        "name": vector_field_name,
        "type_int": 101,  # FLOAT_VECTOR
        "type_name": "FLOAT_VECTOR",
        "is_primary": False,
        "auto_id": False,
        "is_vector": True,
        "dim": vector_dim,
    })
    return fields


# ======================================================================
# Source: Milvus
# ======================================================================


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


def parse_fields(desc):
    """Parse field info from Milvus describe_collection output."""
    fields_info = []
    for f in desc.get("fields", []):
        params = f.get("params", {})
        info = {
            "name": f["name"],
            "type_int": f["type"],
            "type_name": _DTYPE_NAMES.get(f["type"], f"UNKNOWN({f['type']})"),
            "is_primary": f.get("is_primary", False),
            "auto_id": f.get("auto_id", False),
            "is_vector": f["type"] in _VECTOR_TYPES,
        }
        if "dim" in params:
            info["dim"] = int(params["dim"])
        if "max_length" in params:
            info["max_length"] = int(params["max_length"])
        fields_info.append(info)
    return fields_info


def discover_milvus(input_path, collection_name):
    """Connect to Milvus and return (source, collection, fields_info, total_rows, output_fields, vector_field_name, vector_dim)."""
    from pymilvus import MilvusClient
    source = MilvusClient(uri=str(input_path))
    collection = resolve_collection(source, collection_name)
    desc = source.describe_collection(collection)
    fields_info = parse_fields(desc)
    stats = source.get_collection_stats(collection)
    total_rows = stats.get("row_count", 0)
    output_fields = [f["name"] for f in fields_info if not f.get("auto_id")]
    vector_fields = [f for f in fields_info if f["is_vector"]]
    vector_dim = vector_fields[0].get("dim", 0) if vector_fields else 0
    vector_field_name = vector_fields[0]["name"] if vector_fields else None
    return source, collection, fields_info, total_rows, output_fields, vector_field_name, vector_dim


def read_milvus_batches(source, src_collection, output_fields,
                        batch_size, total_rows, skip_rows=0):
    """Yield (raw_batch, batch_len, read_time) from Milvus query_iterator."""
    print("Initializing source iterator (this may take a while for large databases)...",
          flush=True)
    t_iter = time.time()
    iterator = source.query_iterator(
        collection_name=src_collection,
        batch_size=batch_size,
        output_fields=output_fields,
    )
    print(f"Iterator ready in {time.time() - t_iter:.1f}s.", flush=True)

    if skip_rows > 0:
        skipped = 0
        skip_pbar = tqdm(total=skip_rows, desc="Skipping rows", unit="rows")
        while skipped < skip_rows:
            batch = iterator.next()
            if not batch:
                break
            skipped += len(batch)
            skip_pbar.update(len(batch))
        skip_pbar.close()

    first = True
    while True:
        t1 = time.time()
        batch = iterator.next()
        read_time = time.time() - t1
        if not batch:
            break
        if first:
            print(f"First batch received ({len(batch)} rows). Copying...", flush=True)
            first = False
        yield batch, len(batch), read_time

    iterator.close()


# ======================================================================
# Source: LanceDB
# ======================================================================


def _resolve_table(db, table_name):
    """Discover or validate a LanceDB table name."""
    try:
        resp = db.list_tables()
        tables = resp.tables if hasattr(resp, 'tables') else list(resp)
    except AttributeError:
        tables = list(db.table_names())

    if not tables:
        print("Error: no tables found in LanceDB database.", file=sys.stderr)
        sys.exit(1)
    if table_name:
        if table_name not in tables:
            print(f"Error: table '{table_name}' not found. Available: {tables}",
                  file=sys.stderr)
            sys.exit(1)
        return table_name
    elif len(tables) == 1:
        return tables[0]
    else:
        print(f"Multiple tables found: {tables}")
        print("Please specify one with --collection / -c", file=sys.stderr)
        sys.exit(1)


def _detect_vector_column(schema):
    """Find the vector column name and dimension from a PyArrow schema."""
    for field in schema:
        if pa.types.is_fixed_size_list(field.type):
            return field.name, field.type.list_size
        if pa.types.is_list(field.type):
            return field.name, None  # variable-size list, dim unknown
    return None, None


def discover_lancedb(input_path, collection_name):
    """Connect to LanceDB and return (db, table_name, fields_info, total_rows, output_fields, vector_field_name, vector_dim)."""
    import lancedb
    db = lancedb.connect(str(input_path))
    table_name = _resolve_table(db, collection_name)
    table = db.open_table(table_name)
    total_rows = table.count_rows()

    schema = table.schema
    vector_field_name, vector_dim = _detect_vector_column(schema)

    scalar_names = [f.name for f in schema
                    if not pa.types.is_fixed_size_list(f.type)
                    and not pa.types.is_list(f.type)]
    output_fields = scalar_names + ([vector_field_name] if vector_field_name else [])

    fields_info = make_fields_info(scalar_names, vector_field_name, vector_dim or 0)
    return db, table_name, fields_info, total_rows, output_fields, vector_field_name, vector_dim


def read_lancedb_batches(db, table_name, batch_size, total_rows, skip_rows=0):
    """Yield (raw_batch, batch_len, read_time) from a LanceDB table."""
    table = db.open_table(table_name)

    print("Reading LanceDB table into memory...", flush=True)
    t0 = time.time()
    arrow_table = table.to_arrow()
    print(f"Table loaded in {time.time() - t0:.1f}s ({len(arrow_table)} rows).", flush=True)

    col_names = [f.name for f in arrow_table.schema]
    start = skip_rows

    while start < len(arrow_table):
        t1 = time.time()
        end = min(start + batch_size, len(arrow_table))
        slice_table = arrow_table.slice(start, end - start)

        # Convert arrow slice to list of dicts
        cols = {name: slice_table.column(name).to_pylist() for name in col_names}
        batch = [dict(zip(col_names, vals)) for vals in zip(*cols.values())]

        read_time = time.time() - t1
        yield batch, len(batch), read_time
        start = end


# ======================================================================
# Source: FAISS
# ======================================================================


def _find_faiss_index(input_path, index_name):
    """Locate FAISS .index and _metadata.pkl files.

    Returns (index_path, metadata_path, resolved_name).
    """
    p = Path(input_path)

    # If path is a directory, look for faiss_data/ subdir or .index files
    if p.is_dir():
        faiss_dir = p / "faiss_data" if (p / "faiss_data").is_dir() else p
        index_files = sorted(faiss_dir.glob("*.index"))

        if index_name:
            idx_path = faiss_dir / f"{index_name}.index"
            meta_path = faiss_dir / f"{index_name}_metadata.pkl"
            if not idx_path.exists():
                print(f"Error: {idx_path} not found.", file=sys.stderr)
                sys.exit(1)
            return str(idx_path), str(meta_path), index_name

        if len(index_files) == 0:
            print(f"Error: no .index files found in {faiss_dir}", file=sys.stderr)
            sys.exit(1)
        elif len(index_files) == 1:
            idx_path = index_files[0]
            name = idx_path.stem
            meta_path = faiss_dir / f"{name}_metadata.pkl"
            return str(idx_path), str(meta_path), name
        else:
            names = [f.stem for f in index_files]
            print(f"Multiple FAISS indices found: {names}")
            print("Please specify one with --collection / -c", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: FAISS source must be a directory, got: {input_path}",
              file=sys.stderr)
        sys.exit(1)


def discover_faiss(input_path, collection_name):
    """Load FAISS index+metadata and return (index, metadata, index_name, fields_info, total_rows, output_fields, vector_field_name, vector_dim)."""
    import faiss as faiss_lib

    index_path, metadata_path, index_name = _find_faiss_index(input_path, collection_name)

    print(f"Loading FAISS index from {index_path}...", flush=True)
    index = faiss_lib.read_index(index_path)

    if not os.path.exists(metadata_path):
        print(f"Error: metadata file not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    id_map = metadata['id_map']
    metadata_store = metadata['metadata_store']
    total_rows = index.ntotal
    vector_dim = index.d
    vector_field_name = "vector"

    # Determine scalar field names from metadata
    scalar_names = ["id"]
    if metadata_store:
        sample_meta = next(iter(metadata_store.values()))
        for k in sample_meta:
            if k not in scalar_names:
                scalar_names.append(k)

    output_fields = scalar_names + [vector_field_name]
    fields_info = make_fields_info(scalar_names, vector_field_name, vector_dim)

    return index, id_map, metadata_store, index_name, fields_info, total_rows, output_fields, vector_field_name, vector_dim


def read_faiss_batches(index, id_map, metadata_store, vector_field_name,
                       batch_size, total_rows, skip_rows=0):
    """Yield (raw_batch, batch_len, read_time) from FAISS index + metadata."""
    scalar_keys = None
    if metadata_store:
        sample = next(iter(metadata_store.values()))
        scalar_keys = list(sample.keys())

    start = skip_rows
    while start < total_rows:
        t1 = time.time()
        end = min(start + batch_size, total_rows)
        n = end - start

        # Reconstruct vectors
        vectors = index.reconstruct_n(start, n)

        batch = []
        for i in range(n):
            doc_id = id_map[start + i]
            row = {"id": doc_id, vector_field_name: vectors[i].tolist()}
            meta = metadata_store.get(doc_id, {})
            for k in (scalar_keys or []):
                if k in meta:
                    row[k] = meta[k]
            batch.append(row)

        read_time = time.time() - t1
        yield batch, len(batch), read_time
        start = end


# ======================================================================
# Arrow helpers (for LanceDB writer)
# ======================================================================


def build_arrow_schema(fields_info):
    """Build a PyArrow schema from fields_info."""
    arrow_fields = []
    for f in fields_info:
        if f.get("auto_id"):
            continue
        if f["is_vector"]:
            dim = f.get("dim", 0)
            arrow_fields.append(
                pa.field(f["name"], pa.list_(pa.float32(), list_size=dim))
            )
        else:
            arrow_type = _MILVUS_TO_ARROW.get(f["type_int"], pa.utf8())
            arrow_fields.append(pa.field(f["name"], arrow_type))
    return pa.schema(arrow_fields)


def batch_to_arrow(batch, fields_info, schema):
    """Convert a raw batch (list of dicts) to a PyArrow Table."""
    active_fields = [f for f in fields_info if not f.get("auto_id")]
    arrays = []
    for f in active_fields:
        name = f["name"]
        if f["is_vector"]:
            dim = f.get("dim", 0)
            vecs = [row[name] for row in batch]
            flat = np.array(vecs, dtype=np.float32).flatten()
            values = pa.array(flat, type=pa.float32())
            arrays.append(pa.FixedSizeListArray.from_arrays(values, list_size=dim))
        else:
            col_data = [row.get(name) for row in batch]
            arrow_type = _MILVUS_TO_ARROW.get(f["type_int"], pa.utf8())
            if arrow_type == pa.utf8():
                col_data = [str(v) if v is not None else None for v in col_data]
            arrays.append(pa.array(col_data, type=arrow_type))
    return pa.table(arrays, schema=schema)


# ======================================================================
# LanceDB writer
# ======================================================================


def write_lancedb(output_path, table_name, fields_info, arrow_schema,
                  vector_field_name, vector_dim, batch_generator,
                  total_rows, args):
    """Write data to LanceDB. Returns rows copied, or -1 on error."""
    try:
        import lancedb
    except ImportError:
        print("Error: lancedb is required for --todb lancedb. "
              "Install with: pip install lancedb", file=sys.stderr)
        return -1

    db = lancedb.connect(str(output_path))

    try:
        resp = db.list_tables()
        existing_tables = resp.tables if hasattr(resp, 'tables') else list(resp)
    except AttributeError:
        existing_tables = list(db.table_names())

    target_exists = table_name in existing_tables
    already_copied = 0

    if target_exists and args.resume:
        table = db.open_table(table_name)
        already_copied = table.count_rows()
        print(f"\nResuming: target '{table_name}' has {already_copied:,} rows, "
              f"skipping those from source.")
    elif target_exists and args.drop_existing:
        print(f"\nDropping existing target table '{table_name}'...")
        db.drop_table(table_name)
        target_exists = False
    elif target_exists:
        print(f"\nError: target table '{table_name}' already exists. "
              f"Use --drop-existing to overwrite, --resume to continue, "
              f"or --target-name for a different name.", file=sys.stderr)
        return -1

    if not target_exists:
        print(f"\nCreating target table '{table_name}'...")
        table = db.create_table(table_name, schema=arrow_schema)
        print("Table created.")

    remaining = total_rows - already_copied
    if remaining <= 0:
        print(f"\nAll {total_rows:,} rows already in target. Nothing to copy.")
        return 0

    print(f"\nCopying {remaining:,} rows" +
          (f" (skipping first {already_copied:,})..." if already_copied else "..."))

    copied = 0
    read_time = convert_time = write_time = 0.0
    pbar = tqdm(total=remaining, desc="Copying rows", unit="rows")

    for raw_batch, batch_len, r_time in batch_generator:
        read_time += r_time

        t1 = time.time()
        arrow_batch = batch_to_arrow(raw_batch, fields_info, arrow_schema)
        convert_time += time.time() - t1

        t2 = time.time()
        table.add(arrow_batch)
        write_time += time.time() - t2

        copied += batch_len
        pbar.update(batch_len)

    pbar.close()

    total_time = read_time + convert_time + write_time
    if total_time > 0:
        print(f"\nTiming breakdown:")
        print(f"  Source read:      {read_time:8.1f}s ({read_time / total_time * 100:5.1f}%)")
        print(f"  Arrow convert:    {convert_time:8.1f}s ({convert_time / total_time * 100:5.1f}%)")
        print(f"  LanceDB write:    {write_time:8.1f}s ({write_time / total_time * 100:5.1f}%)")

    # Compact
    if not args.no_compact:
        print("\nCompacting fragments (merging small files)...")
        t_compact = time.time()
        try:
            table.optimize.compact_files()
            print(f"Compaction done in {time.time() - t_compact:.1f}s")
            print("Cleaning up old versions...")
            table.cleanup_old_versions()
        except (ImportError, AttributeError):
            try:
                table.compact_files()
                print(f"Compaction done in {time.time() - t_compact:.1f}s")
                print("Cleaning up old versions...")
                table.cleanup_old_versions()
            except ImportError:
                print("Warning: compaction requires pylance. Install with: "
                      "pip install pylance", file=sys.stderr)
                print("Skipping compaction. Data is still usable but "
                      "search performance may be suboptimal.")

    # Build ANN index
    if not args.no_index and vector_field_name:
        nsv = args.num_sub_vectors or (vector_dim // 8)
        print(f"\nBuilding IVF_PQ index (partitions={args.num_partitions}, "
              f"sub_vectors={nsv})...")
        t_idx = time.time()
        table.create_index(
            metric=args.metric,
            vector_column_name=vector_field_name,
            index_type="IVF_PQ",
            num_partitions=args.num_partitions,
            num_sub_vectors=nsv,
            replace=True,
        )
        print(f"Index built in {time.time() - t_idx:.1f}s")

    # Verify
    final_rows = table.count_rows()
    print(f"\nVerification:")
    print(f"  Source rows:  {total_rows:,}")
    print(f"  Target rows:  {final_rows:,}")
    if final_rows == total_rows:
        print("  Status:       OK")
    elif final_rows >= total_rows:
        print(f"  Status:       OK (target has {final_rows - total_rows:,} extra rows, "
              f"possibly from a previous partial copy)")
    else:
        print(f"  Status:       INCOMPLETE ({total_rows - final_rows:,} rows missing)")

    return copied


# ======================================================================
# FAISS writer
# ======================================================================


def write_faiss(output_dir, index_name, fields_info, vector_field_name,
                vector_dim, batch_generator, total_rows, args):
    """Write data to FAISS index + metadata pickle. Returns rows copied, or -1 on error."""
    try:
        import faiss
    except ImportError:
        print("Error: faiss is required for --todb faiss. "
              "Install with: pip install faiss-cpu (or faiss-gpu)", file=sys.stderr)
        return -1

    faiss_dir = os.path.join(output_dir, "faiss_data")
    os.makedirs(faiss_dir, exist_ok=True)

    index_path = os.path.join(faiss_dir, f"{index_name}.index")
    metadata_path = os.path.join(faiss_dir, f"{index_name}_metadata.pkl")

    if os.path.exists(index_path):
        if args.drop_existing:
            os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
        else:
            print(f"\nError: {index_path} already exists. "
                  f"Use --drop-existing to overwrite.", file=sys.stderr)
            return -1

    scalar_fields = [f["name"] for f in fields_info
                     if not f["is_vector"] and not f.get("auto_id")]

    # Create FAISS index
    index_type = args.faiss_index_type
    use_ip = (args.faiss_metric == "IP")

    if index_type == "Flat":
        index = faiss.IndexFlatIP(vector_dim) if use_ip else faiss.IndexFlatL2(vector_dim)
    elif index_type == "IVFFlat":
        quantizer = faiss.IndexFlatIP(vector_dim) if use_ip else faiss.IndexFlatL2(vector_dim)
        metric = faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2
        index = faiss.IndexIVFFlat(quantizer, vector_dim, args.nlist, metric)
    elif index_type == "HNSW":
        index = faiss.IndexHNSWFlat(vector_dim, args.hnsw_m)
    else:
        print(f"Error: unknown FAISS index type: {index_type}", file=sys.stderr)
        return -1

    print(f"\nFAISS index: {index_type} (dim={vector_dim}, metric={args.faiss_metric})")

    id_map = []
    metadata_store = {}

    def _process_batch(raw_batch):
        vecs = np.array([row[vector_field_name] for row in raw_batch],
                        dtype=np.float32)
        for row in raw_batch:
            doc_id = str(row.get("id", ""))
            id_map.append(doc_id)
            meta = {}
            for k in scalar_fields:
                if k in row and k != "id":
                    val = row[k]
                    meta[k] = str(val) if isinstance(val, dict) else val
            metadata_store[doc_id] = meta
        return vecs

    read_time = insert_time = 0.0

    if index_type == "IVFFlat" and not index.is_trained:
        train_size = args.train_size or min(total_rows, 100_000)
        if total_rows > 1_000_000:
            print(f"Warning: IVFFlat training buffers all data in memory. "
                  f"For very large datasets consider --faiss-index-type Flat.",
                  file=sys.stderr)

        print(f"Reading all data and collecting up to {train_size:,} training vectors...")
        training_vectors = []
        all_batches = []
        collected_train = 0

        pbar = tqdm(total=total_rows, desc="Reading data", unit="rows")
        for raw_batch, batch_len, r_time in batch_generator:
            read_time += r_time
            vecs = np.array([row[vector_field_name] for row in raw_batch],
                            dtype=np.float32)
            if collected_train < train_size:
                training_vectors.append(vecs)
                collected_train += len(vecs)
            all_batches.append(raw_batch)
            pbar.update(batch_len)
        pbar.close()

        train_matrix = np.vstack(training_vectors)[:train_size]
        print(f"Training IVFFlat on {len(train_matrix):,} vectors...")
        t_train = time.time()
        index.train(train_matrix)
        print(f"Training done in {time.time() - t_train:.1f}s")
        del train_matrix, training_vectors

        print("Adding vectors to trained index...")
        pbar = tqdm(total=total_rows, desc="Inserting vectors", unit="rows")
        for raw_batch in all_batches:
            t1 = time.time()
            vecs = _process_batch(raw_batch)
            index.add(vecs)
            insert_time += time.time() - t1
            pbar.update(len(raw_batch))
        pbar.close()
        del all_batches
    else:
        pbar = tqdm(total=total_rows, desc="Copying rows", unit="rows")
        for raw_batch, batch_len, r_time in batch_generator:
            read_time += r_time
            t1 = time.time()
            vecs = _process_batch(raw_batch)
            index.add(vecs)
            insert_time += time.time() - t1
            pbar.update(batch_len)
        pbar.close()

    total_time = read_time + insert_time
    if total_time > 0:
        print(f"\nTiming breakdown:")
        print(f"  Source read:      {read_time:8.1f}s ({read_time / total_time * 100:5.1f}%)")
        print(f"  FAISS insert:     {insert_time:8.1f}s ({insert_time / total_time * 100:5.1f}%)")

    print(f"\nSaving FAISS index ({index.ntotal:,} vectors) to {index_path}")
    faiss.write_index(index, index_path)

    metadata = {'id_map': id_map, 'metadata_store': metadata_store}
    print(f"Saving metadata ({len(metadata_store):,} entries) to {metadata_path}")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nVerification:")
    print(f"  Source rows:      {total_rows:,}")
    print(f"  FAISS vectors:    {index.ntotal:,}")
    print(f"  Metadata entries: {len(metadata_store):,}")
    print(f"  ID map length:    {len(id_map):,}")
    if index.ntotal == total_rows:
        print("  Status:           OK")
    else:
        print(f"  Status:           MISMATCH (expected {total_rows:,})")

    return index.ntotal


# ======================================================================
# CLI
# ======================================================================


def display_plan(args, src_name, target_name, total_rows,
                 fields_info, output_fields, vector_field_name, vector_dim):
    """Print the copy plan."""
    print(f"\nSource format:      {args.fromdb}")
    print(f"Target format:      {args.todb}")
    print(f"Source name:        {src_name}  ({total_rows:,} rows)")
    print(f"Target name:        {target_name}")
    print(f"Output path:        {Path(args.output).resolve()}")
    print(f"Batch size:         {args.batch_size:,}")

    if args.todb == "lancedb":
        print(f"Metric:             {args.metric}")
        if not args.no_index:
            nsv = args.num_sub_vectors or (vector_dim // 8 if vector_dim else 0)
            print(f"Index:              IVF_PQ (partitions={args.num_partitions}, "
                  f"sub_vectors={nsv})")
        else:
            print("Index:              none (--no-index)")
    elif args.todb == "faiss":
        print(f"FAISS index type:   {args.faiss_index_type}")
        print(f"FAISS metric:       {args.faiss_metric}")
        if args.faiss_index_type == "IVFFlat":
            print(f"IVF nlist:          {args.nlist}")
            ts = args.train_size or min(total_rows, 100_000)
            print(f"Training vectors:   {ts:,}")
        elif args.faiss_index_type == "HNSW":
            print(f"HNSW M:             {args.hnsw_m}")

    print(f"\nSchema ({len(fields_info)} fields):")
    for f in fields_info:
        details = []
        if f.get("is_primary"):
            details.append("PRIMARY KEY")
        if f.get("auto_id"):
            details.append("auto_id (skip in copy)")
        if f.get("is_vector"):
            details.append(f"dim={f.get('dim', '?')}")
        if "max_length" in f:
            details.append(f"max_length={f['max_length']}")
        detail_str = ", ".join(details) if details else ""
        print(f"  {f['name']:<25} {f['type_name']:<22} {detail_str}")

    print(f"\nFields to copy:     {output_fields}")
    print(f"Vector field:       {vector_field_name} (dim={vector_dim})")

    vec_bytes = total_rows * (vector_dim or 0) * 4
    print(f"Est. vector data:   {vec_bytes / (1024 ** 3):.1f} GB")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Copy data between vector database formats: Milvus, LanceDB, FAISS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Milvus -> LanceDB (auto-detected)
  db2db-copy data.db -o /data/lance_db

  # Milvus -> FAISS
  db2db-copy data.db --todb faiss -o /data/faiss_out

  # LanceDB -> FAISS
  db2db-copy /data/lance_db --todb faiss -o /data/faiss_out

  # FAISS -> LanceDB
  db2db-copy /data/faiss_out --todb lancedb -o /data/lance_db

  # Explicit source and target types
  db2db-copy data.db --fromdb milvus --todb faiss -o /data/faiss_out

  # Dry-run
  db2db-copy data.db -o /data/lance_db --dry-run
""",
    )

    # Common arguments
    parser.add_argument("input", help="Source path (Milvus .db file, LanceDB dir, or FAISS dir)")
    parser.add_argument("--output", "-o", required=True,
                        help="Target output path (directory)")
    parser.add_argument("--fromdb", default=None, choices=_DB_TYPES,
                        help="Source format (auto-detected if omitted)")
    parser.add_argument("--todb", default=None, choices=_DB_TYPES,
                        help="Target format (auto-detected from output path if omitted)")
    # Hidden alias for backwards compat
    parser.add_argument("--target", "-T", default=None, choices=_DB_TYPES,
                        help=argparse.SUPPRESS)
    parser.add_argument("--collection", "-c", default=None,
                        help="Source collection/table/index name (auto-detected if only one)")
    parser.add_argument("--target-name", default=None,
                        help="Target table/index name (defaults to source name)")
    parser.add_argument("--batch-size", "-b", type=int, default=2000,
                        help="Rows per batch for read/insert (default: 2000)")
    parser.add_argument("--drop-existing", action="store_true",
                        help="Drop/overwrite target if it already exists")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous interrupted copy (LanceDB target only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without copying")
    parser.add_argument("--no-index", action="store_true",
                        help="Skip building vector index after copy")

    # LanceDB target options
    lance_group = parser.add_argument_group("LanceDB target options")
    lance_group.add_argument("--metric", default="dot",
                             choices=["dot", "cosine", "L2"],
                             help="Distance metric for ANN index (default: dot)")
    lance_group.add_argument("--num-partitions", type=int, default=256,
                             help="IVF_PQ num_partitions (default: 256)")
    lance_group.add_argument("--num-sub-vectors", type=int, default=None,
                             help="IVF_PQ num_sub_vectors (default: dim/8)")
    lance_group.add_argument("--no-compact", action="store_true",
                             help="Skip compaction after copy")

    # FAISS target options
    faiss_group = parser.add_argument_group("FAISS target options")
    faiss_group.add_argument("--faiss-index-type", default="Flat",
                             choices=["Flat", "IVFFlat", "HNSW"],
                             help="FAISS index type (default: Flat)")
    faiss_group.add_argument("--faiss-metric", default="IP",
                             choices=["IP", "L2"],
                             help="FAISS distance metric (default: IP)")
    faiss_group.add_argument("--nlist", type=int, default=100,
                             help="IVFFlat cluster count (default: 100)")
    faiss_group.add_argument("--hnsw-m", type=int, default=32,
                             help="HNSW M parameter (default: 32)")
    faiss_group.add_argument("--train-size", type=int, default=None,
                             help="Vectors for IVFFlat training "
                                  "(default: min(total_rows, 100000))")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle --target as alias for --todb
    if args.target and not args.todb:
        args.todb = args.target

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    # Auto-detect source type
    if not args.fromdb:
        args.fromdb = detect_db_type(str(input_path))
        if not args.fromdb:
            print(f"Error: cannot auto-detect source format for '{args.input}'. "
                  f"Use --fromdb to specify.", file=sys.stderr)
            return 1
        print(f"Auto-detected source: {args.fromdb}")

    # Auto-detect target type
    if not args.todb:
        detected = detect_db_type(str(args.output))
        if detected:
            args.todb = detected
            print(f"Auto-detected target: {args.todb}")
        else:
            # Default to lancedb if output doesn't exist yet
            args.todb = "lancedb"
            print(f"Target not detected, defaulting to: {args.todb}")

    # Validate
    if args.fromdb == args.todb:
        print(f"Warning: source and target are both '{args.fromdb}'. "
              f"This will create a copy in the same format.", file=sys.stderr)

    if args.todb == "faiss" and args.resume:
        print("Error: --resume is not supported for FAISS target "
              "(index files are written atomically).", file=sys.stderr)
        return 1

    if args.todb == "milvus":
        print("Error: writing to Milvus is not supported. "
              "Use --todb lancedb or --todb faiss.", file=sys.stderr)
        return 1

    # ---- Discover source ----
    print(f"Source:  {input_path.resolve()}  ({args.fromdb})")

    src_name = None
    source_handle = None  # Milvus client or LanceDB db (for cleanup)

    if args.fromdb == "milvus":
        from pymilvus import MilvusClient
        source_client, src_name, fields_info, total_rows, output_fields, vector_field_name, vector_dim = \
            discover_milvus(str(input_path), args.collection)
        source_handle = source_client

    elif args.fromdb == "lancedb":
        db, src_name, fields_info, total_rows, output_fields, vector_field_name, vector_dim = \
            discover_lancedb(str(input_path), args.collection)
        source_handle = db

    elif args.fromdb == "faiss":
        faiss_index, faiss_id_map, faiss_meta_store, src_name, fields_info, total_rows, output_fields, vector_field_name, vector_dim = \
            discover_faiss(str(input_path), args.collection)

    else:
        print(f"Error: unsupported source format: {args.fromdb}", file=sys.stderr)
        return 1

    target_name = args.target_name or src_name

    # Display plan
    display_plan(args, src_name, target_name, total_rows,
                 fields_info, output_fields, vector_field_name, vector_dim)

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        if args.fromdb == "milvus" and source_handle:
            source_handle.close()
        return 0

    # ---- Determine skip_rows for resume ----
    skip_rows = 0
    if args.todb == "lancedb" and args.resume:
        try:
            import lancedb
            db_out = lancedb.connect(str(args.output))
            try:
                resp = db_out.list_tables()
                existing = resp.tables if hasattr(resp, 'tables') else list(resp)
            except AttributeError:
                existing = list(db_out.table_names())
            if target_name in existing:
                t = db_out.open_table(target_name)
                skip_rows = t.count_rows()
        except Exception:
            pass

    # ---- Create batch generator ----
    if args.fromdb == "milvus":
        batch_gen = read_milvus_batches(source_client, src_name, output_fields,
                                        args.batch_size, total_rows,
                                        skip_rows=skip_rows)
    elif args.fromdb == "lancedb":
        batch_gen = read_lancedb_batches(source_handle, src_name,
                                         args.batch_size, total_rows,
                                         skip_rows=skip_rows)
    elif args.fromdb == "faiss":
        batch_gen = read_faiss_batches(faiss_index, faiss_id_map, faiss_meta_store,
                                       vector_field_name, args.batch_size, total_rows,
                                       skip_rows=skip_rows)

    # ---- Build arrow schema (for LanceDB writer) ----
    arrow_schema = build_arrow_schema(fields_info)

    # ---- Dispatch to writer ----
    t0 = time.time()
    if args.todb == "lancedb":
        result = write_lancedb(args.output, target_name, fields_info,
                               arrow_schema, vector_field_name, vector_dim,
                               batch_gen, total_rows, args)
    elif args.todb == "faiss":
        result = write_faiss(args.output, target_name, fields_info,
                             vector_field_name, vector_dim,
                             batch_gen, total_rows, args)
    else:
        result = -1

    # Cleanup
    if args.fromdb == "milvus" and source_handle:
        source_handle.close()

    elapsed = time.time() - t0
    if result < 0:
        return 1

    rate = result / elapsed if elapsed > 0 else 0
    print(f"\nDone. Copied {result:,} rows ({args.fromdb} -> {args.todb}) "
          f"in {elapsed:.1f}s ({rate:,.0f} rows/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
