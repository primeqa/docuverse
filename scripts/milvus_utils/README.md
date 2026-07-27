# Milvus Utilities

Command-line tools for managing Milvus databases, querying collections, creating indexes, and migrating data to other vector stores.

## Prerequisites

```bash
pip install 'pymilvus[model]'    # All scripts
pip install lancedb pyarrow       # For milvus_copy_to_lancedb.py
pip install pylance               # For LanceDB compaction (optional but recommended)
```

## Scripts

| Script | Description |
|--------|-------------|
| `milvus_db_stats.py` | Display database statistics (collections, schemas, indexes, sample data) |
| `milvus_list_collections.py` | List all collections in a database |
| `milvus_query.py` | Interactive query tool using DocUVerse's SearchEngine |
| `milvus_create_hnsw_index.py` | Create or replace vector indexes (HNSW, IVF_FLAT, etc.) |
| `milvus_copy_to_server.py` | Copy a collection from a .db file to a Milvus server |
| `milvus_copy_to_lancedb.py` | Copy a collection from a .db file to LanceDB |

---

## milvus_copy_to_lancedb.py

Copies a Milvus Lite `.db` collection (including all embeddings) to a LanceDB database. No re-computation of embeddings is needed.

### Quick Start

```bash
# Basic copy
python milvus_copy_to_lancedb.py data.db -o /data/output_lance

# Preview what will happen without copying
python milvus_copy_to_lancedb.py data.db -o /data/output_lance --dry-run

# Copy without building an index (fastest)
python milvus_copy_to_lancedb.py data.db -o /data/output_lance --no-index

# Specify a collection (required if the .db has multiple collections)
python milvus_copy_to_lancedb.py data.db -o /data/output_lance -c my_collection
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `db_file` | (required) | Path to source Milvus .db file |
| `-o`, `--output` | (required) | Target LanceDB directory path |
| `-c`, `--collection` | auto-detect | Source collection name |
| `--target-table` | same as source | Target LanceDB table name |
| `-b`, `--batch-size` | 2000 | Rows per read/write batch |
| `--metric` | dot | Distance metric: `dot`, `cosine`, `L2` |
| `--no-index` | false | Skip building ANN index after copy |
| `--num-partitions` | 256 | IVF_PQ partition count |
| `--num-sub-vectors` | dim/8 | IVF_PQ sub-vector count |
| `--drop-existing` | false | Drop target table if it exists |
| `--resume` | false | Resume an interrupted copy |
| `--no-compact` | false | Skip post-copy fragment compaction |
| `--dry-run` | false | Show plan without copying |

### How It Works

1. **Read** - Streams rows from the Milvus .db file using `query_iterator` in batches
2. **Convert** - Converts each batch to a PyArrow Table with `FixedSizeListArray` for vectors (columnar format, ~7x less memory than Python dicts)
3. **Write** - Appends the PyArrow Table to LanceDB via `table.add()`
4. **Compact** - Merges fragment files created by the batch writes
5. **Index** - Builds an IVF_PQ ANN index over all vectors

### Performance

Tested on a 200K-row collection with 384-dim embeddings:

| Phase | Time | Notes |
|-------|------|-------|
| Milvus read | 99.8% of total | Bottleneck is Milvus Lite query_iterator |
| Arrow convert | 0.2% | Negligible |
| LanceDB write | 0.1% | ~200K rows/s effective write speed |

**Batch size is critical.** Milvus Lite's `query_iterator` has a severe performance cliff above ~3000 rows per batch:

| Batch size | Throughput |
|-----------|-----------|
| 2,000 | ~4,200 rows/s |
| 5,000 | ~135 rows/s |

The default of 2,000 is tuned to stay in the fast range.

**Estimated times** (at ~4,200 rows/s with batch_size=2000):

| Dataset size | Time |
|-------------|------|
| 200K rows | ~1 min |
| 1M rows | ~4 min |
| 12M rows | ~48 min |

### Post-Copy Operations

The script runs these steps automatically after copying (unless disabled):

**Compaction** (`--no-compact` to skip): Each `table.add()` call creates a new Lance fragment file. Compaction merges these into larger files for better read/search performance. Requires `pylance` to be installed.

**ANN Index** (`--no-index` to skip): Builds an IVF_PQ index for approximate nearest neighbor search. For 12M rows with 768-dim vectors, recommended settings:

```bash
python milvus_copy_to_lancedb.py data.db -o /data/lance \
    --num-partitions 512 --num-sub-vectors 96 --metric dot
```

Index building loads all vectors into memory. For 12M x 768-dim, expect ~40-50 GB RAM usage during index creation.

### Resume Support

If the copy is interrupted, use `--resume` to continue from where it left off:

```bash
python milvus_copy_to_lancedb.py data.db -o /data/lance --resume
```

This counts rows already in the target table and skips that many rows from the source iterator before resuming writes.

### Example: Full Wikipedia Migration

```bash
# Step 1: Dry run to verify schema
python milvus_copy_to_lancedb.py \
    experiments/wikipedia_en/wiki-en-granite149m-1024.db \
    -o /data/wiki_lance --dry-run

# Step 2: Copy data without index (fastest)
python milvus_copy_to_lancedb.py \
    experiments/wikipedia_en/wiki-en-granite149m-1024.db \
    -o /data/wiki_lance --no-index

# Step 3: Build index separately (if needed)
# Can be done from Python:
#   import lancedb
#   db = lancedb.connect("/data/wiki_lance")
#   table = db.open_table("wiki-en-granite149m-1024")
#   table.create_index(metric="dot", index_type="IVF_PQ",
#                      num_partitions=512, num_sub_vectors=96)
```

### Schema Mapping

The script automatically maps Milvus field types to PyArrow types:

| Milvus Type | PyArrow Type |
|-------------|-------------|
| BOOL | `bool_()` |
| INT8/16/32/64 | `int8/16/32/64()` |
| FLOAT | `float32()` |
| DOUBLE | `float64()` |
| VARCHAR, STRING | `utf8()` |
| FLOAT_VECTOR(dim=N) | `list_(float32(), list_size=N)` |

Auto-id primary key fields (e.g., `_id`) are automatically detected and excluded from the copy.

### Timing Breakdown

The script prints a timing breakdown at the end of each copy:

```
Timing breakdown:
  Milvus read:      1537.8s (99.8%)
  Arrow convert:       2.7s ( 0.2%)
  LanceDB write:       1.0s ( 0.1%)
```

This helps identify whether the bottleneck is reading (Milvus), conversion (PyArrow), or writing (LanceDB).

---

## milvus_copy_to_server.py

Copies a Milvus Lite `.db` collection to a Milvus server (standalone or distributed), preserving all embeddings.

### Quick Start

```bash
# Basic copy with HNSW index
python milvus_copy_to_server.py data.db -s localhost

# Copy without index, build after (much faster for large datasets)
python milvus_copy_to_server.py data.db -s localhost --no-index

# Resume an interrupted copy
python milvus_copy_to_server.py data.db -s localhost --resume

# Use a named server from config/milvus_servers.json
python milvus_copy_to_server.py data.db -s my_server_name
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `db_file` | (required) | Path to source Milvus .db file |
| `-s`, `--server` | (required) | Target server (host, host:port, or name from milvus_servers.json) |
| `-c`, `--collection` | auto-detect | Source collection name |
| `--target-collection` | same as source | Target collection name |
| `-b`, `--batch-size` | 1000 | Rows per batch |
| `-t`, `--index-type` | HNSW | Index type: HNSW, IVF_FLAT, FLAT |
| `-M` | 128 | HNSW M parameter |
| `--ef-construction` | 128 | HNSW efConstruction parameter |
| `--metric` | IP | Distance metric: IP, L2, COSINE |
| `--no-index` | false | Copy without indexing, build index after all data is loaded |
| `--drop-existing` | false | Drop target collection if it exists |
| `--resume` | false | Resume an interrupted copy |
| `--dry-run` | false | Show plan without copying |

### Performance Notes

Throughput depends heavily on whether the server is indexing during inserts:

| Mode | Typical Speed |
|------|--------------|
| With HNSW index | ~64 rows/s |
| `--no-index` (build after) | ~1,000+ rows/s |

For large datasets, always use `--no-index`. The script will build the index in bulk after all data is copied, which is significantly faster than incremental indexing.

### Named Servers

Servers can be defined in `config/milvus_servers.json`:

```json
{
    "my_server": {
        "host": "192.168.1.100",
        "port": 19530
    }
}
```

Then reference by name: `python milvus_copy_to_server.py data.db -s my_server`

---

## milvus_query.py

Interactive query tool that loads a Milvus or LanceDB database and lets you search using DocUVerse's SearchEngine infrastructure. Supports both backends with automatic engine detection.

### Quick Start

```bash
# Milvus .db file (engine auto-detected from .db extension)
python milvus_query.py experiments/sap/sap.db \
    -m ibm-granite/granite-embedding-30m-english

# LanceDB directory (engine auto-detected from .lance files)
python milvus_query.py /data/wiki_lance \
    -m ibm-granite/granite-embedding-30m-english

# Explicit engine type
python milvus_query.py /data/wiki_lance \
    -m ibm-granite/granite-embedding-30m-english --engine lancedb

# Config mode: use a DocUVerse YAML config (works with any engine)
python milvus_query.py --config path/to/config.yaml

# Specify GPU device and result count
python milvus_query.py data.db -m granite-embedding-30m-english -d 1 -k 10
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `db_file` | (positional) | Path to Milvus .db file or LanceDB directory (not needed with `--config`) |
| `--config` | | Path to DocUVerse YAML config (config mode) |
| `--model`, `-m` | | Embedding model name |
| `--engine`, `-e` | auto-detect | Engine type: `milvus` or `lancedb` |
| `--device`, `-d` | | CUDA device index (e.g. `0`, `1`, `cpu`) |
| `--top_k`, `-k` | 5 | Number of results to return |
| `--collection`, `-c` | auto-detect | Collection/table name |

### Engine Auto-Detection

When `--engine` is not specified, the script infers the engine from the path:

| Path | Detected Engine |
|------|----------------|
| `*.db` file | `milvus` |
| Directory containing `.lance` files or `_versions/` | `lancedb` |
| Directory containing a `lancedb_data/` subdirectory | `lancedb` |

If auto-detection fails, specify `--engine` explicitly.

### LanceDB Path Handling

The script handles two LanceDB directory layouts:

| Layout | Example | How it's handled |
|--------|---------|-----------------|
| Created by `ingest_and_test.py` | `/data/project/lancedb_data/<table>` | Pass the parent: `/data/project` |
| Created by `milvus_copy_to_lancedb.py` | `/data/wiki_lance/<table>` | Pass the directory: `/data/wiki_lance` |

Both work automatically. The script detects whether a `lancedb_data/` subdirectory exists and adjusts accordingly.

The vector column name is also auto-detected from the table schema. Tables copied from Milvus (where the vector field may be named `embeddings`) work without any extra configuration.

### Modes of Operation

**Direct mode** — provide a database path and model on the command line:

```bash
# Milvus
python milvus_query.py experiments/sap/sap.db \
    -m ibm-granite/granite-embedding-125m-english

# LanceDB (created by milvus_copy_to_lancedb.py)
python milvus_query.py /data/wiki_lance \
    -m ibm-granite/granite-embedding-30m-english

# LanceDB (created by ingest_and_test.py, has lancedb_data/ subdir)
python milvus_query.py /tmp/lancedb_test \
    -m ibm-granite/granite-embedding-30m-english
```

If the database contains multiple collections/tables, specify one with `-c`:

```bash
python milvus_query.py data.db -m granite-embedding-30m-english -c my_collection
```

**Config mode** — use an existing DocUVerse YAML config file. Works with any `db_engine` (Milvus, LanceDB, etc.):

```bash
python milvus_query.py --config experiments/sap/sap_milvus_dense.granite125.flat.file.yaml
python milvus_query.py --config experiments/sap/sap_lancedb.yaml -k 3
```

### Example: Query a Copied Wikipedia Database

```bash
# Copy from Milvus to LanceDB
python milvus_copy_to_lancedb.py experiments/wikipedia_en/wiki-en-granite149m-1024.db \
    -o /data/wiki_lance --metric cosine

# Query it
python milvus_query.py /data/wiki_lance \
    -m ibm-granite/granite-embedding-149m-english -d 1
```

### Example Session

```
$ python milvus_query.py /data/wiki_lance -m granite-embedding-30m-english
Auto-detected engine: lancedb
Opening database: /data/wiki_lance
Initializing search engine (loading model)...
Initialized in 2.93s

Collection: wiki-en-granite149m-1024
Model: ibm-granite/granite-embedding-30m-english
Top-k: 5

Ready. Enter a question (or 'quit' to exit):

Query> how to reset password

  [1] Score: 0.4884  ID: sap_168729.txt-1175-1960
      Title: Resetting Password of Login Accounts | SAP Help Portal
      Text:  You've successfully reset the passwords of the login accounts...

  [2] Score: 0.4859  ID: sap_168729.txt-1960-2706
      Title: Resetting Password of Login Accounts | SAP Help Portal
      Text:  Updating Login Accounts...

Query> quit
Goodbye.
```

---

## milvus_db_stats.py

Displays comprehensive statistics about a Milvus .db file.

```bash
python milvus_db_stats.py data.db
```

Shows: file size, collections, field schemas, index types, row counts, and sample data.

---

## milvus_create_hnsw_index.py

Creates or replaces vector indexes on a Milvus collection.

```bash
# Create HNSW index on a server collection
python milvus_create_hnsw_index.py --server localhost -c my_collection

# Create IVF_FLAT index on a .db file
python milvus_create_hnsw_index.py data.db -c my_collection --index-type IVF_FLAT
```

Note: Milvus Lite (.db files) only supports FLAT, IVF_FLAT, and AUTOINDEX. HNSW requires a Milvus server.

---

## milvus_list_collections.py

Lists all collections in a Milvus database.

```bash
python milvus_list_collections.py data.db
```
