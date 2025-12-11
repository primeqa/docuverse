# Compute Embeddings from JSONL

This script (`compute_embeddings_from_jsonl.py`) processes JSONL files to compute document embeddings with advanced text tiling support.

## Features

- **Flexible field extraction**: Use jq-like dot notation to extract text and IDs from nested JSON structures
- **Text tiling**: Split long documents into overlapping chunks using the TextTiler class
- **Sentence-aligned tiling**: Optionally align tiles on sentence boundaries for better semantic coherence
- **Truncation mode**: Simple truncation for models with fixed context windows
- **Title handling**: Configure how document titles are incorporated into tiles
- **Batch processing**: Efficient batch encoding with configurable batch sizes
- **Multiple devices**: Support for CUDA, CPU, and MPS (Apple Silicon)
- **Multiple output formats**: Save to pickle files or Milvus vector databases

## Installation

Make sure you have the required dependencies:

```bash
pip install sentence-transformers transformers torch numpy tqdm
```

For sentence-aligned tiling:
```bash
pip install pyizumo
```

For Milvus database output:
```bash
pip install pymilvus
```

## Basic Usage

### Simple embedding computation

```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --text_field text \
    --id_field id
```

### With nested field paths

If your JSONL has nested structures like:
```json
{"document": {"content": "text here", "doc_id": "123"}}
```

Use dot notation:
```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model BAAI/bge-base-en-v1.5 \
    --text_field document.content \
    --id_field document.doc_id
```

## Text Tiling

### Enable tiling with sentence alignment

```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model intfloat/e5-base-v2 \
    --text_field text --id_field id \
    --tile \
    --max_length 512 \
    --stride 128 \
    --sentence_aligned
```

### Tiling without sentence alignment

For faster processing (no sentence parsing):
```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model sentence-transformers/all-mpnet-base-v2 \
    --text_field text --id_field id \
    --tile \
    --max_length 512 \
    --stride 128
```

### Character-based tiling

Instead of token-based:
```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --text_field text --id_field id \
    --tile \
    --max_length 2048 \
    --stride 512 \
    --count_type char
```

## Title Handling

When you have document titles, you can control how they're added to tiles:

```bash
# Add title to all tiles
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model BAAI/bge-base-en-v1.5 \
    --text_field document.text \
    --id_field document.id \
    --title_field document.title \
    --tile --max_length 512 \
    --title_handling all

# Add title only to first tile
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model BAAI/bge-base-en-v1.5 \
    --text_field document.text \
    --id_field document.id \
    --title_field document.title \
    --tile --max_length 512 \
    --title_handling first

# No titles (default)
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model BAAI/bge-base-en-v1.5 \
    --text_field document.text \
    --id_field document.id \
    --tile --max_length 512 \
    --title_handling none
```

## Truncation Mode

For simple truncation without tiling:

```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --text_field text --id_field id \
    --truncate \
    --max_length 512
```

## Milvus Database Output

Save embeddings directly to a Milvus vector database instead of pickle files. This is useful for:
- **Immediate vector search**: Query embeddings right after creation
- **Structured storage**: Stores id, text, and embeddings together
- **File-based database**: No server required, uses local .db file

### Basic Milvus output

```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl embeddings.db \
    --model BAAI/bge-base-en-v1.5 \
    --text_field text --id_field id \
    --output_format milvus
```

### With text tiling

```bash
python scripts/compute_embeddings_from_jsonl.py \
    documents.jsonl documents.db \
    --model intfloat/e5-base-v2 \
    --text_field document.content \
    --id_field document.id \
    --output_format milvus \
    --tile --max_length 512 --stride 128
```

### Milvus Collection Structure

The script creates a Milvus collection with:
- **Collection name**: Derived from the output filename (e.g., `embeddings.db` → `embeddings`)
- **Schema**:
  - `_id`: Auto-generated primary key (INT64)
  - `id`: Your document/tile ID (VARCHAR, max 1000 chars)
  - `text`: Document text (VARCHAR, max 65535 chars)
  - `embeddings`: Vector embeddings (FLOAT_VECTOR)

### Querying a Milvus database

#### Using the provided query script

The repository includes a helper script for querying Milvus databases:

```bash
# Semantic search
python scripts/query_milvus_embeddings.py embeddings.db "machine learning" \
    --model BAAI/bge-base-en-v1.5 --top_k 5

# Query by document ID
python scripts/query_milvus_embeddings.py embeddings.db --query_id doc123

# List all documents
python scripts/query_milvus_embeddings.py embeddings.db --list

# Show statistics
python scripts/query_milvus_embeddings.py embeddings.db --stats
```

#### Using Python directly

```python
from pymilvus import MilvusClient

# Open the database
client = MilvusClient(uri='embeddings.db')

# List collections
collections = client.list_collections()
print(f'Collections: {collections}')

# Get collection stats
stats = client.get_collection_stats('embeddings')
print(f'Total documents: {stats["row_count"]}')

# Query by ID
results = client.query(
    collection_name='embeddings',
    filter='id == "doc123"',
    output_fields=['id', 'text', 'embeddings']
)

# Vector search (find similar documents)
results = client.search(
    collection_name='embeddings',
    data=[query_embedding],  # Your query vector
    limit=10,
    output_fields=['id', 'text']
)

client.close()
```

## Model-Specific Features

### Models with prompts (e.g., E5, Instructor)

```bash
# For E5 models - use "passage" prompt for documents
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model intfloat/e5-base-v2 \
    --text_field text --id_field id \
    --prompt_name passage

# For queries
python scripts/compute_embeddings_from_jsonl.py \
    queries.jsonl query_embeddings.pkl \
    --model intfloat/e5-base-v2 \
    --text_field query --id_field query_id \
    --prompt_name query
```

### Large documents with trimming

For very large documents, trim before tiling:

```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model BAAI/bge-base-en-v1.5 \
    --text_field text --id_field id \
    --tile \
    --max_length 512 \
    --trim_text_to 10000 \
    --sentence_aligned
```

## Performance Options

### Parallel tiling with multiple workers

Speed up text tiling with parallel processing:

```bash
# Use 4 workers for parallel tiling (recommended for large datasets)
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model BAAI/bge-base-en-v1.5 \
    --text_field text --id_field id \
    --tile --max_length 512 --stride 128 \
    --num_workers 4

# Use 8 workers for very large datasets
python scripts/compute_embeddings_from_jsonl.py \
    large_dataset.jsonl output.pkl \
    --model intfloat/e5-base-v2 \
    --text_field document.text --id_field document.id \
    --tile --max_length 512 --stride 128 \
    --sentence_aligned \
    --num_workers 8
```

**Note:** Parallel workers are most beneficial when processing many documents. For small datasets (<100 docs), sequential processing (default) is often faster due to multiprocessing overhead.

### Profiling and performance analysis

Use the `--profile` flag to get detailed timing information for each stage of the pipeline:

```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model BAAI/bge-base-en-v1.5 \
    --text_field text --id_field id \
    --tile --max_length 512 --stride 128 \
    --profile
```

This will display a detailed breakdown showing:
- Model loading time
- Data loading time
- Tiler initialization time
- Text tiling time (main bottleneck for large datasets)
- Embedding computation time
- Save results time

Example output:
```
Name                                       Time (s)       Rel %       Abs %documents/s   tiles/s   chars/s   words/s
compute_embeddings                            45.2s      100.0%      100.0%      22.1      44.2    8856.3    1681.4
  model_loading                                2.1s        4.6%        4.6%     476.2     952.4     190.5k      36.2k
  data_loading                                 1.3s        2.9%        2.9%     769.2    1538.5     307.7k      58.5k
  tiler_initialization                         0.1s        0.2%        0.2%   10000.0   20000.0       4.0M     762.0k
  text_tiling                                 38.5s       85.2%       85.2%      26.0      52.0      10.4k       1.9k
  embedding_computation                        2.9s        6.4%        6.4%     344.8     689.7     137.9k      26.2k
  save_results                                 0.3s        0.7%        0.7%    3333.3    6666.7       1.3M     254.1k
```

The profiling output shows:
- **Time (s)**: Actual time spent in each stage
- **Rel %**: Percentage relative to parent operation
- **Abs %**: Percentage of total execution time
- **documents/s**: Documents processed per second
- **tiles/s**: Tiles processed per second (same as documents if not tiling)
- **chars/s**: Characters processed per second
- **words/s**: Words processed per second
```

Use profiling to identify bottlenecks and optimize your pipeline (e.g., increase `--num_workers` if text_tiling is the bottleneck).

### Adjust batch size

```bash
# Smaller batch for GPU memory constraints
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model BAAI/bge-large-en-v1.5 \
    --text_field text --id_field id \
    --batch_size 32

# Larger batch for faster processing
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --text_field text --id_field id \
    --batch_size 256
```

### CPU processing

```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --text_field text --id_field id \
    --device cpu
```

### Process limited samples

```bash
python scripts/compute_embeddings_from_jsonl.py \
    data.jsonl output.pkl \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --text_field text --id_field id \
    --max_samples 1000
```

## Output Formats

The script supports two output formats: **pickle** (default) and **milvus**.

### Pickle Format (default)

The script saves a pickle file containing a dictionary with two keys:

```python
import pickle

with open('output.pkl', 'rb') as f:
    data = pickle.load(f)

# data = {
#     'embeddings': numpy.ndarray,  # Shape: (num_texts, embedding_dim)
#     'ids': list[str]              # List of document/tile IDs
# }

embeddings = data['embeddings']  # numpy array
ids = data['ids']                 # list of IDs

print(f"Shape: {embeddings.shape}")
print(f"First ID: {ids[0]}")
print(f"First embedding: {embeddings[0]}")
```

### Milvus Format

When using `--output_format milvus`, the script creates a Milvus database file (.db) with:
- Collection name derived from filename (e.g., `embeddings.db` → collection `embeddings`)
- Stores: document ID, full text, and embedding vector
- Supports immediate vector search and filtering

See the [Milvus Database Output](#milvus-database-output) section above for details.

### Tile ID Format

When tiling is enabled, tile IDs follow the format: `{original_id}-{start_pos}-{end_pos}`

Example:
- Original ID: `doc123`
- Tile 1: `doc123-0-512`
- Tile 2: `doc123-384-896`  (with 128 token overlap)
- Tile 3: `doc123-768-1280`

## Command-Line Arguments

### Required Arguments

- `input_file`: Path to input JSONL or JSONL.bz2 file
- `output_file`: Path to output file (.pkl for pickle, .db for Milvus)
- `--text_field`: Dot-separated path to text field
- `--id_field`: Dot-separated path to ID field
- `--model`: SentenceTransformer model name or path

### Optional Arguments

**Output options:**
- `--output_format`: Output format - pickle (default) or milvus

**Field specification:**
- `--title_field`: Dot-separated path to title field

**Model options:**
- `--device`: Device to run on (cuda/cpu/mps, default: cuda)
- `--batch_size`: Batch size for encoding (default: 128)
- `--prompt_name`: Prompt name for supported models
- `--no_normalize`: Disable embedding normalization

**Processing mode:**
- `--tile`: Enable text tiling
- `--truncate`: Enable truncation (mutually exclusive with --tile)

**Tiling options:**
- `--max_length`: Maximum length in tokens/chars (default: 512)
- `--stride`: Overlap/stride in tokens/chars (default: 128)
- `--sentence_aligned`: Align tiles on sentence boundaries
- `--count_type`: token or char (default: token)
- `--title_handling`: all/first/none (default: none)
- `--trim_text_to`: Trim text before tiling
- `--num_workers`: Number of parallel workers for tiling (default: 1)

**Other:**
- `--max_samples`: Maximum samples to process
- `--quiet`: Suppress progress bars and warnings
- `--profile`: Enable detailed profiling and timing information

## Examples

### Scientific papers (long documents)

```bash
python scripts/compute_embeddings_from_jsonl.py \
    papers.jsonl paper_embeddings.pkl \
    --model allenai/specter2 \
    --text_field paper.abstract \
    --id_field paper.id \
    --title_field paper.title \
    --tile --max_length 512 --stride 128 \
    --sentence_aligned \
    --title_handling first
```

### Web pages with nested structure

```bash
python scripts/compute_embeddings_from_jsonl.py \
    webpages.jsonl webpage_embeddings.pkl \
    --model BAAI/bge-base-en-v1.5 \
    --text_field content.body \
    --id_field metadata.url \
    --title_field content.title \
    --tile --max_length 512 \
    --title_handling all \
    --prompt_name passage
```

### Query embeddings (short texts)

```bash
python scripts/compute_embeddings_from_jsonl.py \
    queries.jsonl query_embeddings.pkl \
    --model intfloat/e5-base-v2 \
    --text_field query_text \
    --id_field query_id \
    --truncate --max_length 64 \
    --prompt_name query
```

### Building a searchable knowledge base with Milvus

```bash
# Create embeddings database with tiling for long documents
python scripts/compute_embeddings_from_jsonl.py \
    knowledge_base.jsonl kb.db \
    --model BAAI/bge-base-en-v1.5 \
    --text_field content \
    --id_field doc_id \
    --output_format milvus \
    --tile --max_length 512 --stride 128 \
    --sentence_aligned \
    --prompt_name passage

# Then use Python to search:
# from pymilvus import MilvusClient
# client = MilvusClient(uri='kb.db')
# results = client.search(
#     collection_name='kb',
#     data=[query_vector],
#     limit=10,
#     output_fields=['id', 'text']
# )
```

## Tips

1. **Memory management**: If you run out of GPU memory, reduce `--batch_size`
2. **Speed vs. quality**: Sentence-aligned tiling is slower but produces better semantic chunks
3. **Stride selection**: Common stride values are 25-50% of max_length (e.g., 128 for max_length=512)
4. **Normalization**: Keep normalization enabled for similarity search (disable with `--no_normalize` only if needed)
5. **Compressed files**: The script automatically handles `.jsonl.bz2` files
6. **Parallel tiling**: Use `--num_workers 4-8` for faster tiling on large datasets (>100 documents). Parallel processing adds overhead, so it's only beneficial for larger workloads.
7. **Profiling**: Use `--profile` to identify bottlenecks. If text_tiling takes >80% of time, increase `--num_workers`. If embedding_computation dominates, increase `--batch_size` or use a faster device/model.

## Troubleshooting

### "pyizumo not found"
```bash
pip install pyizumo
```
Or disable sentence alignment with `--tile` without `--sentence_aligned`

### CUDA out of memory
Reduce batch size:
```bash
--batch_size 32
```
Or use CPU:
```bash
--device cpu
```

### "Field not found" errors
Use `scripts/jsonl_utils.py` to preview your file structure:
```bash
python scripts/jsonl_utils.py data.jsonl --preview
```

### "pymilvus not found" (for Milvus output)
```bash
pip install pymilvus
```
Or use the default pickle format by omitting `--output_format milvus`
