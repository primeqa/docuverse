# JSONL Utilities

Utility module for reading JSONL files with flexible nested field extraction.

## Module: `jsonl_utils.py`

### Features

- **Read JSONL and JSONL.bz2 files** - Automatic compression detection
- **Nested field extraction** - Use dot notation to access nested fields
- **Array support** - Index arrays or use wildcards to extract all items
- **Preview functionality** - Inspect file structure before processing
- **Robust error handling** - Skip malformed lines with warnings

### Functions

#### `read_jsonl_file(file_path, field_path=None, max_samples=None, verbose=True)`

Read texts from a JSONL or JSONL.bz2 file.

**Parameters:**
- `file_path` (str): Path to JSONL or JSONL.bz2 file
- `field_path` (str, optional): Dot-separated path to text field
- `max_samples` (int, optional): Maximum number of samples to read
- `verbose` (bool): Print warnings for skipped lines

**Returns:**
- `List[str]`: List of extracted text strings

#### `get_nested_field(obj, path)`

Extract a nested field from a dictionary using dot notation with array support.

**Parameters:**
- `obj` (dict): Dictionary to extract from
- `path` (str): Dot-separated path with optional array indexing

**Returns:**
- The value at the specified path (string or list of strings)

#### `preview_jsonl_file(file_path, num_lines=5)`

Preview the structure of a JSONL file.

**Parameters:**
- `file_path` (str): Path to JSONL or JSONL.bz2 file
- `num_lines` (int): Number of lines to preview

## Field Path Syntax

### Simple Fields

```python
# Top-level field
read_jsonl_file("data.jsonl", field_path="text")

# Nested field
read_jsonl_file("data.jsonl", field_path="document.text")

# Deeply nested
read_jsonl_file("data.jsonl", field_path="metadata.source.content")
```

### Array Indexing

```python
# First item in array
read_jsonl_file("data.jsonl", field_path="documents[0].text")

# Second item in array
read_jsonl_file("data.jsonl", field_path="documents[1].content")

# Specific index
read_jsonl_file("data.jsonl", field_path="results[5].title")
```

### Array Wildcards

```python
# All items in array (using *)
read_jsonl_file("data.jsonl", field_path="documents[*].text")

# All items in array (using empty brackets)
read_jsonl_file("data.jsonl", field_path="documents[].text")

# Nested arrays with wildcard
read_jsonl_file("data.jsonl", field_path="sections[*].paragraphs[*].text")
```

### Auto-Detection

If `field_path` is not specified, the module will try common field names:
- `text`
- `content`
- `question`
- `query`
- `passage`
- `document` (and `document.text` if it's a dict)

```python
# Auto-detect text field
texts = read_jsonl_file("data.jsonl")
```

## Usage Examples

### Example 1: Simple Text Field

```python
from jsonl_utils import read_jsonl_file

# File: data.jsonl
# {"text": "Hello world"}
# {"text": "Another example"}

texts = read_jsonl_file("data.jsonl", field_path="text")
# Result: ["Hello world", "Another example"]
```

### Example 2: Nested Fields

```python
# File: documents.jsonl
# {"document": {"text": "First doc", "title": "Doc 1"}}
# {"document": {"text": "Second doc", "title": "Doc 2"}}

texts = read_jsonl_file("documents.jsonl", field_path="document.text")
# Result: ["First doc", "Second doc"]
```

### Example 3: Array Indexing

```python
# File: multi.jsonl
# {"documents": [{"text": "A"}, {"text": "B"}]}
# {"documents": [{"text": "C"}, {"text": "D"}]}

# Get first item from each line
texts = read_jsonl_file("multi.jsonl", field_path="documents[0].text")
# Result: ["A", "C"]
```

### Example 4: Array Wildcards

```python
# File: multi.jsonl
# {"documents": [{"text": "A"}, {"text": "B"}]}
# {"documents": [{"text": "C"}, {"text": "D"}]}

# Get all items from all arrays
texts = read_jsonl_file("multi.jsonl", field_path="documents[*].text")
# Result: ["A", "B", "C", "D"]
```

### Example 5: Compressed Files

```python
# Works with .bz2 compression automatically
texts = read_jsonl_file("data.jsonl.bz2", field_path="text", max_samples=1000)
```

### Example 6: Preview File Structure

```python
from jsonl_utils import preview_jsonl_file

preview_jsonl_file("data.jsonl", num_lines=3)
```

Output:
```
Previewing: data.jsonl
================================================================================

Line 1:
  Type: dict
  Keys: ['document', 'metadata']
    document: "This is the first document..."
    metadata: {dict with keys: ['source', 'date']}

Line 2:
  Type: dict
  Keys: ['document', 'metadata']
    document: "This is the second document..."
    metadata: {dict with keys: ['source', 'date']}

================================================================================
```

## Command-Line Usage

The module can also be used as a standalone script:

```bash
# Preview file structure
python scripts/jsonl_utils.py data.jsonl --preview

# Extract texts with field path
python scripts/jsonl_utils.py data.jsonl --field_path document.text

# Count texts
python scripts/jsonl_utils.py data.jsonl --count

# Limit samples
python scripts/jsonl_utils.py data.jsonl --field_path text --max_samples 100

# Array wildcards
python scripts/jsonl_utils.py data.jsonl --field_path "documents[*].text"
```

## Integration Example

This utility is used by `quantize_granite_gpu_int8.py`:

```bash
# Use JSONL file for quantization testing
python scripts/quantize_granite_gpu_int8.py \
  --input_file data.jsonl \
  --field_path document.text \
  --max_samples 500
```

## Error Handling

The module provides robust error handling:

- **Invalid JSON**: Lines with malformed JSON are skipped with a warning
- **Missing fields**: Lines without the specified field are skipped with a warning
- **Array index out of bounds**: Handled gracefully with a warning
- **Type mismatches**: Clear error messages when field types don't match expectations

Example warning output:
```
Warning: Key 'text' not found in path 'document.text' at line 42
Warning: Skipping invalid JSON at line 87: Expecting ',' delimiter
```

## Performance Considerations

- **Streaming**: Processes files line-by-line, suitable for large files
- **Memory efficient**: Only loads requested samples into memory
- **Compression**: Automatic decompression with minimal overhead
- **Early stopping**: `max_samples` stops reading early, saving time

## Related Scripts

- `quantize_granite_gpu_int8.py` - Uses this utility for loading test data
- `benchmark_embedding_timing.py` - Originally contained this functionality
