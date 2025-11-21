# Array Field Path Quick Reference

This guide shows how to use array notation in the `--field_path` argument.

## Syntax Summary

| Notation | Description | Example |
|----------|-------------|---------|
| `field[*]` | All items in array (wildcard) | `documents[*].text` |
| `field[]` | All items in array (same as `[*]`) | `documents[].text` |
| `field[0]` | First item (zero-indexed) | `documents[0].text` |
| `field[2]` | Third item | `documents[2].text` |
| `field.subfield` | Implicit wildcard if field is array | `documents.text` |

## Common Use Cases

### 1. Extract All Documents from Array

**JSON:**
```json
{"documents": [{"text": "Doc1"}, {"text": "Doc2"}, {"text": "Doc3"}]}
```

**Command:**
```bash
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --field_path "documents[*].text"
```

**Result:** Extracts 3 texts per JSON line: "Doc1", "Doc2", "Doc3"

---

### 2. Extract Only First Document

**JSON:**
```json
{"documents": [{"text": "Doc1"}, {"text": "Doc2"}, {"text": "Doc3"}]}
```

**Command:**
```bash
python benchmark_embedding_timing.py \
    --input_file data.jsonl \
    --field_path "documents[0].text"
```

**Result:** Extracts 1 text per JSON line: "Doc1"

---

### 3. Nested Arrays

**JSON:**
```json
{
  "results": [
    {"items": [{"text": "A1"}, {"text": "A2"}]},
    {"items": [{"text": "B1"}, {"text": "B2"}]}
  ]
}
```

**Extract from first result:**
```bash
--field_path "results[0].items[*].text"
```
Result: "A1", "A2"

**Extract first item from first result:**
```bash
--field_path "results[0].items[0].text"
```
Result: "A1"

---

### 4. Array of Simple Values

**JSON:**
```json
{"queries": ["What is AI?", "How does ML work?", "What is NLP?"]}
```

**Command:**
```bash
--field_path "queries[*]"
```

**Result:** Extracts all 3 queries

---

### 5. Implicit Array Wildcard

**JSON:**
```json
{"documents": [{"text": "Doc1"}, {"text": "Doc2"}]}
```

**Command (no explicit array index):**
```bash
--field_path "documents.text"
```

**Result:** Same as `documents[*].text` - extracts "Doc1" and "Doc2"

---

## Real-World Examples

### RAG System with Multiple Retrieved Documents

```json
{
  "question": "What is machine learning?",
  "retrieved_docs": [
    {"title": "ML Intro", "content": "Machine learning is..."},
    {"title": "ML Applications", "content": "ML is used in..."},
    {"title": "Deep Learning", "content": "Deep learning is a subset..."}
  ]
}
```

**Benchmark all retrieved document contents:**
```bash
python benchmark_embedding_timing.py \
    --input_file rag_data.jsonl \
    --field_path "retrieved_docs[*].content" \
    --batch_sizes 1,8,16
```

**Benchmark only the question:**
```bash
python benchmark_embedding_timing.py \
    --input_file rag_data.jsonl \
    --field_path "question" \
    --batch_sizes 1,8,16
```

**Benchmark top retrieved document only:**
```bash
python benchmark_embedding_timing.py \
    --input_file rag_data.jsonl \
    --field_path "retrieved_docs[0].content" \
    --batch_sizes 1,8,16
```

---

### Multi-Language Dataset

```json
{
  "translations": [
    {"lang": "en", "text": "Hello world"},
    {"lang": "es", "text": "Hola mundo"},
    {"lang": "fr", "text": "Bonjour le monde"}
  ]
}
```

**Benchmark all translations:**
```bash
python benchmark_embedding_timing.py \
    --input_file translations.jsonl \
    --field_path "translations[*].text"
```

---

## Error Handling

### What Happens When...

**Array index is out of bounds?**
- The line is skipped with a warning
- Other lines continue processing

**A field doesn't have the expected structure?**
- The line is skipped with a warning
- Shows available fields for debugging

**Some items in array are missing the field?**
- Those items are skipped
- Other valid items are still extracted

---

## Tips

1. **Use wildcards for comprehensive benchmarks**: `documents[*].text` ensures you test with realistic data volumes

2. **Use specific indices for controlled tests**: `documents[0].text` gives consistent single-item extraction

3. **Mix with max_samples**: Limit total texts while extracting from arrays
   ```bash
   --field_path "docs[*].text" --max_samples 100
   ```
   This reads 100 JSON lines but may extract multiple texts per line

4. **Check your data first**: Run a quick test to see what gets extracted
   ```bash
   python -c "
   import json
   from benchmark_embedding_timing import read_jsonl_file
   texts = read_jsonl_file('data.jsonl', 'documents[*].text', max_samples=2)
   print(f'Extracted {len(texts)} texts')
   print('First text:', texts[0][:100] if texts else 'None')
   "
   ```

---

## See Also

- `BENCHMARK_README.md` - Full documentation
- `test_jsonl_reading.py` - Test suite with more examples
- Run tests: `python test_jsonl_reading.py`
