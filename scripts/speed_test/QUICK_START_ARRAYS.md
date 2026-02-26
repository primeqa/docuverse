# Quick Start: Array Field Paths

## TL;DR

Extract from arrays in JSONL files using these notations:

```bash
# All items in array
--field_path "documents[*].text"

# First item only
--field_path "documents[0].text"

# Second item only
--field_path "documents[1].text"
```

## Live Demo

Run this to see it in action:
```bash
python example_array_usage.py
```

## Common Patterns

### Pattern 1: RAG Retrieved Documents

**Data:**
```json
{"retrieved_docs": [{"text": "Doc1"}, {"text": "Doc2"}, {"text": "Doc3"}]}
```

**All documents:**
```bash
--field_path "retrieved_docs[*].text"
```

**Top document:**
```bash
--field_path "retrieved_docs[0].text"
```

### Pattern 2: Multi-turn Conversation

**Data:**
```json
{"messages": [{"text": "Hello"}, {"text": "Hi there"}, {"text": "How are you?"}]}
```

**All messages:**
```bash
--field_path "messages[*].text"
```

### Pattern 3: Translation Variants

**Data:**
```json
{"translations": [{"lang": "en", "text": "Hello"}, {"lang": "es", "text": "Hola"}]}
```

**All translations:**
```bash
--field_path "translations[*].text"
```

## Test It

```bash
# Run comprehensive tests
python test_jsonl_reading.py

# See realistic example
python example_array_usage.py
```

## Full Documentation

- `ARRAY_FIELD_PATH_GUIDE.md` - Complete guide
- `BENCHMARK_README.md` - Full benchmark docs
- `ARRAY_SUPPORT_SUMMARY.md` - Implementation details
