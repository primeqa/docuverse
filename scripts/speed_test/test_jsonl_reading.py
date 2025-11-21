#!/usr/bin/env python3
"""
Simple test script to verify JSONL reading functionality.
"""

import json
import tempfile
import os
from benchmark_embedding_timing import read_jsonl_file, get_nested_field


def test_get_nested_field():
    """Test nested field extraction."""
    print("Testing nested field extraction...")

    # Test simple field
    data = {"text": "Hello world"}
    assert get_nested_field(data, "text") == "Hello world"
    print("  ✓ Simple field works")

    # Test nested field
    data = {"document": {"text": "Nested text", "id": 123}}
    assert get_nested_field(data, "document.text") == "Nested text"
    assert get_nested_field(data, "document.id") == 123
    print("  ✓ Nested field works")

    # Test deeply nested field
    data = {"metadata": {"content": {"body": {"text": "Deep nested"}}}}
    assert get_nested_field(data, "metadata.content.body.text") == "Deep nested"
    print("  ✓ Deeply nested field works")

    # Test array with wildcard [*]
    data = {"documents": [{"text": "First"}, {"text": "Second"}, {"text": "Third"}]}
    result = get_nested_field(data, "documents[*].text")
    assert result == ["First", "Second", "Third"]
    print("  ✓ Array wildcard [*] works")

    # Test array with empty brackets []
    result = get_nested_field(data, "documents[].text")
    assert result == ["First", "Second", "Third"]
    print("  ✓ Array wildcard [] works")

    # Test array with specific index
    result = get_nested_field(data, "documents[0].text")
    assert result == "First"
    print("  ✓ Array specific index [0] works")

    result = get_nested_field(data, "documents[2].text")
    assert result == "Third"
    print("  ✓ Array specific index [2] works")

    # Test array without index (implicit wildcard)
    result = get_nested_field(data, "documents.text")
    assert result == ["First", "Second", "Third"]
    print("  ✓ Array implicit wildcard works")

    # Test nested array
    data = {
        "results": [
            {"items": [{"text": "A1"}, {"text": "A2"}]},
            {"items": [{"text": "B1"}, {"text": "B2"}]}
        ]
    }
    result = get_nested_field(data, "results[0].items[*].text")
    assert result == ["A1", "A2"]
    print("  ✓ Nested array works")

    # Test array of simple values
    data = {"tags": ["tag1", "tag2", "tag3"]}
    result = get_nested_field(data, "tags[*]")
    assert result == ["tag1", "tag2", "tag3"]
    print("  ✓ Array of simple values works")

    print("✓ All nested field tests passed!\n")


def test_read_jsonl():
    """Test JSONL reading."""
    print("Testing JSONL reading...")

    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write test data
        json.dump({"text": "First text"}, f)
        f.write('\n')
        json.dump({"text": "Second text"}, f)
        f.write('\n')
        json.dump({"document": {"text": "Nested text"}}, f)
        f.write('\n')
        json.dump({"question": "What is AI?"}, f)
        f.write('\n')
        temp_file = f.name

    try:
        # Test reading with explicit field path
        texts = read_jsonl_file(temp_file, field_path="text")
        assert len(texts) == 2
        assert texts[0] == "First text"
        assert texts[1] == "Second text"
        print("  ✓ Reading with explicit field path works")

        # Test reading with auto-detection
        # Should find: "First text", "Second text", "What is AI?" (skips nested document)
        texts = read_jsonl_file(temp_file)
        assert len(texts) == 3
        assert "First text" in texts
        assert "What is AI?" in texts
        print("  ✓ Auto-detection of common fields works")

        # Test reading with nested field path
        texts = read_jsonl_file(temp_file, field_path="document.text")
        assert len(texts) == 1
        assert texts[0] == "Nested text"
        print("  ✓ Reading with nested field path works")

        # Test max_samples
        texts = read_jsonl_file(temp_file, max_samples=2)
        assert len(texts) == 2
        print("  ✓ max_samples limit works")

    finally:
        os.unlink(temp_file)

    print("✓ All JSONL reading tests passed!\n")


def test_read_jsonl_with_arrays():
    """Test JSONL reading with array fields."""
    print("Testing JSONL reading with arrays...")

    # Create temporary JSONL file with array fields
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write test data with arrays
        json.dump({"documents": [{"text": "Doc1"}, {"text": "Doc2"}]}, f)
        f.write('\n')
        json.dump({"documents": [{"text": "Doc3"}]}, f)
        f.write('\n')
        json.dump({"documents": [{"text": "Doc4"}, {"text": "Doc5"}, {"text": "Doc6"}]}, f)
        f.write('\n')
        temp_file = f.name

    try:
        # Test reading with array wildcard
        texts = read_jsonl_file(temp_file, field_path="documents[*].text")
        assert len(texts) == 6  # 2 + 1 + 3 = 6 total texts
        assert texts[0] == "Doc1"
        assert texts[2] == "Doc3"
        assert texts[5] == "Doc6"
        print("  ✓ Reading arrays with [*] wildcard works")

        # Test reading with specific index
        texts = read_jsonl_file(temp_file, field_path="documents[0].text")
        assert len(texts) == 3  # First document from each line
        assert texts[0] == "Doc1"
        assert texts[1] == "Doc3"
        assert texts[2] == "Doc4"
        print("  ✓ Reading arrays with specific index works")

    finally:
        os.unlink(temp_file)

    print("✓ All array JSONL reading tests passed!\n")


def test_read_jsonl_bz2():
    """Test compressed JSONL reading."""
    print("Testing JSONL.bz2 reading...")

    import bz2

    # Create temporary compressed JSONL file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.jsonl.bz2', delete=False) as f:
        # Write compressed test data
        content = '\n'.join([
            json.dumps({"text": "Compressed text 1"}),
            json.dumps({"text": "Compressed text 2"}),
            json.dumps({"text": "Compressed text 3"}),
        ])
        f.write(bz2.compress(content.encode('utf-8')))
        temp_file = f.name

    try:
        # Test reading compressed file
        texts = read_jsonl_file(temp_file, field_path="text")
        assert len(texts) == 3
        assert texts[0] == "Compressed text 1"
        assert texts[2] == "Compressed text 3"
        print("  ✓ Reading compressed JSONL.bz2 works")

    finally:
        os.unlink(temp_file)

    print("✓ All JSONL.bz2 reading tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("JSONL Reading Functionality Tests")
    print("="*60 + "\n")

    test_get_nested_field()
    test_read_jsonl()
    test_read_jsonl_with_arrays()
    test_read_jsonl_bz2()

    print("="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
