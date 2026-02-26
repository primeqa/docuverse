#!/usr/bin/env python3
"""
Utility functions for reading JSONL files with nested field extraction.

This module provides functionality to:
- Read JSONL and JSONL.bz2 files
- Extract nested fields using dot notation
- Support array indexing and wildcards
"""

import json
import bz2
import re
from typing import List, Any, Dict, Optional


def get_nested_field(obj: Dict[str, Any], path: str) -> Any:
    """
    Extract a nested field from a dictionary using dot notation with array support.

    Args:
        obj: Dictionary to extract from
        path: Dot-separated path with optional array indexing
              Examples:
              - "document.text" - simple nested field
              - "documents[*].text" - all items in array
              - "documents[0].text" - first item in array
              - "documents[].text" - same as [*], all items

    Returns:
        The value at the specified path. If array wildcard is used, returns list of values.

    Raises:
        KeyError: If path doesn't exist in the object
        IndexError: If array index is out of bounds
    """
    # Split path into segments, handling array notation
    # e.g., "documents[*].text" -> ["documents[*]", "text"]
    segments = path.split('.')
    current = obj

    for segment in segments:
        # Check if segment contains array indexing
        array_match = re.match(r'^([^\[]+)\[([^\]]*)\]$', segment)

        if array_match:
            # Handle array indexing: field[index] or field[*] or field[]
            field_name = array_match.group(1)
            index_str = array_match.group(2)

            # Navigate to the field
            if isinstance(current, dict):
                if field_name not in current:
                    raise KeyError(f"Key '{field_name}' not found in path '{path}'")
                current = current[field_name]
            else:
                raise KeyError(f"Cannot navigate to '{field_name}' in path '{path}' - current value is not a dict")

            # Handle the array indexing
            if not isinstance(current, list):
                raise KeyError(f"Field '{field_name}' in path '{path}' is not an array (got {type(current).__name__})")

            if index_str == '*' or index_str == '':
                # Wildcard: collect all items
                # Continue processing remaining path for each item
                remaining_path = '.'.join(segments[segments.index(segment) + 1:])
                if remaining_path:
                    # Recursively get nested field from each item
                    results = []
                    for item in current:
                        try:
                            results.append(get_nested_field({'_': item}, '_.' + remaining_path))
                        except (KeyError, IndexError):
                            continue  # Skip items that don't have the field
                    return results
                else:
                    # No more path, return all items
                    return current
            else:
                # Specific index
                try:
                    index = int(index_str)
                    current = current[index]
                except ValueError:
                    raise KeyError(f"Invalid array index '{index_str}' in path '{path}'")
                except IndexError:
                    raise IndexError(f"Array index {index} out of bounds in path '{path}'")
        else:
            # Regular field access
            if isinstance(current, dict):
                if segment not in current:
                    raise KeyError(f"Key '{segment}' not found in path '{path}'")
                current = current[segment]
            elif isinstance(current, list):
                # Implicit array wildcard - apply to all items
                remaining_path = '.'.join(segments[segments.index(segment):])
                results = []
                for item in current:
                    try:
                        results.append(get_nested_field({'_': item}, '_.' + remaining_path))
                    except (KeyError, IndexError):
                        continue
                return results
            else:
                raise KeyError(f"Cannot navigate to '{segment}' in path '{path}' - current value is not a dict or list")

    return current


def read_jsonl_file(
    file_path: str,
    field_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> List[str]:
    """
    Read texts from a JSONL or JSONL.bz2 file.

    Args:
        file_path: Path to JSONL or JSONL.bz2 file
        field_path: Dot-separated path to text field (e.g., "document.text")
                   Supports array indexing:
                   - "documents[0].text" - first item
                   - "documents[*].text" or "documents[].text" - all items
                   If None, assumes each line is a string or has a "text" field
        max_samples: Maximum number of samples to read (None = read all)
        verbose: Print warnings for skipped lines

    Returns:
        List of text strings

    Examples:
        >>> # Simple text field
        >>> texts = read_jsonl_file("data.jsonl", field_path="text")

        >>> # Nested field
        >>> texts = read_jsonl_file("data.jsonl", field_path="document.content")

        >>> # Array indexing - first item
        >>> texts = read_jsonl_file("data.jsonl", field_path="documents[0].text")

        >>> # Array wildcard - all items
        >>> texts = read_jsonl_file("data.jsonl", field_path="documents[*].text")

        >>> # Compressed file with limit
        >>> texts = read_jsonl_file("data.jsonl.bz2", field_path="text", max_samples=1000)
    """
    texts = []

    # Determine if file is compressed
    is_compressed = file_path.endswith('.bz2')

    # Open file with appropriate handler
    if is_compressed:
        file_handle = bz2.open(file_path, 'rt', encoding='utf-8')
    else:
        file_handle = open(file_path, 'r', encoding='utf-8')

    try:
        for i, line in enumerate(file_handle):
            # Check max_samples limit
            if max_samples is not None and i >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Skipping invalid JSON at line {i+1}: {e}")
                continue

            # Extract text based on field_path
            try:
                if field_path:
                    result = get_nested_field(data, field_path)
                elif isinstance(data, str):
                    result = data
                elif isinstance(data, dict):
                    # Try common field names
                    if 'text' in data:
                        result = data['text']
                    elif 'content' in data:
                        result = data['content']
                    elif 'question' in data:
                        result = data['question']
                    elif 'query' in data:
                        result = data['query']
                    elif 'passage' in data:
                        result = data['passage']
                    elif 'document' in data:
                        # Try to extract text from document object
                        if isinstance(data['document'], str):
                            result = data['document']
                        elif isinstance(data['document'], dict) and 'text' in data['document']:
                            result = data['document']['text']
                        else:
                            if verbose:
                                print(f"Warning: No obvious text field in 'document' at line {i+1}")
                            continue
                    else:
                        if verbose:
                            print(f"Warning: No obvious text field found at line {i+1}. "
                                  f"Available fields: {list(data.keys())}")
                        continue
                else:
                    if verbose:
                        print(f"Warning: Unexpected data type at line {i+1}: {type(data)}")
                    continue

                # Handle result - could be a string or list (from array wildcard)
                if isinstance(result, list):
                    # Array wildcard was used - add all items
                    for item in result:
                        if isinstance(item, str):
                            texts.append(item)
                        else:
                            texts.append(str(item))
                elif isinstance(result, str):
                    texts.append(result)
                else:
                    texts.append(str(result))

            except KeyError as e:
                if verbose:
                    print(f"Warning: {e} at line {i+1}")
                continue
            except IndexError as e:
                if verbose:
                    print(f"Warning: {e} at line {i+1}")
                continue

    finally:
        file_handle.close()

    return texts


def preview_jsonl_file(file_path: str, num_lines: int = 5) -> None:
    """
    Preview the structure of a JSONL file.

    Args:
        file_path: Path to JSONL or JSONL.bz2 file
        num_lines: Number of lines to preview (default: 5)
    """
    print(f"Previewing: {file_path}")
    print("=" * 80)

    is_compressed = file_path.endswith('.bz2')

    if is_compressed:
        file_handle = bz2.open(file_path, 'rt', encoding='utf-8')
    else:
        file_handle = open(file_path, 'r', encoding='utf-8')

    try:
        for i, line in enumerate(file_handle):
            if i >= num_lines:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                print(f"\nLine {i+1}:")
                print(f"  Type: {type(data).__name__}")

                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")
                    # Show structure of first few fields
                    for key in list(data.keys())[:3]:
                        value = data[key]
                        if isinstance(value, str):
                            preview = value[:60] + "..." if len(value) > 60 else value
                            print(f"    {key}: \"{preview}\"")
                        elif isinstance(value, list):
                            print(f"    {key}: [list with {len(value)} items]")
                        elif isinstance(value, dict):
                            print(f"    {key}: {{dict with keys: {list(value.keys())}}}")
                        else:
                            print(f"    {key}: {type(value).__name__}")
                elif isinstance(data, str):
                    preview = data[:100] + "..." if len(data) > 100 else data
                    print(f"  Content: \"{preview}\"")
                else:
                    print(f"  Content: {data}")

            except json.JSONDecodeError as e:
                print(f"\nLine {i+1}: Invalid JSON - {e}")

    finally:
        file_handle.close()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Utility for working with JSONL files")
    parser.add_argument("file_path", help="Path to JSONL or JSONL.bz2 file")
    parser.add_argument("--preview", action="store_true",
                        help="Preview file structure")
    parser.add_argument("--field_path", type=str,
                        help="Field path to extract (e.g., 'document.text', 'documents[0].text')")
    parser.add_argument("--max_samples", type=int,
                        help="Maximum number of samples to read")
    parser.add_argument("--count", action="store_true",
                        help="Just count the texts")

    args = parser.parse_args()

    if args.preview:
        preview_jsonl_file(args.file_path)
    else:
        texts = read_jsonl_file(args.file_path, args.field_path, args.max_samples)

        if args.count:
            print(f"Total texts: {len(texts)}")
        else:
            print(f"Loaded {len(texts)} texts")
            if texts:
                preview = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
                print(f"First text: {preview}")
                if len(texts) > 1:
                    preview = texts[-1][:100] + "..." if len(texts[-1]) > 100 else texts[-1]
                    print(f"Last text: {preview}")
