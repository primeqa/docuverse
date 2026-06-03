#!/usr/bin/env python3
"""
Tests for docuverse.utils.jsonl_utils.

Run as a unittest module:
    python -m unittest tests.test_jsonl_utils

Or as a CLI to exercise read_jsonl_file / get_nested_field on a real file:
    python tests/test_jsonl_utils.py path/to/file.jsonl --field_path negatives.docs[*] --num_samples 3
"""

import argparse
import bz2
import json
import os
import sys
import tempfile
import unittest

from docuverse.utils.jsonl_utils import (
    get_nested_field,
    read_jsonl_file,
    preview_jsonl_file,
)


SAMPLE_RECORDS = [
    {
        "query_id": "929814",
        "query": "What instrument does Timo play?",
        "positives": {
            "doc_id": ["929814"],
            "docs": ["Timo Korhonen is a guitarist."],
            "titles": ["Timo Korhonen"],
        },
        "negatives": {
            "doc_id": ["1070927", "320530"],
            "docs": ["Negative doc one.", "Negative doc two."],
            "titles": ["Neg title one", "Neg title two"],
            "negative_scores": [10.15, 9.61],
        },
    },
    {
        "query_id": "11",
        "query": "Another question?",
        "positives": {
            "doc_id": ["42"],
            "docs": ["Positive answer text."],
            "titles": ["Some title"],
        },
        "negatives": {
            "doc_id": ["43", "44", "45"],
            "docs": ["Neg A", "Neg B", "Neg C"],
            "titles": ["TA", "TB", "TC"],
            "negative_scores": [5.0, 4.0, 3.0],
        },
    },
]


def _write_jsonl(path, records, compressed=False):
    opener = (lambda p: bz2.open(p, "wt", encoding="utf-8")) if compressed else (
        lambda p: open(p, "w", encoding="utf-8")
    )
    with opener(path) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestGetNestedField(unittest.TestCase):
    def setUp(self):
        self.obj = SAMPLE_RECORDS[0]

    def test_simple_field(self):
        self.assertEqual(get_nested_field(self.obj, "query_id"), "929814")

    def test_nested_field(self):
        self.assertEqual(
            get_nested_field(self.obj, "positives.docs"),
            ["Timo Korhonen is a guitarist."],
        )

    def test_array_index(self):
        self.assertEqual(
            get_nested_field(self.obj, "negatives.docs[0]"),
            "Negative doc one.",
        )

    def test_array_wildcard_star(self):
        self.assertEqual(
            get_nested_field(self.obj, "negatives.docs[*]"),
            ["Negative doc one.", "Negative doc two."],
        )

    def test_array_wildcard_empty(self):
        self.assertEqual(
            get_nested_field(self.obj, "negatives.docs[]"),
            ["Negative doc one.", "Negative doc two."],
        )

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            get_nested_field(self.obj, "does_not_exist")

    def test_index_out_of_bounds(self):
        with self.assertRaises(IndexError):
            get_nested_field(self.obj, "negatives.docs[99]")


class TestReadJsonlFile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.plain = os.path.join(self.tmpdir, "data.jsonl")
        self.compressed = os.path.join(self.tmpdir, "data.jsonl.bz2")
        _write_jsonl(self.plain, SAMPLE_RECORDS, compressed=False)
        _write_jsonl(self.compressed, SAMPLE_RECORDS, compressed=True)

    def tearDown(self):
        for p in (self.plain, self.compressed):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(self.tmpdir)

    def test_read_simple_field(self):
        texts = read_jsonl_file(self.plain, field_path="query", verbose=False)
        self.assertEqual(texts, ["What instrument does Timo play?", "Another question?"])

    def test_read_nested_wildcard(self):
        texts = read_jsonl_file(self.plain, field_path="negatives.docs[*]", verbose=False)
        self.assertEqual(
            texts,
            ["Negative doc one.", "Negative doc two.", "Neg A", "Neg B", "Neg C"],
        )

    def test_read_compressed(self):
        texts = read_jsonl_file(self.compressed, field_path="query", verbose=False)
        self.assertEqual(len(texts), 2)

    def test_max_samples(self):
        texts = read_jsonl_file(
            self.plain, field_path="query", max_samples=1, verbose=False
        )
        self.assertEqual(texts, ["What instrument does Timo play?"])

    def test_default_field_inference(self):
        path = os.path.join(self.tmpdir, "plain_text.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "hello"}) + "\n")
            f.write(json.dumps({"text": "world"}) + "\n")
        try:
            texts = read_jsonl_file(path, verbose=False)
            self.assertEqual(texts, ["hello", "world"])
        finally:
            os.remove(path)


def run_cli(file_path, field_path=None, num_samples=None, preview=False):
    if preview:
        preview_jsonl_file(file_path)
        return

    texts = read_jsonl_file(
        file_path,
        field_path=field_path,
        max_samples=num_samples,
        verbose=True,
    )
    print(f"Loaded {len(texts)} texts from {file_path}")
    for i, t in enumerate(texts):
        snippet = t[:120] + ("..." if len(t) > 120 else "")
        print(f"[{i}] {snippet}")


def main():
    parser = argparse.ArgumentParser(
        description="Test/exercise jsonl_utils on a real file"
    )
    parser.add_argument("file_path", nargs="?",
                        help="Path to JSONL or JSONL.bz2 file. If omitted, runs unittests.")
    parser.add_argument("--field_path", type=str, default=None,
                        help="Field path (e.g. 'negatives.docs[*]')")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Maximum number of records to read")
    parser.add_argument("--preview", action="store_true",
                        help="Preview file structure instead of extracting")
    args = parser.parse_args()

    if args.file_path is None:
        # Fall back to unittest discovery if no file is given
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        run_cli(
            args.file_path,
            field_path=args.field_path,
            num_samples=args.num_samples,
            preview=args.preview,
        )


if __name__ == "__main__":
    main()
