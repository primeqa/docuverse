import os
import pickle
import tempfile
import unittest

import pytest

from docuverse.engines.search_data import SearchData
from docuverse.utils.text_tiler import TextTiler


class TestSearchData(unittest.TestCase):

    def setUp(self):
        # SearchData.__init__ takes filenames, not raw records, so populate
        # entries directly. Entries are stored as plain dicts; __getitem__
        # wraps them in Entry on access.
        self.data = {"text": "test", "id": 1}
        self.search = SearchData([])
        self.search.entries = [self.data]

    def test_entry(self):
        entry = SearchData.Entry(self.data)
        self.assertEqual(entry.get_text(), self.data.get("text"))
        self.assertEqual(entry.get_id(), self.data.get("id"))

    def test_get_text(self):
        self.assertEqual(self.search.get_text(0), self.data.get("text"))

    def test_get_item(self):
        entry = self.search[0]
        self.assertEqual(entry.get_text(), self.data.get("text"))
        self.assertEqual(entry.get_id(), self.data.get("id"))

    def test_len(self):
        self.assertEqual(len(self.search), 1)

    def test_get_cached_filename(self):
        # get_cached_filename composes a path string; it does not create the
        # file. A char-mode TextTiler avoids the heavy HF tokenizer load.
        tiler = TextTiler(max_doc_length=100, stride=10, tokenizer=None,
                          count_type='char', aligned_on_sentences=False)
        filename = SearchData.get_cached_filename(
            'test_doc',
            max_doc_size=100,
            stride=2,
            aligned=True,
            tiler=tiler,
        )
        self.assertIsInstance(filename, str)
        self.assertTrue(filename.endswith('.pickle.xz'))

    def test_read_cache_file_if_needed(self):
        # No cache file on disk → empty list, not None.
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = os.path.join(tmp, 'test_cache.txt')
            data = SearchData.read_cache_file_if_needed(cache_file, 'test_input.txt')
            self.assertEqual(data, [])

    def test_write_cache_file(self):
        # write_cache_file pickles the passages list to disk; verify by
        # round-tripping through pickle.load.
        with tempfile.TemporaryDirectory() as tmp:
            passages = ['test']
            cache_file = os.path.join(tmp, 'test_cache_out.txt')
            SearchData.write_cache_file(cache_file, passages)
            with open(cache_file, 'rb') as f:
                roundtrip = pickle.load(f)
            self.assertEqual(roundtrip, passages)


if __name__ == '__main__':
    unittest.main()
