import unittest
import os
from docuverse.engines.search_data import SearchData

class TestSearchData(unittest.TestCase):
    
    def setUp(self):
        # Assuming there is a specification for input data.
        # Change this to suitable value
        self.data = {"text": "test", "id": 1}
        self.search = SearchData([self.data])

    def test_entry(self):
        entry = SearchData.Entry(self.data)
        self.assertEqual(entry.get_text(), self.data.get("text"))
        self.assertEqual(entry.get_id(), self.data.get("id"))
        
    def test_get_text(self):
        self.assertEqual(self.search.get_text(0), self.data.get("text"))

    def test_get_item(self):
        entry = self.search.__getitem__(0)
        self.assertEqual(entry.get_text(), self.data.get("text"))
        self.assertEqual(entry.get_id(), self.data.get("id"))

    def test_len(self):
        self.assertEqual(self.search.__len__(), len([self.data]))
    
    def test_get_cached_filename(self):
        # Assuming that the `tiler` object is created somewhere in the code base. Use a suitable instance.
        tiler_instance = None
        filename = self.search.get_cached_filename('test_doc',
                                                   max_doc_size = 100,
                                                   stride = 2,
                                                   tiler = tiler_instance)
        self.assertTrue(os.path.isfile(filename))

    def test_open_file(self):
        file_name = 'test.txt'
        with open(file_name, 'w') as f:
            f.write('test')
        file = SearchData._open_file(file_name)
        self.assertIsNotNone(file)

    def test_read_cache_file_if_needed(self):
        # Assuming that the cache file is created correctly. Replace it with suitable filename.
        cache_file = 'test_cache.txt'
        data = SearchData.read_cache_file_if_needed(cache_file, 'test_input.txt')
        self.assertIsNotNone(data)

    def test_write_cache_file(self):
        passages = ['test']
        cache_file = 'test_cache_out.txt'
        # Assuming the function won't return any value and will only write given data to a file.
        SearchData.write_cache_file(cache_file, passages)
        with open(cache_file, 'r') as f:
            data = f.read()
        self.assertEqual(data, passages[0])

    def test_process_text(self):
        # Provide other necessary arguments based on your dataset and the way process_text is implemented
        SearchData.process_text(tiler='test_tiler',
                         unit='characters',
                         max_doc_size=20,
                         stride=10,
                         id='0',
                         remove_url=True,
                         doc_url=None,
                         uniform_product_name=None,
                         data_type="sap",
                         title_handling="all",
                         processor='test')

    def test_remove_stopwords(self):
        text_with_stopwords = "this is a test"
        stopwords_removed = SearchData.remove_stopwords(text_with_stopwords)
        self.assertNotIn('this', stopwords_removed)
        self.assertNotIn('is', stopwords_removed)
        self.assertNotIn('a', stopwords_removed)

    def test_get_orig_docid(self):
        original_id = '123_orig'
        obtained_id = SearchData.get_orig_docid(original_id)
        self.assertEqual(original_id[:-5], obtained_id)

if __name__ == '__main__':
    unittest.main()