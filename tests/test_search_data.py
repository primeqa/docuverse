import json
import os
import re
import unittest
from unittest.mock import patch, MagicMock

from docuverse.engines.search_data import DefaultProcessor

class TestDefaultProcessor(unittest.TestCase):
    
    def setUp(self):
        self.dp = DefaultProcessor()

    @patch('docuverse.engines.search_data.json.load')
    @patch('docuverse.engines.search_data.open')
    def test_read_stopwords(self, mock_open, mock_json):
        stopwords = {'en': ['stop', 'word']}
        mock_json.return_value = stopwords
        DefaultProcessor.read_stopwords()
        mock_open.assert_called_once_with('resources/stopwords.json')
        mock_json.assert_called_once_with(mock_open.return_value)
        self.assertEqual(DefaultProcessor.stopwords['en'].pattern, '\\b(stop|word)\\b')
    
    def test_cleanup(self):
        text = 'Some random text with stop word.'
        with patch('docuverse.engines.search_data.DefaultProcessor.remove_stopwords', return_value='clean text') as mock_remove:
            result = DefaultProcessor.cleanup(text, lang='en', remv_stopwords=True)
            self.assertEqual(result, 'clean text')
            mock_remove.assert_called_once_with(text, lang='en', do_replace=True)
    
    def test_remove_stopwords(self):
        text = 'Some random text with stop word.'
        stopwords = {'en': re.compile("\\b(stop|word)\\b", re.IGNORECASE)}
        DefaultProcessor.stopwords = stopwords
        clean_text = DefaultProcessor.remove_stopwords(text)
        self.assertEqual(clean_text, 'Some random text with .')
    
    def test_increment_product_counts(self):
        product_id = 'product1'
        DefaultProcessor.increment_product_counts(product_id)
        self.assertEqual(DefaultProcessor.product_counts[product_id], 1)

if __name__ == '__main__':
    unittest.main()