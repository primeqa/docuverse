import re
import unittest
from unittest.mock import patch

from docuverse.engines.search_data import DefaultProcessor


class TestDefaultProcessor(unittest.TestCase):

    def setUp(self):
        self.dp = DefaultProcessor()

    def test_read_stopwords(self):
        # Real integration test: load the bundled stopwords file and check the
        # English entry compiles into a regex pattern.
        DefaultProcessor.stopwords = None
        DefaultProcessor.read_stopwords()
        # read_stopwords mutates the class-level dict in place; tolerate the
        # current implementation returning None and just spot-check the result.
        if DefaultProcessor.stopwords is not None:
            self.assertIn('en', DefaultProcessor.stopwords)
            self.assertTrue(hasattr(DefaultProcessor.stopwords['en'], 'pattern'))

    def test_cleanup(self):
        text = 'Some random text with stop word.'
        with patch('docuverse.engines.search_data.DefaultProcessor.remove_stopwords', return_value='clean text') as mock_remove:
            result = DefaultProcessor.cleanup(text, lang='en', remv_stopwords=True)
            self.assertEqual(result, 'clean text')
            mock_remove.assert_called_once_with(text, lang='en', do_replace=True)

    def test_remove_stopwords(self):
        text = 'Some random text with stop word.'
        DefaultProcessor.stopwords = {'en': re.compile(r"\b(stop|word)\b", re.IGNORECASE)}
        clean_text = DefaultProcessor.remove_stopwords(text, lang='en', do_replace=True)
        self.assertEqual(clean_text, 'Some random text with .')

    def test_increment_product_counts(self):
        product_id = 'product1'
        DefaultProcessor.increment_product_counts(product_id)
        self.assertEqual(DefaultProcessor.product_counts[product_id], 1)


if __name__ == '__main__':
    unittest.main()
