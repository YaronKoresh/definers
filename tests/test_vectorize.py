import unittest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from definers import vectorize, create_vectorizer

class TestVectorize(unittest.TestCase):

    def setUp(self):
        self.texts = ["hello world", "python is great", "hello python"]
        self.vectorizer = create_vectorizer(self.texts)

    def test_vectorize_basic(self):
        new_texts = ["hello", "great world"]
        vectorized_data = vectorize(self.vectorizer, new_texts)
        self.assertIsInstance(vectorized_data, np.ndarray)
        self.assertEqual(vectorized_data.shape, (2, 4))
        self.assertGreater(vectorized_data[0, self.vectorizer.vocabulary_['hello']], 0)
        self.assertEqual(vectorized_data[0, self.vectorizer.vocabulary_['world']], 0)

    def test_vectorize_vectorizer_is_none(self):
        result = vectorize(None, self.texts)
        self.assertIsNone(result)

    def test_vectorize_texts_is_none(self):
        result = vectorize(self.vectorizer, None)
        self.assertIsNone(result)

    def test_vectorize_empty_texts_list(self):
        vectorized_data = vectorize(self.vectorizer, [])
        self.assertIsInstance(vectorized_data, np.ndarray)
        self.assertEqual(vectorized_data.shape, (0, 4))

    def test_vectorize_unseen_words(self):
        new_texts = ["unseen word test"]
        vectorized_data = vectorize(self.vectorizer, new_texts)
        self.assertIsInstance(vectorized_data, np.ndarray)
        self.assertEqual(np.count_nonzero(vectorized_data), 0)

    def test_vectorize_single_text(self):
        new_texts = ["hello python great"]
        vectorized_data = vectorize(self.vectorizer, new_texts)
        self.assertIsInstance(vectorized_data, np.ndarray)
        self.assertEqual(vectorized_data.shape, (1, 4))
        self.assertGreater(vectorized_data[0, self.vectorizer.vocabulary_['hello']], 0)
        self.assertGreater(vectorized_data[0, self.vectorizer.vocabulary_['python']], 0)
        self.assertGreater(vectorized_data[0, self.vectorizer.vocabulary_['great']], 0)

if __name__ == '__main__':
    unittest.main()
