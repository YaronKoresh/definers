import unittest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from definers import unvectorize, create_vectorizer, vectorize

class TestUnvectorize(unittest.TestCase):

    def setUp(self):
        self.texts = ["alpha bravo charlie", "delta echo foxtrot", "alpha foxtrot golf"]
        self.vectorizer = create_vectorizer(self.texts)
        self.vectorized_data = vectorize(self.vectorizer, self.texts)

    def test_unvectorize_basic(self):
        unvectorized_texts = unvectorize(self.vectorizer, self.vectorized_data)
        self.assertIsInstance(unvectorized_texts, list)
        self.assertEqual(len(unvectorized_texts), 3)
        self.assertIn("alpha", unvectorized_texts[0])
        self.assertIn("bravo", unvectorized_texts[0])
        self.assertIn("charlie", unvectorized_texts[0])

    def test_unvectorize_vectorizer_is_none(self):
        result = unvectorize(None, self.vectorized_data)
        self.assertIsNone(result)

    def test_unvectorize_data_is_none(self):
        result = unvectorize(self.vectorizer, None)
        self.assertIsNone(result)

    def test_unvectorize_empty_data(self):
        empty_data = np.array([])
        unvectorized_texts = unvectorize(self.vectorizer, empty_data)
        self.assertEqual(unvectorized_texts, [])

    def test_unvectorize_single_document(self):
        single_vector = vectorize(self.vectorizer, ["alpha golf"])
        unvectorized_text = unvectorize(self.vectorizer, single_vector)
        self.assertEqual(len(unvectorized_text), 1)
        self.assertTrue(all(word in unvectorized_text[0] for word in ["alpha", "golf"]))

    def test_unvectorize_zeros_vector(self):
        zeros_vector = np.zeros((1, len(self.vectorizer.vocabulary_)))
        unvectorized_text = unvectorize(self.vectorizer, zeros_vector)
        self.assertEqual(len(unvectorized_text), 1)
        self.assertEqual(unvectorized_text[0], "")

if __name__ == '__main__':
    unittest.main()
