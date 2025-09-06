import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from definers import create_vectorizer, vectorize


class TestVectorize(unittest.TestCase):

    def test_vectorize_basic(self):
        texts = ["hello world", "hello definers"]
        vectorizer = create_vectorizer(texts)
        vectorized_data = vectorize(vectorizer, texts)

        self.assertIsInstance(vectorized_data, np.ndarray)
        self.assertEqual(vectorized_data.shape, (2, 3))

    def test_vectorize_single_text(self):
        texts = ["this is a single sentence"]
        vectorizer = create_vectorizer(texts)
        vectorized_data = vectorize(vectorizer, texts)

        self.assertEqual(vectorized_data.shape, (1, 5))

    def test_vectorize_empty_texts_list(self):
        texts = []
        vectorizer = create_vectorizer(
            ["some content to build vocab"]
        )
        vectorized_data = vectorize(vectorizer, texts)

        self.assertEqual(vectorized_data.shape, (0, 5))

    def test_vectorize_with_none_input(self):
        self.assertIsNone(vectorize(None, ["a", "b"]))

        vectorizer = create_vectorizer(["a", "b"])
        self.assertIsNone(vectorize(vectorizer, None))

        self.assertIsNone(vectorize(None, None))


if __name__ == "__main__":
    unittest.main()
