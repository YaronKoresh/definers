import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from definers import features_to_text


class TestFeaturesToText(unittest.TestCase):
    def setUp(self):
        self.texts = ["hello world", "python is fun"]
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.texts)
        self.features = self.vectorizer.transform(
            self.texts
        ).toarray()[0]

    def test_successful_reconstruction(self):
        reconstructed_text = features_to_text(
            self.features, vectorizer=self.vectorizer
        )
        self.assertIsNotNone(reconstructed_text)
        self.assertIn("hello", reconstructed_text)
        self.assertIn("world", reconstructed_text)

    def test_with_vocabulary(self):
        vocab = self.vectorizer.vocabulary_
        reconstructed_text = features_to_text(
            self.features, vocabulary=list(vocab.keys())
        )
        self.assertIsNotNone(reconstructed_text)
        self.assertIn("hello", reconstructed_text)
        self.assertIn("world", reconstructed_text)

    def test_missing_vectorizer_and_vocabulary(self):
        with self.assertRaises(ValueError):
            features_to_text(self.features)

    def test_exception_handling(self):
        mock_vectorizer = MagicMock()
        mock_vectorizer.get_feature_names_out.side_effect = Exception(
            "Test exception"
        )
        reconstructed_text = features_to_text(
            self.features, vectorizer=mock_vectorizer
        )
        self.assertIsNone(reconstructed_text)


if __name__ == "__main__":
    unittest.main()
