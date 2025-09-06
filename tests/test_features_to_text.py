import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from definers import features_to_text


class TestFeaturesToText(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            "this is a sample document",
            "this is another document",
        ]
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.corpus)
        self.vocabulary = self.vectorizer.get_feature_names_out()

        text_to_test = "a sample document"
        self.test_features = (
            self.vectorizer.transform([text_to_test])
            .toarray()
            .flatten()
        )

    def test_reconstruction_with_vectorizer(self):
        reconstructed_text = features_to_text(
            self.test_features, vectorizer=self.vectorizer
        )
        self.assertIsNotNone(reconstructed_text)
        self.assertIsInstance(reconstructed_text, str)
        words = reconstructed_text.split()
        self.assertIn("sample", words)
        self.assertIn("document", words)

    def test_reconstruction_with_vocabulary(self):
        reconstructed_text = features_to_text(
            self.test_features, vocabulary=self.vocabulary
        )
        self.assertIsNotNone(reconstructed_text)
        words = reconstructed_text.split()
        self.assertIn("sample", words)
        self.assertIn("document", words)

    def test_no_vectorizer_or_vocabulary_provided(self):
        result = features_to_text(self.test_features)
        self.assertIsNone(result)

    def test_empty_features(self):
        empty_features = np.zeros_like(self.test_features)
        reconstructed_text = features_to_text(
            empty_features, vectorizer=self.vectorizer
        )
        self.assertEqual(reconstructed_text, "")

    def test_none_features(self):
        result = features_to_text(None, vectorizer=self.vectorizer)
        self.assertIsNone(result)

    @patch("definers.TfidfVectorizer")
    def test_exception_handling(self, mock_vectorizer):
        mock_instance = MagicMock()
        mock_instance.get_feature_names_out.side_effect = Exception(
            "Vectorizer error"
        )
        mock_vectorizer.return_value = mock_instance

        reconstructed_text = features_to_text(
            self.test_features, vectorizer=mock_instance
        )
        self.assertIsNone(reconstructed_text)


if __name__ == "__main__":
    unittest.main()
