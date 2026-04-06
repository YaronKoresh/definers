import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from definers.application_ml import features_to_text


class TestFeaturesToText(unittest.TestCase):
    def setUp(self):
        self.texts = ["hello world", "python is fun"]
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.texts)
        self.features = self.vectorizer.transform(self.texts).toarray()[0]

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

    def test_reconstruction_orders_tokens_by_weight_then_name(self):
        weighted_features = np.array([0.4, 0.4, 0.9, 0.0], dtype=np.float32)
        mock_vectorizer = MagicMock()
        mock_vectorizer.get_feature_names_out.return_value = np.array(
            ["zulu", "alpha", "middle", "zero"]
        )

        reconstructed_text = features_to_text(
            weighted_features,
            vectorizer=mock_vectorizer,
        )

        self.assertEqual(reconstructed_text, "middle alpha zulu")

    def test_extract_text_features_keeps_provided_vectorizer_vocabulary(self):
        from definers.application_ml.inference import extract_text_features

        trained_vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
        trained_vectorizer.fit(["alpha beta", "gamma"])
        expected_vocabulary = dict(trained_vectorizer.vocabulary_)

        extracted = extract_text_features(
            "alpha unseen token", trained_vectorizer
        )

        self.assertEqual(trained_vectorizer.vocabulary_, expected_vocabulary)
        self.assertEqual(extracted.shape[0], len(expected_vocabulary))
        self.assertGreater(extracted.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
