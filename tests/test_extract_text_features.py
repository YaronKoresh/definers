import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from definers import extract_text_features


class TestExtractTextFeatures(unittest.TestCase):

    def test_successful_extraction_no_vectorizer(self):
        text = "this is a simple test sentence"
        features = extract_text_features(text)
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 1)
        self.assertGreater(features.shape[0], 0)
        self.assertTrue(np.all(features >= 0))

    def test_successful_extraction_with_vectorizer(self):
        training_text = ["a b c"]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(training_text)
        test_text = "a c"
        features = extract_text_features(
            test_text, vectorizer=vectorizer
        )
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 3)
        self.assertNotEqual(features[vectorizer.vocabulary_["a"]], 0)
        self.assertEqual(features[vectorizer.vocabulary_["b"]], 0)
        self.assertNotEqual(features[vectorizer.vocabulary_["c"]], 0)

    def test_empty_text(self):
        text = ""
        features = extract_text_features(text)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 0)

    def test_text_is_none(self):
        features = extract_text_features(None)
        self.assertIsNone(features)

    @patch(
        "sklearn.feature_extraction.text.TfidfVectorizer.fit_transform"
    )
    def test_internal_vectorizer_error(self, mock_fit_transform):
        mock_fit_transform.side_effect = Exception("Vectorizer Error")
        features = extract_text_features("some text")
        self.assertIsNone(features)

    def test_dtype_is_float32(self):
        text = "feature dtype test"
        features = extract_text_features(text)
        self.assertEqual(features.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
