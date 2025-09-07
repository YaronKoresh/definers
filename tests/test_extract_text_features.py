import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from definers import extract_text_features


class TestExtractTextFeatures(unittest.TestCase):

    def test_successful_extraction_no_vectorizer(self):
        text = "this is a test sentence"
        features = extract_text_features(text)
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(features.ndim, 1)

    def test_successful_extraction_with_vectorizer(self):
        training_text = ["a b c"]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        vectorizer.fit(training_text)

        text = "a test sentence"
        features = extract_text_features(text, vectorizer)
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)

    def test_text_with_special_chars(self):
        text = "special$ characters^ are here!"
        features = extract_text_features(text)
        self.assertIsNotNone(features)

    def test_empty_text(self):
        text = ""
        features = extract_text_features(text)
        self.assertIsNone(features)

    def test_none_input(self):
        features = extract_text_features(None)
        self.assertIsNone(features)


if __name__ == "__main__":
    unittest.main()
