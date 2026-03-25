import os
import shutil
import tempfile
import unittest

import numpy as np

from definers.application_ml.regression_predictor import RegressionPredictor
from definers.application_ml.text_feature_extractor import TextFeatureExtractor
from definers.application_ml.text_feature_reconstructor import (
    TextFeatureReconstructor,
)
from definers.application_ml.training import LinearRegressionTorch


class TestApplicationMlInferenceServices(unittest.TestCase):
    def test_text_feature_extractor_preserves_vectorizer_vocabulary(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        trained_vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
        trained_vectorizer.fit(["alpha beta", "gamma"])
        expected_vocabulary = dict(trained_vectorizer.vocabulary_)

        extracted = TextFeatureExtractor.extract(
            "alpha unseen token",
            trained_vectorizer,
        )

        self.assertEqual(trained_vectorizer.vocabulary_, expected_vocabulary)
        self.assertEqual(extracted.shape[0], len(expected_vocabulary))
        self.assertGreater(extracted.sum(), 0.0)

    def test_text_feature_reconstructor_orders_tokens(self):
        weighted_features = np.array([0.4, 0.4, 0.9, 0.0], dtype=np.float32)
        feature_names = np.array(["zulu", "alpha", "middle", "zero"])

        reconstructed_text = TextFeatureReconstructor.reconstruct(
            weighted_features,
            vectorizer=_StaticVectorizer(feature_names),
        )

        self.assertEqual(reconstructed_text, "middle alpha zulu")

    def test_regression_predictor_rejects_missing_model(self):
        test_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(test_dir, "missing_model.pth")
            X_new = np.array([[1.0, 2.0]], dtype=np.float32)

            predictions = RegressionPredictor.predict(
                X_new,
                model_path,
                factory=LinearRegressionTorch,
            )

            self.assertIsNone(predictions)
        finally:
            shutil.rmtree(test_dir)


class _StaticVectorizer:
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def get_feature_names_out(self):
        return self._feature_names