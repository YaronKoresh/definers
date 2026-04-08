import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from definers.ml.inference import predict_linear_regression
from tests.torch_stubs import FakeModel, build_fake_torch


class TestPredictLinearRegression(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_model.pth")
        self.input_dim = 2
        self.model_state = {"input_dim": self.input_dim}

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_successful_prediction(self):
        X_new = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        fake_torch = build_fake_torch(load_return_value=self.model_state)
        with (
            patch(
                "definers.ml.inference._sanitize_prediction_path",
                side_effect=lambda x: x,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            predictions = predict_linear_regression(
                X_new,
                self.model_path,
                factory=FakeModel,
            )
        self.assertIsNotNone(predictions)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (2,))

    def test_model_file_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "non_existent.pth")
        X_new = np.array([[1.0, 2.0]], dtype=np.float32)
        fake_torch = build_fake_torch(
            load_side_effect=FileNotFoundError(non_existent_path)
        )
        with (
            patch(
                "definers.ml.inference._sanitize_prediction_path",
                side_effect=lambda x: x,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            predictions = predict_linear_regression(
                X_new,
                non_existent_path,
                factory=FakeModel,
            )
        self.assertIsNone(predictions)

    def test_input_dimension_mismatch(self):
        X_new = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        fake_torch = build_fake_torch(load_return_value=self.model_state)
        with (
            patch(
                "definers.ml.inference._sanitize_prediction_path",
                side_effect=lambda x: x,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            predictions = predict_linear_regression(
                X_new,
                self.model_path,
                factory=FakeModel,
            )
        self.assertIsNone(predictions)

    def test_empty_input(self):
        X_new = np.empty((0, self.input_dim), dtype=np.float32)
        fake_torch = build_fake_torch(load_return_value=self.model_state)
        with (
            patch(
                "definers.ml.inference._sanitize_prediction_path",
                side_effect=lambda x: x,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            predictions = predict_linear_regression(
                X_new,
                self.model_path,
                factory=FakeModel,
            )
        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.shape, (0,))

    def test_single_prediction(self):
        X_new = np.array([[5.0, 6.0]], dtype=np.float32)
        fake_torch = build_fake_torch(load_return_value=self.model_state)
        with (
            patch(
                "definers.ml.inference._sanitize_prediction_path",
                side_effect=lambda x: x,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            predictions = predict_linear_regression(
                X_new,
                self.model_path,
                factory=FakeModel,
            )
        self.assertIsNotNone(predictions)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (1,))

    def test_rejected_model_path(self):
        X_new = np.array([[5.0, 6.0]], dtype=np.float32)
        with patch(
            "definers.ml.inference._sanitize_prediction_path",
            return_value=None,
        ):
            predictions = predict_linear_regression(
                X_new,
                self.model_path,
                factory=FakeModel,
            )
        self.assertIsNone(predictions)


if __name__ == "__main__":
    unittest.main()
