import unittest
import os
import tempfile
import shutil
import numpy as np
import torch
from definers import predict_linear_regression, LinearRegressionTorch

class TestPredictLinearRegression(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_model.pth")
        self.input_dim = 2
        
        self.model = LinearRegressionTorch(self.input_dim)
        torch.save(self.model.state_dict(), self.model_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_successful_prediction(self):
        X_new = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        predictions = predict_linear_regression(X_new, self.model_path)
        
        self.assertIsNotNone(predictions)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (2,))

    def test_model_file_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "non_existent.pth")
        X_new = np.array([[1.0, 2.0]], dtype=np.float32)
        
        predictions = predict_linear_regression(X_new, non_existent_path)
        
        self.assertIsNone(predictions)

    def test_input_dimension_mismatch(self):
        X_new = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        
        predictions = predict_linear_regression(X_new, self.model_path)
        
        self.assertIsNone(predictions)

    def test_empty_input(self):
        X_new = np.empty((0, self.input_dim), dtype=np.float32)
        
        predictions = predict_linear_regression(X_new, self.model_path)

        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.shape, (0,))

    def test_single_prediction(self):
        X_new = np.array([[5.0, 6.0]], dtype=np.float32)
        
        predictions = predict_linear_regression(X_new, self.model_path)
        
        self.assertIsNotNone(predictions)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (1,))

if __name__ == '__main__':
    unittest.main()
