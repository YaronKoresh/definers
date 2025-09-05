import unittest
import numpy as np
from definers import linear_regression

class TestLinearRegression(unittest.TestCase):

    def test_simple_linear_relationship(self):
        X = np.array([[1], [2], [3], [4]], dtype=np.float64)
        y = np.array([3, 5, 7, 9], dtype=np.float64) 
        weights, bias = linear_regression(X, y, learning_rate=0.01, epochs=1000)
        
        expected_weights = np.array([2.0])
        expected_bias = 1.0
        
        np.testing.assert_allclose(weights, expected_weights, rtol=1e-2)
        self.assertAlmostEqual(bias, expected_bias, delta=0.1)

    def test_multiple_features(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float64)
        y = np.array([5, 8, 11, 14], dtype=np.float64)
        weights, bias = linear_regression(X, y, learning_rate=0.01, epochs=1000)
        
        expected_weights = np.array([1.0, 2.0])
        expected_bias = 0.0

        np.testing.assert_allclose(weights, expected_weights, rtol=1e-1)
        self.assertAlmostEqual(bias, expected_bias, delta=0.1)

    def test_zero_epochs(self):
        X = np.array([[1], [2]], dtype=np.float64)
        y = np.array([1, 2], dtype=np.float64)
        weights, bias = linear_regression(X, y, epochs=0)
        
        expected_weights = np.zeros(1)
        expected_bias = 0
        
        np.testing.assert_array_equal(weights, expected_weights)
        self.assertEqual(bias, expected_bias)
        
    def test_negative_values(self):
        X = np.array([[-1], [-2], [-3]], dtype=np.float64)
        y = np.array([-1, -3, -5,], dtype=np.float64)
        weights, bias = linear_regression(X, y, learning_rate=0.01, epochs=1000)

        expected_weights = np.array([2.0])
        expected_bias = 1.0

        np.testing.assert_allclose(weights, expected_weights, rtol=1e-2)
        self.assertAlmostEqual(bias, expected_bias, delta=0.1)

    def test_data_with_noise(self):
        X = np.array([[1], [2], [3], [4]], dtype=np.float64)
        y = np.array([3.1, 4.9, 7.2, 8.8], dtype=np.float64)
        weights, bias = linear_regression(X, y, learning_rate=0.01, epochs=1000)
        
        expected_weights = np.array([1.95])
        expected_bias = 1.225
        
        np.testing.assert_allclose(weights, expected_weights, rtol=1e-1)
        self.assertAlmostEqual(bias, expected_bias, delta=0.1)

if __name__ == '__main__':
    unittest.main()
