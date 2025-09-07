import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from definers import feed, HybridModel

class TestFeed(unittest.TestCase):
    def setUp(self):
        self.X_new_np = np.array([[1, 2], [3, 4]])
        self.y_new_np = np.array([1, 0])

    @patch("definers.log")
    def test_feed_unsupervised_initial(self, mock_log):
        model = feed(None, self.X_new_np)
        self.assertIsInstance(model, HybridModel)
        np.testing.assert_array_equal(model.X_all, self.X_new_np)
        self.assertFalse(hasattr(model, "y_all"))

    @patch("definers.log")
    def test_feed_supervised_initial(self, mock_log):
        model = feed(None, self.X_new_np, self.y_new_np)
        self.assertIsInstance(model, HybridModel)
        np.testing.assert_array_equal(model.X_all, self.X_new_np)
        np.testing.assert_array_equal(model.y_all, self.y_new_np)

    @patch("definers.log")
    @patch("definers.np.concatenate", wraps=np.concatenate)
    def test_feed_unsupervised_append(self, mock_concatenate, mock_log):
        initial_model = HybridModel()
        initial_model.X_all = np.array([[0, 0]])
        model = feed(initial_model, self.X_new_np)
        self.assertEqual(mock_concatenate.call_count, 1)
        expected_X = np.array([[0, 0], [1, 2], [3, 4]])
        np.testing.assert_array_equal(model.X_all, expected_X)

    @patch("definers.log")
    @patch("definers.np.concatenate", wraps=np.concatenate)
    def test_feed_supervised_append(self, mock_concatenate, mock_log):
        initial_model = HybridModel()
        initial_model.X_all = np.array([[0, 0]])
        initial_model.y_all = np.array([0])
        model = feed(initial_model, self.X_new_np, self.y_new_np)
        self.assertEqual(mock_concatenate.call_count, 2)
        expected_X = np.array([[0, 0], [1, 2], [3, 4]])
        expected_y = np.array([0, 1, 0])
        np.testing.assert_array_equal(model.X_all, expected_X)
        np.testing.assert_array_equal(model.y_all, expected_y)

    @patch("definers.log")
    @patch("definers.np.concatenate", wraps=np.concatenate)
    def test_feed_with_epochs(self, mock_concatenate, mock_log):
        epochs = 3
        model = feed(None, self.X_new_np, epochs=epochs)
        self.assertEqual(mock_log.call_count, epochs)
        expected_X = np.concatenate([self.X_new_np] * epochs, axis=0)
        np.testing.assert_array_equal(model.X_all, expected_X)
        self.assertEqual(mock_concatenate.call_count, epochs - 1)

if __name__ == "__main__":
    unittest.main()
