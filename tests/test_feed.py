import unittest
from unittest.mock import MagicMock

import numpy as np

from definers.application_ml.training import HybridModel, feed


class TestFeed(unittest.TestCase):
    def setUp(self):
        self.X_new_np = np.array([[1, 2], [3, 4]])
        self.y_new_np = np.array([1, 0])

    def test_feed_unsupervised_initial(self):
        mock_log = MagicMock()
        model = feed(
            None,
            self.X_new_np,
            logger=mock_log,
            concatenate=np.concatenate,
        )
        self.assertIsInstance(model, HybridModel)
        np.testing.assert_array_equal(model.X_all, self.X_new_np)
        self.assertFalse(hasattr(model, "y_all"))

    def test_feed_supervised_initial(self):
        mock_log = MagicMock()
        model = feed(
            None,
            self.X_new_np,
            self.y_new_np,
            logger=mock_log,
            concatenate=np.concatenate,
        )
        self.assertIsInstance(model, HybridModel)
        np.testing.assert_array_equal(model.X_all, self.X_new_np)
        np.testing.assert_array_equal(model.y_all, self.y_new_np)

    def test_feed_unsupervised_append(self):
        mock_log = MagicMock()
        mock_concatenate = MagicMock(wraps=np.concatenate)
        initial_model = HybridModel()
        initial_model.X_all = np.array([[0, 0]])
        model = feed(
            initial_model,
            self.X_new_np,
            logger=mock_log,
            concatenate=mock_concatenate,
        )
        self.assertEqual(mock_concatenate.call_count, 1)
        expected_X = np.array([[0, 0], [1, 2], [3, 4]])
        np.testing.assert_array_equal(model.X_all, expected_X)

    def test_feed_supervised_append(self):
        mock_log = MagicMock()
        mock_concatenate = MagicMock(wraps=np.concatenate)
        initial_model = HybridModel()
        initial_model.X_all = np.array([[0, 0]])
        initial_model.y_all = np.array([0])
        model = feed(
            initial_model,
            self.X_new_np,
            self.y_new_np,
            logger=mock_log,
            concatenate=mock_concatenate,
        )
        self.assertEqual(mock_concatenate.call_count, 2)
        expected_X = np.array([[0, 0], [1, 2], [3, 4]])
        expected_y = np.array([0, 1, 0])
        np.testing.assert_array_equal(model.X_all, expected_X)
        np.testing.assert_array_equal(model.y_all, expected_y)

    def test_feed_with_epochs(self):
        epochs = 3
        mock_log = MagicMock()
        mock_concatenate = MagicMock(wraps=np.concatenate)
        model = feed(
            None,
            self.X_new_np,
            epochs=epochs,
            logger=mock_log,
            concatenate=mock_concatenate,
        )
        self.assertEqual(mock_log.call_count, epochs)
        expected_X = np.concatenate([self.X_new_np] * epochs, axis=0)
        np.testing.assert_array_equal(model.X_all, expected_X)
        self.assertEqual(mock_concatenate.call_count, 0)


if __name__ == "__main__":
    unittest.main()
