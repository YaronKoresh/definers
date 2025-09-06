import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from definers import HybridModel, feed


class TestFeed(unittest.TestCase):

    def setUp(self):
        self.X_new_np = np.array([[1, 2], [3, 4]])
        self.y_new_np = np.array([5, 6])

    @patch("definers.log")
    @patch("definers.one_dim_numpy", side_effect=lambda x: x)
    def test_feed_unsupervised_initial(
        self, mock_one_dim_numpy, mock_log
    ):
        model = feed(None, self.X_new_np)
        self.assertIsInstance(model, HybridModel)
        self.assertTrue(hasattr(model, "X_all"))
        self.assertFalse(hasattr(model, "y_all"))
        np.testing.assert_array_equal(model.X_all, self.X_new_np)
        mock_log.assert_called_once_with(
            "Feeding epoch 1 X", self.X_new_np
        )

    @patch("definers.log")
    @patch("definers.one_dim_numpy", side_effect=lambda x: x)
    @patch(
        "definers.np.concatenate",
        side_effect=lambda arrays, axis: np.concatenate(
            arrays, axis=axis
        ),
    )
    def test_feed_unsupervised_append(
        self, mock_concatenate, mock_one_dim_numpy, mock_log
    ):
        initial_model = HybridModel()
        initial_model.X_all = np.array([[0, 0]])
        model = feed(initial_model, self.X_new_np)

        self.assertTrue(mock_concatenate.called)
        self.assertEqual(model.X_all.shape[0], 3)
        np.testing.assert_array_equal(model.X_all[-2:], self.X_new_np)

    @patch("definers.log")
    @patch("definers.one_dim_numpy", side_effect=lambda x: x)
    def test_feed_supervised_initial(
        self, mock_one_dim_numpy, mock_log
    ):
        model = feed(None, self.X_new_np, self.y_new_np)
        self.assertIsInstance(model, HybridModel)
        self.assertTrue(hasattr(model, "X_all"))
        self.assertTrue(hasattr(model, "y_all"))
        np.testing.assert_array_equal(model.X_all, self.X_new_np)
        np.testing.assert_array_equal(model.y_all, self.y_new_np)
        self.assertEqual(mock_log.call_count, 2)

    @patch("definers.log")
    @patch("definers.one_dim_numpy", side_effect=lambda x: x)
    @patch(
        "definers.np.concatenate",
        side_effect=lambda arrays, axis: np.concatenate(
            arrays, axis=axis
        ),
    )
    def test_feed_supervised_append(
        self, mock_concatenate, mock_one_dim_numpy, mock_log
    ):
        initial_model = HybridModel()
        initial_model.X_all = np.array([[0, 0]])
        initial_model.y_all = np.array([0])
        model = feed(initial_model, self.X_new_np, self.y_new_np)

        self.assertTrue(mock_concatenate.called)
        self.assertEqual(model.X_all.shape[0], 3)
        self.assertEqual(model.y_all.shape[0], 3)
        np.testing.assert_array_equal(model.X_all[-2:], self.X_new_np)
        np.testing.assert_array_equal(model.y_all[-2:], self.y_new_np)

    @patch("definers.log")
    @patch("definers.one_dim_numpy", side_effect=lambda x: x)
    def test_feed_multiple_epochs(self, mock_one_dim_numpy, mock_log):
        model = feed(None, self.X_new_np, self.y_new_np, epochs=3)
        self.assertEqual(model.X_all.shape[0], 6)
        self.assertEqual(model.y_all.shape[0], 6)
        self.assertEqual(mock_log.call_count, 6)


if __name__ == "__main__":
    unittest.main()
