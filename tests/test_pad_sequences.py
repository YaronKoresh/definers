import unittest
from unittest.mock import patch

import numpy as np
import torch

from definers import pad_sequences


class TestPadSequences(unittest.TestCase):

    def test_pad_sequences_with_numpy_arrays(self):
        X = [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])]

        with patch(
            "definers.cupy_to_numpy", side_effect=lambda x: x
        ) as mock_cupy_to_numpy:
            padded_X = pad_sequences(X)

            self.assertIsInstance(padded_X, torch.Tensor)
            self.assertEqual(padded_X.shape, (2, 2, 2))
            self.assertTrue(
                torch.equal(
                    padded_X[1, 1],
                    torch.tensor([0, 0], dtype=torch.float64),
                )
            )

    def test_pad_sequences_with_single_sequence(self):
        X = [np.array([[1, 2, 3]])]

        with patch("definers.cupy_to_numpy", side_effect=lambda x: x):
            padded_X = pad_sequences(X)

            self.assertEqual(padded_X.shape, (1, 1, 3))
            self.assertTrue(
                torch.equal(
                    padded_X[0, 0],
                    torch.tensor([1, 2, 3], dtype=torch.float64),
                )
            )

    def test_pad_sequences_already_padded(self):
        X = [np.array([[1, 2]]), np.array([[3, 4]])]

        with patch("definers.cupy_to_numpy", side_effect=lambda x: x):
            padded_X = pad_sequences(X)

            self.assertEqual(padded_X.shape, (2, 1, 2))

    @patch(
        "definers.three_dim_numpy",
        return_value=np.array([[[1], [2]], [[3]]], dtype=object),
    )
    @patch(
        "definers.cupy_to_numpy",
        side_effect=lambda x: np.array(
            [[[1.0], [2.0]], [[3.0]]], dtype=object
        ),
    )
    def test_pad_sequences_calls_dependencies(
        self, mock_cupy_to_numpy, mock_three_dim_numpy
    ):
        X = "dummy_input"
        pad_sequences(X)

        mock_three_dim_numpy.assert_called_once_with(X)
        mock_cupy_to_numpy.assert_called_once()

    def test_empty_input(self):
        X = []
        with patch("definers.cupy_to_numpy", side_effect=lambda x: x):
            padded_X = pad_sequences(X)
            self.assertIsInstance(padded_X, torch.Tensor)
            self.assertEqual(padded_X.shape, (0,))


if __name__ == "__main__":
    unittest.main()
