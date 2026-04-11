import unittest
from unittest.mock import patch

import numpy as np

from definers.data.preparation import pad_sequences
from tests.torch_stubs import build_fake_torch_modules


class TestPadSequences(unittest.TestCase):
    def _torch_modules(self):
        fake_torch, fake_torch_utils, fake_torch_data = (
            build_fake_torch_modules()
        )
        modules = {
            "torch": fake_torch,
            "torch.utils": fake_torch_utils,
            "torch.utils.data": fake_torch_data,
        }
        return modules, fake_torch

    def test_pad_sequences_with_numpy_arrays(self):
        modules, fake_torch = self._torch_modules()
        X = [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])]
        with patch.dict("sys.modules", modules):
            with patch(
                "definers.data.arrays.cupy_to_numpy",
                side_effect=lambda x: x,
            ):
                padded_X = pad_sequences(X)
                self.assertIsInstance(padded_X, fake_torch.Tensor)
                self.assertEqual(padded_X.shape, (2, 2, 2))
                self.assertTrue(
                    fake_torch.equal(
                        padded_X[1, 1],
                        fake_torch.tensor([0, 0], dtype=fake_torch.float64),
                    )
                )

    def test_pad_sequences_with_single_sequence(self):
        modules, fake_torch = self._torch_modules()
        X = [np.array([[1, 2, 3]])]
        with patch.dict("sys.modules", modules):
            with patch(
                "definers.data.arrays.cupy_to_numpy",
                side_effect=lambda x: x,
            ):
                padded_X = pad_sequences(X)
                self.assertEqual(padded_X.shape, (1, 1, 3))
                self.assertTrue(
                    fake_torch.equal(
                        padded_X[0, 0],
                        fake_torch.tensor([1, 2, 3], dtype=fake_torch.float64),
                    )
                )

    def test_pad_sequences_already_padded(self):
        modules, _ = self._torch_modules()
        X = [np.array([[1, 2]]), np.array([[3, 4]])]
        with patch.dict("sys.modules", modules):
            with patch(
                "definers.data.arrays.cupy_to_numpy",
                side_effect=lambda x: x,
            ):
                padded_X = pad_sequences(X)
                self.assertEqual(padded_X.shape, (2, 1, 2))

    @patch(
        "definers.data.arrays.three_dim_numpy",
        return_value=np.array([[[1], [2]], [[3]]], dtype=object),
    )
    @patch(
        "definers.data.arrays.cupy_to_numpy",
        side_effect=lambda x: np.array([[[1.0], [2.0]], [[3.0]]], dtype=object),
    )
    def test_pad_sequences_calls_dependencies(
        self, mock_cupy_to_numpy, mock_three_dim_numpy
    ):
        modules, _ = self._torch_modules()
        X = "dummy_input"
        with patch.dict("sys.modules", modules):
            pad_sequences(X)
            mock_three_dim_numpy.assert_called_once_with(X)
            mock_cupy_to_numpy.assert_called_once()

    def test_empty_input(self):
        modules, fake_torch = self._torch_modules()
        X = []
        with patch.dict("sys.modules", modules):
            with patch(
                "definers.data.arrays.cupy_to_numpy",
                side_effect=lambda x: x,
            ):
                padded_X = pad_sequences(X)
                self.assertIsInstance(padded_X, fake_torch.Tensor)
                self.assertEqual(padded_X.shape, (0,))


if __name__ == "__main__":
    unittest.main()
