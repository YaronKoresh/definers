import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Mock cupy if not available
try:
    import cupy
except ImportError:
    cupy = None

from definers import cupy_to_numpy, process_rows


class MockPreprocessor:
    def fit_transform(self, data):
        return data


class TestProcessRows(unittest.TestCase):

    def setUp(self):
        self.mock_scaler = MockPreprocessor()
        self.mock_normalizer = MockPreprocessor()
        self.mock_imputer = MockPreprocessor()

    def _run_process_rows_test(
        self, mock_scaler_cls, mock_normalizer_cls, mock_imputer_cls
    ):
        mock_scaler_cls.return_value = self.mock_scaler
        mock_normalizer_cls.return_value = self.mock_normalizer
        mock_imputer_cls.return_value = self.mock_imputer

        batch_input = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]

        result = process_rows(batch_input)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))

        self.assertEqual(mock_scaler_cls.call_count, 2)
        self.assertEqual(mock_normalizer_cls.call_count, 2)
        self.assertEqual(mock_imputer_cls.call_count, 2)

    @patch("definers.SimpleImputer")
    @patch("definers.Normalizer")
    @patch("definers.StandardScaler")
    def test_process_rows_with_cuml(
        self,
        mock_scaler_cuml,
        mock_normalizer_cuml,
        mock_imputer_cuml,
    ):
        self._run_process_rows_test(
            mock_scaler_cuml, mock_normalizer_cuml, mock_imputer_cuml
        )

    @patch("definers.SimpleImputer", create=True)
    @patch("definers.Normalizer", create=True)
    @patch("definers.StandardScaler", create=True)
    def test_process_rows_with_sklearn_fallback(
        self,
        mock_scaler_sklearn,
        mock_normalizer_sklearn,
        mock_imputer_sklearn,
    ):
        with patch.dict("sys.modules", {"cuml": None}):
            self._run_process_rows_test(
                mock_scaler_sklearn,
                mock_normalizer_sklearn,
                mock_imputer_sklearn,
            )

    def test_process_rows_empty_input(self):
        batch_input = []
        result = process_rows(batch_input)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0, 0))

    @patch("definers.SimpleImputer")
    @patch("definers.Normalizer")
    @patch("definers.StandardScaler")
    def test_single_row_input(
        self, mock_scaler_cls, mock_normalizer_cls, mock_imputer_cls
    ):
        mock_scaler_cls.return_value = self.mock_scaler
        mock_normalizer_cls.return_value = self.mock_normalizer
        mock_imputer_cls.return_value = self.mock_imputer

        batch_input = [np.array([10.0, 20.0])]
        result = process_rows(batch_input)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(mock_scaler_cls.call_count, 1)
        self.assertEqual(mock_normalizer_cls.call_count, 1)
        self.assertEqual(mock_imputer_cls.call_count, 1)


if __name__ == "__main__":
    unittest.main()
