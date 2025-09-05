import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
from definers import fit

class TestFit(unittest.TestCase):

    def setUp(self):
        self.mock_model_supervised = MagicMock()
        self.mock_model_supervised.X_all = np.array([[1, 2], [3, 4]])
        self.mock_model_supervised.y_all = np.array([1, 2])
        
        self.mock_model_unsupervised = MagicMock()
        self.mock_model_unsupervised.X_all = np.array([[5, 6], [7, 8]])
        del self.mock_model_unsupervised.y_all

    @patch('definers.log')
    @patch('definers.get_max_shapes', return_value=[(2, 2), (2,)])
    @patch('definers.reshape_numpy', side_effect=lambda x, lengths: x)
    @patch('definers.numpy_to_cupy', side_effect=lambda x: x)
    @patch('definers.cupy_to_numpy', side_effect=lambda x: x)
    def test_fit_supervised(self, mock_cupy_to_numpy, mock_numpy_to_cupy, mock_reshape, mock_get_max_shapes, mock_log):
        returned_model = fit(self.mock_model_supervised)

        mock_log.assert_any_call("Features", self.mock_model_supervised.X_all)
        mock_log.assert_any_call("Labels", self.mock_model_supervised.y_all)
        mock_get_max_shapes.assert_called_once_with(self.mock_model_supervised.X_all, self.mock_model_supervised.y_all)
        self.mock_model_supervised.fit.assert_called_once_with(self.mock_model_supervised.X_all, self.mock_model_supervised.y_all)
        self.assertIs(returned_model, self.mock_model_supervised)

    @patch('definers.log')
    @patch('definers.get_max_shapes', return_value=[(2, 2)])
    @patch('definers.reshape_numpy', side_effect=lambda x, lengths: x)
    @patch('definers.numpy_to_cupy', side_effect=lambda x: x)
    @patch('definers.cupy_to_numpy', side_effect=lambda x: x)
    def test_fit_unsupervised(self, mock_cupy_to_numpy, mock_numpy_to_cupy, mock_reshape, mock_get_max_shapes, mock_log):
        returned_model = fit(self.mock_model_unsupervised)

        mock_log.assert_any_call("Features", self.mock_model_unsupervised.X_all)
        mock_get_max_shapes.assert_called_once_with(self.mock_model_unsupervised.X_all)
        self.mock_model_unsupervised.fit.assert_called_once_with(self.mock_model_unsupervised.X_all)
        self.assertIs(returned_model, self.mock_model_unsupervised)

    @patch('definers.log')
    @patch('definers.get_max_shapes', return_value=[(2, 2), (2,)])
    @patch('definers.reshape_numpy', side_effect=lambda x, lengths: x)
    @patch('definers.numpy_to_cupy', side_effect=lambda x: x)
    @patch('definers.cupy_to_numpy', side_effect=lambda x: x)
    @patch('definers.catch')
    def test_fit_supervised_exception(self, mock_catch, mock_cupy_to_numpy, mock_numpy_to_cupy, mock_reshape, mock_get_max_shapes, mock_log):
        self.mock_model_supervised.fit.side_effect = Exception("Supervised fit error")
        
        fit(self.mock_model_supervised)
        
        mock_catch.assert_called_once()
        self.assertIsInstance(mock_catch.call_args[0][0], Exception)

    @patch('definers.log')
    @patch('definers.get_max_shapes', return_value=[(2, 2)])
    @patch('definers.reshape_numpy', side_effect=lambda x, lengths: x)
    @patch('definers.numpy_to_cupy', side_effect=lambda x: x)
    @patch('definers.cupy_to_numpy', side_effect=lambda x: x)
    @patch('definers.catch')
    def test_fit_unsupervised_exception(self, mock_catch, mock_cupy_to_numpy, mock_numpy_to_cupy, mock_reshape, mock_get_max_shapes, mock_log):
        self.mock_model_unsupervised.fit.side_effect = Exception("Unsupervised fit error")
        
        fit(self.mock_model_unsupervised)
        
        mock_catch.assert_called_once()
        self.assertIsInstance(mock_catch.call_args[0][0], Exception)

if __name__ == '__main__':
    unittest.main()
