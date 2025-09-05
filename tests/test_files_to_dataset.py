import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from definers import files_to_dataset

class TestFilesToDataset(unittest.TestCase):

    @patch('definers.load_as_numpy')
    @patch('definers.cupy_to_numpy')
    @patch('definers.get_max_shapes')
    @patch('definers.reshape_numpy')
    @patch('definers.convert_tensor_dtype')
    @patch('torch.utils.data.TensorDataset')
    def test_files_to_dataset_with_features_and_labels(
        self, mock_tensor_dataset, mock_convert_dtype, mock_reshape, 
        mock_get_max_shapes, mock_cupy_to_numpy, mock_load_as_numpy
    ):
        mock_load_as_numpy.side_effect = [np.array([1]), np.array([2])]
        mock_cupy_to_numpy.side_effect = lambda x: x
        mock_get_max_shapes.return_value = (1, 1)
        mock_reshape.side_effect = lambda x, lengths: x
        mock_convert_dtype.side_effect = lambda x: x
        mock_tensor_dataset.return_value = "TensorDataset_Object"

        features_paths = ['feature1.npy']
        labels_paths = ['label1.npy']
        
        result = files_to_dataset(features_paths, labels_paths)
        
        self.assertEqual(result, "TensorDataset_Object")
        self.assertEqual(mock_load_as_numpy.call_count, 2)
        mock_tensor_dataset.assert_called_once()

    @patch('definers.load_as_numpy')
    @patch('definers.cupy_to_numpy')
    @patch('torch.utils.data.TensorDataset')
    def test_files_to_dataset_with_features_only(
        self, mock_tensor_dataset, mock_cupy_to_numpy, mock_load_as_numpy
    ):
        mock_load_as_numpy.return_value = np.array([1, 2, 3])
        mock_cupy_to_numpy.side_effect = lambda x: x
        mock_tensor_dataset.return_value = "FeaturesOnly_TensorDataset_Object"

        features_paths = ['feature.npy']
        
        result = files_to_dataset(features_paths)
        
        self.assertEqual(result, "FeaturesOnly_TensorDataset_Object")
        mock_load_as_numpy.assert_called_once_with('feature.npy', training=True)
        mock_tensor_dataset.assert_called_once()
        self.assertEqual(len(mock_tensor_dataset.call_args[0][0]), 1)

    @patch('definers.load_as_numpy', return_value=None)
    @patch('builtins.print')
    def test_files_to_dataset_loading_feature_fails(self, mock_print, mock_load_as_numpy):
        features_paths = ['bad_feature.npy']
        labels_paths = ['label.npy']
        
        result = files_to_dataset(features_paths, labels_paths)
        
        self.assertIsNone(result)
        mock_print.assert_any_call("Error loading feature file: bad_feature.npy")

    @patch('definers.load_as_numpy')
    @patch('builtins.print')
    def test_files_to_dataset_loading_label_fails(self, mock_print, mock_load_as_numpy):
        mock_load_as_numpy.side_effect = [np.array([1]), None]
        features_paths = ['feature.npy']
        labels_paths = ['bad_label.npy']
        
        result = files_to_dataset(features_paths, labels_paths)
        
        self.assertIsNone(result)
        mock_print.assert_any_call("Error loading label file: bad_label.npy")

    @patch('definers.load_as_numpy')
    @patch('torch.utils.data.TensorDataset')
    def test_files_to_dataset_handles_list_from_loader(self, mock_tensor_dataset, mock_load_as_numpy):
        mock_load_as_numpy.return_value = [np.array([1]), np.array([2])]
        
        features_paths = ['features.list']
        
        files_to_dataset(features_paths)
        
        self.assertEqual(mock_load_as_numpy.call_count, 1)
        self.assertEqual(len(mock_tensor_dataset.call_args[0][0][0]), 2)

    @patch('builtins.print')
    def test_files_to_dataset_empty_input(self, mock_print):
        result = files_to_dataset([], [])
        self.assertIsNone(result)
        mock_print.assert_any_call("No features or labels loaded.")

    @patch('definers.load_as_numpy')
    @patch('definers.catch')
    @patch('torch.stack', side_effect=Exception("Tensor stacking failed"))
    def test_files_to_dataset_tensor_creation_fails(self, mock_stack, mock_catch, mock_load):
        mock_load.return_value = np.array([1])
        features_paths = ['feature.npy']
        
        result = files_to_dataset(features_paths)
        
        self.assertIsNone(result)
        mock_catch.assert_called()
        self.assertIn("Tensor stacking failed", str(mock_catch.call_args[0][0]))

if __name__ == '__main__':
    unittest.main()
