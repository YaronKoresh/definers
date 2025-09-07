import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from torch.utils.data import TensorDataset
from definers import files_to_dataset


class TestFilesToDataset(unittest.TestCase):
    def setUp(self):
        self.cupy_patcher = patch("definers.cupy_to_numpy", side_effect=lambda x: x)
        self.mock_cupy = self.cupy_patcher.start()
        self.addCleanup(self.cupy_patcher.stop)

    @patch("definers.load_as_numpy", return_value=np.array([[1, 2], [3, 4]]))
    def test_successful_run_with_features_and_labels(self, mock_load):
        features_paths = ["f1.npy", "f2.npy"]
        labels_paths = ["l1.npy", "l2.npy"]
        dataset = files_to_dataset(features_paths, labels_paths)
        self.assertIsInstance(dataset, TensorDataset)
        self.assertEqual(len(dataset.tensors), 2)
        self.assertEqual(len(dataset), 2)

    @patch("definers.load_as_numpy", return_value=np.array([1, 2]))
    def test_successful_run_features_only(self, mock_load):
        features_paths = ["f1.npy"]
        dataset = files_to_dataset(features_paths)
        self.assertIsInstance(dataset, TensorDataset)
        self.assertEqual(len(dataset.tensors), 1)
        self.assertEqual(len(dataset), 1)

    @patch("definers.load_as_numpy", return_value=None)
    @patch("builtins.print")
    def test_loading_feature_fails(self, mock_print, mock_load):
        features_paths = ["bad_feature.npy"]
        labels_paths = ["label.npy"]
        result = files_to_dataset(features_paths, labels_paths)
        self.assertIsNone(result)
        mock_print.assert_any_call("Error loading feature file: bad_feature.npy")

    @patch("definers.load_as_numpy")
    @patch("builtins.print")
    def test_loading_label_fails(self, mock_print, mock_load):
        mock_load.side_effect = [np.array([1]), None]
        features_paths = ["feature.npy"]
        labels_paths = ["bad_label.npy"]
        result = files_to_dataset(features_paths, labels_paths)
        self.assertIsNone(result)
        mock_print.assert_any_call("Error loading label file: bad_label.npy")

    def test_empty_input_lists(self):
        result = files_to_dataset([], [])
        self.assertIsNone(result)

    @patch("definers.load_as_numpy", return_value=[np.array([1]), np.array([2])])
    def test_load_returns_list(self, mock_load):
        features_paths = ["features.list"]
        dataset = files_to_dataset(features_paths)
        self.assertIsInstance(dataset, TensorDataset)
        self.assertEqual(len(dataset), 2)

    @patch("torch.stack", side_effect=Exception("Tensor creation failed"))
    @patch("definers.catch")
    @patch("definers.load_as_numpy", return_value=np.array([1]))
    def test_tensor_creation_fails(self, mock_load, mock_catch, mock_stack):
        features_paths = ["feature.npy"]
        result = files_to_dataset(features_paths)
        self.assertIsNone(result)
        mock_catch.assert_called()

if __name__ == "__main__":
    unittest.main()