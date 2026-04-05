import unittest
from unittest.mock import patch

import numpy as np
from torch.utils.data import TensorDataset

from definers.application_data.loaders import files_to_dataset


class TestFilesToDataset(unittest.TestCase):
    def setUp(self):
        self.cupy_patcher = patch(
            "definers.application_data.arrays.cupy_to_numpy",
            side_effect=lambda x: x,
        )
        self.mock_cupy = self.cupy_patcher.start()
        self.addCleanup(self.cupy_patcher.stop)

    @patch(
        "definers.application_data.loaders.load_as_numpy",
        return_value=np.array([[1, 2], [3, 4]]),
    )
    def test_successful_run_with_features_and_labels(self, mock_load):
        features_paths = ["f1.npy", "f2.npy"]
        labels_paths = ["l1.npy", "l2.npy"]
        dataset = files_to_dataset(features_paths, labels_paths)
        self.assertIsInstance(dataset, TensorDataset)
        self.assertEqual(len(dataset.tensors), 2)
        self.assertEqual(len(dataset), 2)
        mock_load.assert_any_call("f1.npy", training=True)
        mock_load.assert_any_call("l1.npy", training=True)

    @patch(
        "definers.application_data.loaders.load_as_numpy",
        return_value=np.array([1, 2]),
    )
    def test_successful_run_features_only(self, mock_load):
        features_paths = ["f1.npy"]
        dataset = files_to_dataset(features_paths)
        self.assertIsInstance(dataset, TensorDataset)
        self.assertEqual(len(dataset.tensors), 1)
        self.assertEqual(len(dataset), 1)
        mock_load.assert_called_once_with("f1.npy", training=True)

    @patch("definers.application_data.loaders.load_as_numpy", return_value=None)
    @patch("definers.application_data.loader_runtime.logger.exception")
    def test_loading_feature_fails(self, mock_logger_exc, mock_load):
        features_paths = ["bad_feature.npy"]
        labels_paths = ["label.npy"]
        result = files_to_dataset(features_paths, labels_paths)
        self.assertIsNone(result)
        mock_logger_exc.assert_called_once()

    @patch("definers.application_data.loader_runtime.logger.warning")
    def test_empty_input_lists(self, mock_logger_warn):
        result = files_to_dataset([], [])
        self.assertIsNone(result)
        mock_logger_warn.assert_called_once_with("No valid data loaded.")

    @patch(
        "definers.application_data.loaders.load_as_numpy",
        return_value=[np.array([1]), np.array([2])],
    )
    def test_load_returns_list(self, mock_load):
        features_paths = ["features.list"]
        dataset = files_to_dataset(features_paths)
        self.assertIsInstance(dataset, TensorDataset)
        self.assertEqual(len(dataset), 2)

    @patch("torch.stack", side_effect=Exception("Tensor creation failed"))
    @patch("definers.application_data.loaders._catch")
    @patch(
        "definers.application_data.loaders.load_as_numpy",
        return_value=np.array([1]),
    )
    def test_tensor_creation_fails(self, mock_load, mock_catch, mock_stack):
        features_paths = ["feature.npy"]
        result = files_to_dataset(features_paths)
        self.assertIsNone(result)
        self.assertEqual(mock_catch.call_count, 2)


if __name__ == "__main__":
    unittest.main()
