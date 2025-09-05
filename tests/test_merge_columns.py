import unittest
from unittest.mock import patch, MagicMock
import torch
from definers import merge_columns

class TestMergeColumns(unittest.TestCase):

    @patch('definers.TensorDataset')
    def test_merge_columns_with_x_and_y(self, mock_tensor_dataset):
        mock_tensor_x = torch.randn(5, 2)
        mock_tensor_y = torch.randn(5, 1)
        mock_dataset_instance = MagicMock()
        mock_tensor_dataset.return_value = mock_dataset_instance

        result = merge_columns(mock_tensor_x, mock_tensor_y)

        mock_tensor_dataset.assert_called_once_with(mock_tensor_x, mock_tensor_y)
        self.assertEqual(result, mock_dataset_instance)

    @patch('definers.TensorDataset')
    def test_merge_columns_with_x_only(self, mock_tensor_dataset):
        mock_tensor_x = torch.randn(5, 2)

        result = merge_columns(mock_tensor_x)

        mock_tensor_dataset.assert_not_called()
        self.assertIs(result, mock_tensor_x)

    def test_merge_columns_with_none_y(self):
        mock_tensor_x = torch.randn(5, 2)
        
        result = merge_columns(mock_tensor_x, y=None)
        
        self.assertIs(result, mock_tensor_x)

if __name__ == '__main__':
    unittest.main()
