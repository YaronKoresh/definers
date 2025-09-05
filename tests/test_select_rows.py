import unittest
from unittest.mock import MagicMock, patch
from definers import select_rows

class TestSelectRows(unittest.TestCase):

    def setUp(self):
        self.mock_dataset = MagicMock()
        self.mock_dataset.column_names = ['col_a', 'col_b']
        self.mock_dataset.__getitem__.side_effect = self.get_item_side_effect

    def get_item_side_effect(self, key):
        if key == 'col_a':
            return [10, 20, 30, 40, 50]
        if key == 'col_b':
            return ['A', 'B', 'C', 'D', 'E']
        raise KeyError(f"Column {key} not found")

    @patch('definers.Dataset.from_dict')
    def test_selects_a_slice_of_rows(self, mock_from_dict):
        mock_from_dict.return_value = "sliced_dataset"
        start_index = 1
        end_index = 4
        
        result = select_rows(self.mock_dataset, start_index, end_index)
        
        expected_subset_data = {
            'col_a': [20, 30, 40],
            'col_b': ['B', 'C', 'D']
        }
        
        mock_from_dict.assert_called_once_with(expected_subset_data)
        self.assertEqual(result, "sliced_dataset")

    @patch('definers.Dataset.from_dict')
    def test_selects_from_start(self, mock_from_dict):
        mock_from_dict.return_value = "start_slice_dataset"
        start_index = 0
        end_index = 2

        result = select_rows(self.mock_dataset, start_index, end_index)

        expected_subset_data = {
            'col_a': [10, 20],
            'col_b': ['A', 'B']
        }

        mock_from_dict.assert_called_once_with(expected_subset_data)
        self.assertEqual(result, "start_slice_dataset")

    @patch('definers.Dataset.from_dict')
    def test_selects_until_end(self, mock_from_dict):
        mock_from_dict.return_value = "end_slice_dataset"
        start_index = 3
        end_index = 5

        result = select_rows(self.mock_dataset, start_index, end_index)

        expected_subset_data = {
            'col_a': [40, 50],
            'col_b': ['D', 'E']
        }

        mock_from_dict.assert_called_once_with(expected_subset_data)
        self.assertEqual(result, "end_slice_dataset")

    @patch('definers.Dataset.from_dict')
    def test_selects_single_row(self, mock_from_dict):
        mock_from_dict.return_value = "single_row_dataset"
        start_index = 2
        end_index = 3

        result = select_rows(self.mock_dataset, start_index, end_index)

        expected_subset_data = {
            'col_a': [30],
            'col_b': ['C']
        }

        mock_from_dict.assert_called_once_with(expected_subset_data)
        self.assertEqual(result, "single_row_dataset")
        
    @patch('definers.Dataset.from_dict')
    def test_handles_empty_slice(self, mock_from_dict):
        mock_from_dict.return_value = "empty_dataset"
        start_index = 2
        end_index = 2

        result = select_rows(self.mock_dataset, start_index, end_index)

        expected_subset_data = {
            'col_a': [],
            'col_b': []
        }

        mock_from_dict.assert_called_once_with(expected_subset_data)
        self.assertEqual(result, "empty_dataset")

if __name__ == '__main__':
    unittest.main()
