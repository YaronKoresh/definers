import unittest
from unittest.mock import MagicMock
from definers import drop_columns

class TestDropColumns(unittest.TestCase):

    def setUp(self):
        self.mock_dataset = MagicMock()
        self.mock_dataset.column_names = ['col_a', 'col_b', 'col_c', 'col_d']
        self.mock_dataset.remove_columns.return_value = "dataset with columns removed"

    def test_drops_specified_columns(self):
        drop_list = ['col_b', 'col_d']
        
        result = drop_columns(self.mock_dataset, drop_list)
        
        self.mock_dataset.remove_columns.assert_called_once_with(['col_b', 'col_d'])
        self.assertEqual(result, "dataset with columns removed")

    def test_handles_empty_droplist(self):
        test_cases = [None, [], [""]]
        for drop_list in test_cases:
            with self.subTest(drop_list=drop_list):
                result = drop_columns(self.mock_dataset, drop_list)
                
                self.mock_dataset.remove_columns.assert_not_called()
                self.assertIs(result, self.mock_dataset)
                
                self.mock_dataset.reset_mock()

    def test_handles_nonexistent_columns(self):
        drop_list = ['col_a', 'col_x', 'col_c', 'col_y']
        
        result = drop_columns(self.mock_dataset, drop_list)
        
        self.mock_dataset.remove_columns.assert_called_once_with(['col_a', 'col_c'])
        self.assertEqual(result, "dataset with columns removed")
        
    def test_drops_all_columns(self):
        drop_list = ['col_a', 'col_b', 'col_c', 'col_d']
        
        result = drop_columns(self.mock_dataset, drop_list)
        
        self.mock_dataset.remove_columns.assert_called_once_with(['col_a', 'col_b', 'col_c', 'col_d'])
        self.assertEqual(result, "dataset with columns removed")

if __name__ == '__main__':
    unittest.main()
