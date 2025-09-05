import unittest
from unittest.mock import patch, MagicMock
from definers import select_columns

class TestSelectColumns(unittest.TestCase):

    def setUp(self):
        self.mock_dataset = MagicMock()
        self.mock_dataset.column_names = ['col_a', 'col_b', 'col_c', 'col_d']

    @patch('definers.drop_columns')
    def test_selects_subset_of_columns(self, mock_drop_columns):
        mock_drop_columns.return_value = "subset dataset"
        cols_to_select = ['col_a', 'col_d']
        
        result = select_columns(self.mock_dataset, cols_to_select)
        
        mock_drop_columns.assert_called_once_with(self.mock_dataset, ['col_b', 'col_c'])
        self.assertEqual(result, "subset dataset")

    @patch('definers.drop_columns')
    def test_selects_all_columns(self, mock_drop_columns):
        mock_drop_columns.return_value = self.mock_dataset
        cols_to_select = ['col_a', 'col_b', 'col_c', 'col_d']

        result = select_columns(self.mock_dataset, cols_to_select)

        mock_drop_columns.assert_called_once_with(self.mock_dataset, [])
        self.assertEqual(result, self.mock_dataset)

    @patch('definers.drop_columns')
    def test_handles_empty_selection(self, mock_drop_columns):
        test_cases = [None, [], [""]]
        for cols_to_select in test_cases:
            with self.subTest(cols_to_select=cols_to_select):
                result = select_columns(self.mock_dataset, cols_to_select)
                
                mock_drop_columns.assert_not_called()
                self.assertIs(result, self.mock_dataset)

                mock_drop_columns.reset_mock()

    @patch('definers.drop_columns')
    def test_handles_nonexistent_columns(self, mock_drop_columns):
        mock_drop_columns.return_value = "empty dataset"
        cols_to_select = ['col_x', 'col_y']

        result = select_columns(self.mock_dataset, cols_to_select)

        mock_drop_columns.assert_called_once_with(self.mock_dataset, ['col_a', 'col_b', 'col_c', 'col_d'])
        self.assertEqual(result, "empty dataset")

    @patch('definers.drop_columns')
    def test_selects_mix_of_existing_and_nonexistent_columns(self, mock_drop_columns):
        mock_drop_columns.return_value = "mixed selection dataset"
        cols_to_select = ['col_b', 'col_z', 'col_c']

        result = select_columns(self.mock_dataset, cols_to_select)

        mock_drop_columns.assert_called_once_with(self.mock_dataset, ['col_a', 'col_d'])
        self.assertEqual(result, "mixed selection dataset")

if __name__ == '__main__':
    unittest.main()
