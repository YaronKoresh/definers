import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from definers import split_columns

class TestSplitColumns(unittest.TestCase):

    @patch('definers.select_columns')
    @patch('definers.drop_columns')
    def test_split_columns_dataset_mode(self, mock_drop_columns, mock_select_columns):
        mock_dataset = MagicMock()
        labels = ['label1', 'label2']

        mock_drop_columns.return_value = "features_dataset"
        mock_select_columns.return_value = "labels_dataset"
        
        features, labels_data = split_columns(mock_dataset, labels, is_batch=False)
        
        mock_drop_columns.assert_called_once_with(mock_dataset, labels)
        mock_select_columns.assert_called_once_with(mock_dataset, labels)
        self.assertEqual(features, "features_dataset")
        self.assertEqual(labels_data, "labels_dataset")

    def test_split_columns_dataset_mode_no_labels(self):
        data = ('X_data', 'y_data')
        
        test_cases = [None, [], [""]]
        for labels in test_cases:
            with self.subTest(labels=labels):
                X, y = split_columns(data, labels, is_batch=False)
                
                self.assertIs(X, 'X_data')
                self.assertIs(y, 'y_data')

    def test_split_columns_batch_mode(self):
        data = {
            'feature1': [1, 2, 3],
            'label1': ['A', 'B', 'C'],
            'feature2': np.array([1.1, 2.2, 3.3]),
            'label2': ['X', 'Y', 'Z']
        }
        labels = ['label1', 'label2']
        
        X_batch, y_batch = split_columns(data, labels, is_batch=True)
        
        expected_X_batch = [
            {'feature1': 1, 'feature2': 1.1},
            {'feature1': 2, 'feature2': 2.2},
            {'feature1': 3, 'feature2': 3.3}
        ]
        expected_y_batch = [
            {'label1': 'A', 'label2': 'X'},
            {'label1': 'B', 'label2': 'Y'},
            {'label1': 'C', 'label2': 'Z'}
        ]
        
        self.assertEqual(X_batch, expected_X_batch)
        self.assertEqual(y_batch, expected_y_batch)

    def test_split_columns_batch_mode_no_labels(self):
        data = ('X_data', 'y_data')
        
        test_cases = [None, [], [""]]
        for labels in test_cases:
            with self.subTest(labels=labels):
                X, y = split_columns(data, labels, is_batch=True)
                
                self.assertIs(X, 'X_data')
                self.assertIs(y, 'y_data')

    def test_split_columns_batch_mode_empty_input(self):
        data = {}
        labels = ['label1']
        
        X_batch, y_batch = split_columns(data, labels, is_batch=True)
        
        self.assertEqual(X_batch, [])
        self.assertEqual(y_batch, [])

    def test_split_columns_batch_mode_no_list_values(self):
        data = {'feature1': 1, 'label1': 'A'}
        labels = ['label1']

        X_batch, y_batch = split_columns(data, labels, is_batch=True)

        self.assertEqual(X_batch, [])
        self.assertEqual(y_batch, [])

if __name__ == '__main__':
    unittest.main()
