import unittest
from unittest.mock import MagicMock
import numpy as np
from definers import get_cluster_content

class TestGetClusterContent(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.labels_ = np.array([0, 1, 0, 2, 1, 0])
        self.mock_model.x_all = np.array([
            [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]
        ])

    def test_get_content_for_existing_cluster(self):
        cluster_index = 0
        expected_content = [
            np.array([1, 1]),
            np.array([3, 3]),
            np.array([6, 6])
        ]
        
        result = get_cluster_content(self.mock_model, cluster_index)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        for res, exp in zip(result, expected_content):
            np.testing.assert_array_equal(res, exp)

    def test_get_content_for_another_existing_cluster(self):
        cluster_index = 1
        expected_content = [
            np.array([2, 2]),
            np.array([5, 5])
        ]
        
        result = get_cluster_content(self.mock_model, cluster_index)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        for res, exp in zip(result, expected_content):
            np.testing.assert_array_equal(res, exp)

    def test_get_content_for_non_existent_cluster(self):
        cluster_index = 99
        result = get_cluster_content(self.mock_model, cluster_index)
        self.assertIsNone(result)

    def test_raises_value_error_if_no_labels_attribute(self):
        invalid_model = MagicMock()
        # Remove the 'labels_' attribute if it exists from a previous mock run
        if hasattr(invalid_model, 'labels_'):
            del invalid_model.labels_

        with self.assertRaises(ValueError) as context:
            get_cluster_content(invalid_model, 0)
        self.assertEqual(str(context.exception), "Model must be a trained KMeans model.")

    def test_empty_cluster_content(self):
        self.mock_model.labels_ = np.array([0, 0, 0])
        self.mock_model.x_all = np.array([[1], [2], [3]])
        
        result = get_cluster_content(self.mock_model, 1) # Cluster 1 is empty
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
