import unittest
from unittest.mock import MagicMock

from definers import is_clusters_model


class TestIsClustersModel(unittest.TestCase):

    def test_returns_true_for_model_with_cluster_centers(self):
        mock_model = MagicMock()
        mock_model.cluster_centers_ = [[1, 2], [3, 4]]
        self.assertTrue(is_clusters_model(mock_model))

    def test_returns_false_for_model_without_cluster_centers(self):
        mock_model = MagicMock()
        if hasattr(mock_model, "cluster_centers_"):
            delattr(mock_model, "cluster_centers_")
        self.assertFalse(is_clusters_model(mock_model))

    def test_returns_false_for_non_model_object_like_string(self):
        self.assertFalse(is_clusters_model("not a model"))

    def test_returns_false_for_none_input(self):
        self.assertFalse(is_clusters_model(None))

    def test_returns_false_for_object_with_different_attributes(self):
        mock_object = MagicMock()
        mock_object.some_other_attribute = "value"
        self.assertFalse(is_clusters_model(mock_object))


if __name__ == "__main__":
    unittest.main()
