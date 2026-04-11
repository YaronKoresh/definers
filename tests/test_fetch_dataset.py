import unittest
from unittest.mock import MagicMock, patch

from definers.data.loaders import fetch_dataset


class TestFetchDataset(unittest.TestCase):
    @patch("definers.data.loaders._load_remote_dataset")
    def test_successful_load(self, mock_load_remote_dataset):
        mock_dataset = MagicMock()
        mock_load_remote_dataset.return_value = mock_dataset
        dataset = fetch_dataset("some_dataset")
        self.assertEqual(dataset, mock_dataset)
        mock_load_remote_dataset.assert_called_once_with(
            "some_dataset", None, sample_rows=None
        )

    @patch("definers.data.loaders._load_remote_dataset")
    def test_successful_load_with_revision(self, mock_load_remote_dataset):
        mock_dataset = MagicMock()
        mock_load_remote_dataset.return_value = mock_dataset
        dataset = fetch_dataset("some_dataset", revision="v1.0")
        self.assertEqual(dataset, mock_dataset)
        mock_load_remote_dataset.assert_called_once_with(
            "some_dataset", "v1.0", sample_rows=None
        )

    @patch("definers.data.loaders._load_remote_dataset")
    def test_successful_load_with_sample_rows(self, mock_load_remote_dataset):
        mock_dataset = MagicMock()
        mock_load_remote_dataset.return_value = mock_dataset

        dataset = fetch_dataset("some_dataset", sample_rows=25)

        self.assertEqual(dataset, mock_dataset)
        mock_load_remote_dataset.assert_called_once_with(
            "some_dataset", None, sample_rows=25
        )

    @patch(
        "definers.data.loaders._load_remote_dataset",
        side_effect=FileNotFoundError,
    )
    def test_file_not_found_no_fallback(self, mock_load_remote_dataset):
        dataset = fetch_dataset("non_existent_dataset")
        self.assertIsNone(dataset)
        mock_load_remote_dataset.assert_called_once_with(
            "non_existent_dataset", None, sample_rows=None
        )

    @patch(
        "definers.data.loaders._load_remote_dataset",
        side_effect=ConnectionError,
    )
    def test_connection_error_no_fallback(self, mock_load_remote_dataset):
        dataset = fetch_dataset("flaky_connection_dataset")
        self.assertIsNone(dataset)
        mock_load_remote_dataset.assert_called_once_with(
            "flaky_connection_dataset", None, sample_rows=None
        )

    @patch("definers.data.loaders._load_remote_dataset_fallback")
    @patch("definers.data.loaders._load_remote_dataset")
    def test_fallback_on_error(
        self, mock_load_remote_dataset, mock_load_remote_dataset_fallback
    ):
        mock_dataset = MagicMock()
        mock_load_remote_dataset.side_effect = Exception("Initial error")
        mock_load_remote_dataset_fallback.return_value = mock_dataset
        dataset = fetch_dataset("some_url", url_type="json")
        self.assertEqual(dataset, mock_dataset)
        mock_load_remote_dataset.assert_called_once_with(
            "some_url", None, sample_rows=None
        )
        mock_load_remote_dataset_fallback.assert_called_once_with(
            "some_url", "json", None, sample_rows=None
        )

    @patch("definers.data.loaders._load_remote_dataset_fallback")
    @patch("definers.data.loaders._load_remote_dataset")
    def test_fallback_with_revision(
        self, mock_load_remote_dataset, mock_load_remote_dataset_fallback
    ):
        mock_dataset = MagicMock()
        mock_load_remote_dataset.side_effect = Exception("Initial error")
        mock_load_remote_dataset_fallback.return_value = mock_dataset
        dataset = fetch_dataset("some_url", url_type="json", revision="v2")
        self.assertEqual(dataset, mock_dataset)
        mock_load_remote_dataset.assert_called_once_with(
            "some_url", "v2", sample_rows=None
        )
        mock_load_remote_dataset_fallback.assert_called_once_with(
            "some_url", "json", "v2", sample_rows=None
        )

    @patch("definers.data.loaders._load_remote_dataset_fallback")
    @patch("definers.data.loaders._load_remote_dataset")
    def test_fallback_preserves_sample_rows(
        self, mock_load_remote_dataset, mock_load_remote_dataset_fallback
    ):
        mock_dataset = MagicMock()
        mock_load_remote_dataset.side_effect = Exception("Initial error")
        mock_load_remote_dataset_fallback.return_value = mock_dataset

        dataset = fetch_dataset("some_url", url_type="json", sample_rows=50)

        self.assertEqual(dataset, mock_dataset)
        mock_load_remote_dataset.assert_called_once_with(
            "some_url", None, sample_rows=50
        )
        mock_load_remote_dataset_fallback.assert_called_once_with(
            "some_url", "json", None, sample_rows=50
        )

    @patch(
        "definers.data.loaders._load_remote_dataset_fallback",
        side_effect=FileNotFoundError,
    )
    @patch(
        "definers.data.loaders._load_remote_dataset",
        side_effect=Exception("Initial error"),
    )
    def test_fallback_file_not_found(
        self, mock_load_remote_dataset, mock_load_remote_dataset_fallback
    ):
        dataset = fetch_dataset("some_url", url_type="csv")
        self.assertIsNone(dataset)
        mock_load_remote_dataset.assert_called_once_with(
            "some_url", None, sample_rows=None
        )
        mock_load_remote_dataset_fallback.assert_called_once_with(
            "some_url", "csv", None, sample_rows=None
        )

    @patch(
        "definers.data.loaders._load_remote_dataset_fallback",
        side_effect=ConnectionError,
    )
    @patch(
        "definers.data.loaders._load_remote_dataset",
        side_effect=Exception("Initial error"),
    )
    def test_fallback_connection_error(
        self, mock_load_remote_dataset, mock_load_remote_dataset_fallback
    ):
        dataset = fetch_dataset("some_url", url_type="parquet")
        self.assertIsNone(dataset)
        mock_load_remote_dataset.assert_called_once_with(
            "some_url", None, sample_rows=None
        )
        mock_load_remote_dataset_fallback.assert_called_once_with(
            "some_url", "parquet", None, sample_rows=None
        )


if __name__ == "__main__":
    unittest.main()
