import unittest
from unittest.mock import MagicMock, patch

from definers import fetch_dataset


class TestFetchDataset(unittest.TestCase):
    @patch("datasets.load_dataset")
    def test_successful_load(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        dataset = fetch_dataset("some_dataset")
        self.assertEqual(dataset, mock_dataset)
        mock_load_dataset.assert_called_once_with(
            "some_dataset", split="train"
        )

    @patch("datasets.load_dataset")
    def test_successful_load_with_revision(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        dataset = fetch_dataset("some_dataset", revision="v1.0")
        self.assertEqual(dataset, mock_dataset)
        mock_load_dataset.assert_called_once_with(
            "some_dataset", revision="v1.0", split="train"
        )

    @patch("datasets.load_dataset")
    def test_file_not_found_no_fallback(self, mock_load_dataset):
        mock_load_dataset.side_effect = FileNotFoundError
        dataset = fetch_dataset("non_existent_dataset")
        self.assertIsNone(dataset)

    @patch("datasets.load_dataset")
    def test_connection_error_no_fallback(self, mock_load_dataset):
        mock_load_dataset.side_effect = ConnectionError
        dataset = fetch_dataset("flaky_connection_dataset")
        self.assertIsNone(dataset)

    @patch("datasets.load_dataset")
    def test_fallback_on_error(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_load_dataset.side_effect = [
            Exception("Initial error"),
            mock_dataset,
        ]
        dataset = fetch_dataset("some_url", url_type="json")
        self.assertEqual(dataset, mock_dataset)
        self.assertEqual(mock_load_dataset.call_count, 2)

    @patch("datasets.load_dataset")
    def test_fallback_with_revision(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_load_dataset.side_effect = [
            Exception("Initial error"),
            mock_dataset,
        ]
        dataset = fetch_dataset(
            "some_url", url_type="json", revision="v2"
        )
        self.assertEqual(dataset, mock_dataset)
        mock_load_dataset.assert_any_call(
            "json",
            data_files={"train": "some_url"},
            revision="v2",
            split="train",
        )

    @patch("datasets.load_dataset", side_effect=FileNotFoundError)
    def test_fallback_file_not_found(self, mock_load_dataset):
        dataset = fetch_dataset("some_url", url_type="csv")
        self.assertIsNone(dataset)

    @patch("datasets.load_dataset", side_effect=ConnectionError)
    def test_fallback_connection_error(self, mock_load_dataset):
        dataset = fetch_dataset("some_url", url_type="parquet")
        self.assertIsNone(dataset)


if __name__ == "__main__":
    unittest.main()
