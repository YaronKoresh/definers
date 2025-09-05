import unittest
from unittest.mock import patch, MagicMock
import PIL
from definers import fetch_dataset

class TestFetchDataset(unittest.TestCase):

    @patch('definers.load_dataset')
    def test_successful_load(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        src = "some/dataset"
        dataset = fetch_dataset(src)
        
        mock_load_dataset.assert_called_once_with(src, split="train")
        self.assertIs(dataset, mock_dataset)

    @patch('definers.load_dataset')
    def test_successful_load_with_revision(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        src = "some/dataset"
        revision = "v1"
        dataset = fetch_dataset(src, revision=revision)
        
        mock_load_dataset.assert_called_once_with(src, revision=revision, split="train")
        self.assertIs(dataset, mock_dataset)

    @patch('definers.logging.error')
    @patch('definers.load_dataset', side_effect=FileNotFoundError("Not found"))
    def test_file_not_found_no_fallback(self, mock_load_dataset, mock_log_error):
        src = "nonexistent/dataset"
        dataset = fetch_dataset(src)
        
        mock_load_dataset.assert_called_once_with(src, split="train")
        mock_log_error.assert_called_once_with(f"Dataset {src} not found.")
        self.assertIsNone(dataset)

    @patch('definers.logging.error')
    @patch('definers.load_dataset', side_effect=ConnectionError("Connection failed"))
    def test_connection_error_no_fallback(self, mock_load_dataset, mock_log_error):
        src = "unreachable/dataset"
        dataset = fetch_dataset(src)
        
        mock_load_dataset.assert_called_once_with(src, split="train")
        mock_log_error.assert_called_once_with(f"Connection error while loading dataset {src}.")
        self.assertIsNone(dataset)

    @patch('definers.logging.error')
    @patch('definers.load_dataset')
    def test_fallback_successful(self, mock_load_dataset, mock_log_error):
        mock_dataset = MagicMock()
        mock_load_dataset.side_effect = [FileNotFoundError("Initial fail"), mock_dataset]
        
        src = "http://example.com/data.csv"
        url_type = "csv"
        dataset = fetch_dataset(src, url_type=url_type)
        
        self.assertEqual(mock_load_dataset.call_count, 2)
        mock_load_dataset.assert_any_call(src, split="train")
        mock_load_dataset.assert_called_with(url_type, data_files={"train": src}, split="train")
        self.assertIs(dataset, mock_dataset)

    @patch('definers.logging.error')
    @patch('definers.load_dataset')
    def test_fallback_with_revision_successful(self, mock_load_dataset, mock_log_error):
        mock_dataset = MagicMock()
        mock_load_dataset.side_effect = [FileNotFoundError("Initial fail"), mock_dataset]
        
        src = "http://example.com/data.csv"
        url_type = "csv"
        revision = "main"
        dataset = fetch_dataset(src, url_type=url_type, revision=revision)
        
        self.assertEqual(mock_load_dataset.call_count, 2)
        mock_load_dataset.assert_any_call(src, revision=revision, split="train")
        mock_load_dataset.assert_called_with(url_type, data_files={"train": src}, revision=revision, split="train")
        self.assertIs(dataset, mock_dataset)

    @patch('definers.logging.error')
    @patch('definers.load_dataset', side_effect=[FileNotFoundError("Initial fail"), FileNotFoundError("Fallback fail")])
    def test_fallback_fails_file_not_found(self, mock_load_dataset, mock_log_error):
        src = "http://example.com/data.csv"
        url_type = "csv"
        dataset = fetch_dataset(src, url_type=url_type)
        
        self.assertEqual(mock_load_dataset.call_count, 2)
        mock_log_error.assert_any_call(f"Dataset {src} not found.")
        mock_log_error.assert_called_with(f"Dataset {url_type} with data_files {src} not found.")
        self.assertIsNone(dataset)

    @patch('definers.logging.error')
    @patch('definers.load_dataset', side_effect=[ConnectionError("Initial fail"), ConnectionError("Fallback fail")])
    def test_fallback_fails_connection_error(self, mock_load_dataset, mock_log_error):
        src = "http://example.com/data.csv"
        url_type = "csv"
        dataset = fetch_dataset(src, url_type=url_type)
        
        self.assertEqual(mock_load_dataset.call_count, 2)
        mock_log_error.assert_any_call(f"Connection error while loading dataset {src}.")
        mock_log_error.assert_called_with(f"Connection error while loading dataset {url_type} with data_files {src}.")
        self.assertIsNone(dataset)

if __name__ == '__main__':
    unittest.main()
