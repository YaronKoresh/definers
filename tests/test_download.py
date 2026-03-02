import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.definers._web import download_and_unzip, download_file


class TestDownloadFile(unittest.TestCase):
    @patch("src.definers._web._execute_async_operation", return_value=False)
    def test_returns_none_on_failure(self, mock_exec: MagicMock) -> None:
        result = download_file("https://example.com/file.bin", "/tmp/file.bin")
        self.assertIsNone(result)
        mock_exec.assert_called_once()

    @patch("src.definers._web._execute_async_operation", return_value=True)
    def test_returns_destination_on_success(self, mock_exec: MagicMock) -> None:
        destination = "/tmp/file.bin"
        result = download_file("https://example.com/file.bin", destination)
        self.assertEqual(result, destination)


class TestDownloadAndUnzip(unittest.TestCase):
    @patch("src.definers._web._execute_async_operation", return_value=True)
    def test_returns_true_on_success(self, mock_exec: MagicMock) -> None:
        result = download_and_unzip("https://example.com/file.zip", "/tmp/out")
        self.assertTrue(result)

    @patch("src.definers._web._execute_async_operation", return_value=False)
    def test_returns_false_on_failure(self, mock_exec: MagicMock) -> None:
        result = download_and_unzip("https://example.com/file.zip", "/tmp/out")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
