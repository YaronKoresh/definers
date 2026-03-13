import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.definers._web import download_and_unzip, download_file


def _consume_coroutine(coro, return_value):
    try:
        asyncio.run(coro)
    except Exception:
        pass
    return return_value


class TestDownloadFile(unittest.TestCase):
    @patch("src.definers._web._execute_async_operation")
    def test_returns_none_on_failure(self, mock_exec: MagicMock) -> None:
        mock_exec.side_effect = lambda coro: _consume_coroutine(coro, False)
        result = download_file("https://example.com/file.bin", "/tmp/file.bin")
        self.assertIsNone(result)
        mock_exec.assert_called_once()

    @patch("src.definers._web._execute_async_operation")
    def test_returns_destination_on_success(self, mock_exec: MagicMock) -> None:
        destination = "/tmp/file.bin"
        mock_exec.side_effect = lambda coro: _consume_coroutine(coro, True)
        result = download_file("https://example.com/file.bin", destination)
        self.assertEqual(result, destination)
        mock_exec.assert_called_once()


class TestDownloadAndUnzip(unittest.TestCase):
    @patch("src.definers._web._execute_async_operation")
    def test_returns_true_on_success(self, mock_exec: MagicMock) -> None:
        mock_exec.side_effect = lambda coro: _consume_coroutine(coro, True)
        result = download_and_unzip("https://example.com/file.zip", "/tmp/out")
        self.assertTrue(result)

    @patch("src.definers._web._execute_async_operation")
    def test_returns_false_on_failure(self, mock_exec: MagicMock) -> None:
        mock_exec.side_effect = lambda coro: _consume_coroutine(coro, False)
        result = download_and_unzip("https://example.com/file.zip", "/tmp/out")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
