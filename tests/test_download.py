import asyncio
import unittest
from unittest.mock import MagicMock, patch

import definers.media.web_transfer as transport_module
import definers.os_utils as os_utils
import definers.path_utils as path_utils
import definers.web as web_module

if not hasattr(os_utils, "get_python_version"):
    os_utils.get_python_version = lambda: "3.10"
if not hasattr(os_utils, "get_linux_distribution"):
    os_utils.get_linux_distribution = lambda: "linux"

for _name, _value in {
    "normalize_path": lambda path: str(path),
    "full_path": lambda *parts: "/".join(
        str(part) for part in parts if str(part)
    ),
    "paths": lambda *patterns: [],
    "unique": lambda items: list(dict.fromkeys(items)),
    "cwd": lambda: ".",
    "parent_directory": lambda path: "",
    "path_end": lambda path: str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
    "path_ext": lambda path: (
        "" if "." not in str(path) else "." + str(path).rsplit(".", 1)[-1]
    ),
    "path_name": lambda path: (
        str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    ),
    "tmp": lambda *args, **kwargs: "/tmp/mock",
    "secure_path": lambda path, *args, **kwargs: path,
}.items():
    if not hasattr(path_utils, _name):
        setattr(path_utils, _name, _value)

from definers.web import download_and_unzip, download_file


class TestDownloadFile(unittest.TestCase):
    @patch("definers.media.web_transfer.download_file")
    def test_returns_none_on_failure(self, mock_download: MagicMock) -> None:
        mock_download.return_value = None
        result = download_file("https://example.com/file.bin", "/tmp/file.bin")
        self.assertIsNone(result)
        mock_download.assert_called_once_with(
            "https://example.com/file.bin",
            "/tmp/file.bin",
            executor=transport_module.execute_async_operation,
            orchestrator_factory=transport_module.create_http_orchestrator,
        )

    @patch("definers.media.web_transfer.download_file")
    def test_returns_destination_on_success(
        self, mock_download: MagicMock
    ) -> None:
        destination = "/tmp/file.bin"
        mock_download.return_value = destination
        result = download_file("https://example.com/file.bin", destination)
        self.assertEqual(result, destination)
        mock_download.assert_called_once_with(
            "https://example.com/file.bin",
            destination,
            executor=transport_module.execute_async_operation,
            orchestrator_factory=transport_module.create_http_orchestrator,
        )


class TestDownloadAndUnzip(unittest.TestCase):
    @patch("definers.media.web_transfer.download_and_unzip")
    def test_returns_true_on_success(self, mock_download: MagicMock) -> None:
        mock_download.return_value = True
        result = download_and_unzip("https://example.com/file.zip", "/tmp/out")
        self.assertTrue(result)
        mock_download.assert_called_once_with(
            "https://example.com/file.zip",
            "/tmp/out",
            executor=transport_module.execute_async_operation,
            orchestrator_factory=transport_module.create_zip_orchestrator,
        )

    @patch("definers.media.web_transfer.download_and_unzip")
    def test_returns_false_on_failure(self, mock_download: MagicMock) -> None:
        mock_download.return_value = False
        result = download_and_unzip("https://example.com/file.zip", "/tmp/out")
        self.assertFalse(result)
        mock_download.assert_called_once_with(
            "https://example.com/file.zip",
            "/tmp/out",
            executor=transport_module.execute_async_operation,
            orchestrator_factory=transport_module.create_zip_orchestrator,
        )

    @patch("definers.media.web_transfer.create_zip_orchestrator")
    @patch("definers.media.web_transfer.download_and_unzip")
    def test_uses_zip_orchestrator_factory(
        self, mock_download: MagicMock, mock_factory: MagicMock
    ) -> None:
        mock_download.return_value = True
        result = download_and_unzip("https://example.com/file.zip", "/tmp/out")

        self.assertTrue(result)
        self.assertIs(
            mock_download.call_args.kwargs["orchestrator_factory"],
            mock_factory,
        )


class TestDownloadFileOrchestrator(unittest.TestCase):
    @patch("definers.media.web_transfer.create_http_orchestrator")
    @patch("definers.media.web_transfer.download_file")
    def test_uses_http_orchestrator_factory(
        self, mock_download: MagicMock, mock_factory: MagicMock
    ) -> None:
        mock_download.return_value = "/tmp/file.bin"
        result = download_file("https://example.com/file.bin", "/tmp/file.bin")

        self.assertEqual(result, "/tmp/file.bin")
        self.assertIs(
            mock_download.call_args.kwargs["orchestrator_factory"],
            mock_factory,
        )


if __name__ == "__main__":
    unittest.main()
