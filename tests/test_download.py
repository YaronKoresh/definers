import asyncio
import importlib
import unittest
from unittest.mock import MagicMock, patch

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

TRANSFER_PATCH_SYMBOLS = (
    "download_file",
    "download_and_unzip",
    "execute_async_operation",
    "create_http_orchestrator",
    "create_zip_orchestrator",
)


class TestMediaImportHealth(unittest.TestCase):
    def test_import_probe_for_definers_media(self) -> None:
        imported_media_module = importlib.import_module("definers.media")

        self.assertEqual(imported_media_module.__name__, "definers.media")
        self.assertTrue(hasattr(imported_media_module, "__path__"))

    def test_import_probe_for_definers_media_web_transfer(self) -> None:
        imported_transport_module = importlib.import_module(
            "definers.media.web_transfer"
        )

        self.assertEqual(
            imported_transport_module.__name__, "definers.media.web_transfer"
        )
        for symbol_name in TRANSFER_PATCH_SYMBOLS:
            self.assertTrue(hasattr(imported_transport_module, symbol_name))


def get_transport_module():
    transport_module = web_module.web_transfer
    imported_transport_module = importlib.import_module(
        "definers.media.web_transfer"
    )
    for symbol_name in TRANSFER_PATCH_SYMBOLS:
        assert hasattr(imported_transport_module, symbol_name)
        assert hasattr(transport_module, symbol_name)
    return transport_module


class TestDownloadFile(unittest.TestCase):
    def test_returns_none_on_failure(self) -> None:
        transport_module = get_transport_module()

        with patch.object(
            transport_module, "download_file", return_value=None
        ) as mock_download:
            result = download_file(
                "https://example.com/file.bin", "/tmp/file.bin"
            )
        self.assertIsNone(result)
        mock_download.assert_called_once_with(
            "https://example.com/file.bin",
            "/tmp/file.bin",
            executor=transport_module.execute_async_operation,
            orchestrator_factory=transport_module.create_http_orchestrator,
        )

    def test_returns_destination_on_success(self) -> None:
        transport_module = get_transport_module()
        destination = "/tmp/file.bin"
        with patch.object(
            transport_module, "download_file", return_value=destination
        ) as mock_download:
            result = download_file("https://example.com/file.bin", destination)
        self.assertEqual(result, destination)
        mock_download.assert_called_once_with(
            "https://example.com/file.bin",
            destination,
            executor=transport_module.execute_async_operation,
            orchestrator_factory=transport_module.create_http_orchestrator,
        )


class TestDownloadAndUnzip(unittest.TestCase):
    def test_returns_true_on_success(self) -> None:
        transport_module = get_transport_module()

        with patch.object(
            transport_module, "download_and_unzip", return_value=True
        ) as mock_download:
            result = download_and_unzip(
                "https://example.com/file.zip", "/tmp/out"
            )
        self.assertTrue(result)
        mock_download.assert_called_once_with(
            "https://example.com/file.zip",
            "/tmp/out",
            executor=transport_module.execute_async_operation,
            orchestrator_factory=transport_module.create_zip_orchestrator,
        )

    def test_returns_false_on_failure(self) -> None:
        transport_module = get_transport_module()

        with patch.object(
            transport_module, "download_and_unzip", return_value=False
        ) as mock_download:
            result = download_and_unzip(
                "https://example.com/file.zip", "/tmp/out"
            )
        self.assertFalse(result)
        mock_download.assert_called_once_with(
            "https://example.com/file.zip",
            "/tmp/out",
            executor=transport_module.execute_async_operation,
            orchestrator_factory=transport_module.create_zip_orchestrator,
        )

    def test_uses_zip_orchestrator_factory(self) -> None:
        transport_module = get_transport_module()

        with (
            patch.object(
                transport_module, "create_zip_orchestrator"
            ) as mock_factory,
            patch.object(
                transport_module, "download_and_unzip", return_value=True
            ) as mock_download,
        ):
            result = download_and_unzip(
                "https://example.com/file.zip", "/tmp/out"
            )

        self.assertTrue(result)
        self.assertIs(
            mock_download.call_args.kwargs["orchestrator_factory"],
            mock_factory,
        )


class TestDownloadFileOrchestrator(unittest.TestCase):
    def test_uses_http_orchestrator_factory(self) -> None:
        transport_module = get_transport_module()

        with (
            patch.object(
                transport_module, "create_http_orchestrator"
            ) as mock_factory,
            patch.object(
                transport_module, "download_file", return_value="/tmp/file.bin"
            ) as mock_download,
        ):
            result = download_file(
                "https://example.com/file.bin", "/tmp/file.bin"
            )

        self.assertEqual(result, "/tmp/file.bin")
        self.assertIs(
            mock_download.call_args.kwargs["orchestrator_factory"],
            mock_factory,
        )


if __name__ == "__main__":
    unittest.main()
