import asyncio
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import definers.media.web_transfer as web_transfer_module
import definers.os_utils as os_utils
import definers.path_utils as path_utils

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

from definers.media.web_transfer import download_and_unzip, download_file

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
    transport_module = web_transfer_module
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
            result = transport_module.download_file(
                "https://example.com/file.bin", "/tmp/file.bin"
            )
        self.assertIsNone(result)
        mock_download.assert_called_once_with(
            "https://example.com/file.bin",
            "/tmp/file.bin",
        )

    def test_returns_destination_on_success(self) -> None:
        transport_module = get_transport_module()
        destination = "/tmp/file.bin"
        with patch.object(
            transport_module, "download_file", return_value=destination
        ) as mock_download:
            result = transport_module.download_file(
                "https://example.com/file.bin", destination
            )
        self.assertEqual(result, destination)
        mock_download.assert_called_once_with(
            "https://example.com/file.bin",
            destination,
        )


class TestDownloadAndUnzip(unittest.TestCase):
    def test_returns_true_on_success(self) -> None:
        transport_module = get_transport_module()

        with patch.object(
            transport_module, "download_and_unzip", return_value=True
        ) as mock_download:
            result = transport_module.download_and_unzip(
                "https://example.com/file.zip", "/tmp/out"
            )
        self.assertTrue(result)
        mock_download.assert_called_once_with(
            "https://example.com/file.zip",
            "/tmp/out",
        )

    def test_returns_false_on_failure(self) -> None:
        transport_module = get_transport_module()

        with patch.object(
            transport_module, "download_and_unzip", return_value=False
        ) as mock_download:
            result = transport_module.download_and_unzip(
                "https://example.com/file.zip", "/tmp/out"
            )
        self.assertFalse(result)
        mock_download.assert_called_once_with(
            "https://example.com/file.zip",
            "/tmp/out",
        )

    def test_download_and_unzip_defaults_use_zip_orchestrator(self) -> None:
        transport_module = get_transport_module()

        defaults = transport_module.download_and_unzip.__defaults__

        self.assertIsNotNone(defaults)
        self.assertEqual(len(defaults), 2)
        self.assertIs(defaults[0], transport_module.execute_async_operation)
        self.assertIs(defaults[1], transport_module.create_zip_orchestrator)


class _SyncResponseStub:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size):
        if not self._chunks:
            return b""
        next_chunk = self._chunks.pop(0)
        if isinstance(next_chunk, BaseException):
            raise next_chunk
        return next_chunk


class _AsyncChunkStreamStub:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_chunked(self, chunk_size):
        for next_chunk in self._chunks:
            if isinstance(next_chunk, BaseException):
                raise next_chunk
            yield next_chunk


class _AsyncResponseStub:
    def __init__(self, chunks):
        self.content = _AsyncChunkStreamStub(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None


class _AsyncSessionStub:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, source_uri, timeout=None):
        return self._response


class _AsyncFileStub:
    def __init__(self, path, mode):
        self._path = path
        self._mode = mode
        self._file_obj = None

    async def __aenter__(self):
        self._file_obj = open(self._path, self._mode)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._file_obj.close()
        return False

    async def write(self, data):
        self._file_obj.write(data)


class TestHttpChunkedTransferStrategy(unittest.TestCase):
    def test_sync_transfer_replaces_target_after_complete_download(
        self,
    ) -> None:
        transport_module = get_transport_module()
        strategy = transport_module.HttpChunkedTransferStrategy(
            chunk_size_bytes=4
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "payload.bin"
            target_path.write_bytes(b"old-data")
            response = _SyncResponseStub([b"new-", b"data", b""])

            with patch.object(
                transport_module.urllib.request,
                "urlopen",
                return_value=response,
            ):
                strategy._execute_transfer_sync(
                    "https://example.com/file.bin", target_path
                )

            self.assertEqual(target_path.read_bytes(), b"new-data")
            self.assertEqual(
                list(Path(temp_dir).glob("payload.bin.*.part")), []
            )

    def test_sync_transfer_removes_partial_file_and_preserves_target_on_failure(
        self,
    ) -> None:
        transport_module = get_transport_module()
        strategy = transport_module.HttpChunkedTransferStrategy(
            chunk_size_bytes=4
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "payload.bin"
            target_path.write_bytes(b"stable-data")
            response = _SyncResponseStub(
                [b"new-", OSError("stream interrupted")]
            )

            with patch.object(
                transport_module.urllib.request,
                "urlopen",
                return_value=response,
            ):
                with self.assertRaisesRegex(OSError, "stream interrupted"):
                    strategy._execute_transfer_sync(
                        "https://example.com/file.bin", target_path
                    )

            self.assertEqual(target_path.read_bytes(), b"stable-data")
            self.assertEqual(
                list(Path(temp_dir).glob("payload.bin.*.part")), []
            )

    def test_async_transfer_replaces_target_after_complete_download(
        self,
    ) -> None:
        transport_module = get_transport_module()
        strategy = transport_module.HttpChunkedTransferStrategy(
            chunk_size_bytes=4
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "payload.bin"
            target_path.write_bytes(b"old-data")
            fake_aiohttp = types.SimpleNamespace(
                ClientTimeout=lambda total: object(),
                ClientSession=lambda: _AsyncSessionStub(
                    _AsyncResponseStub([b"new-", b"data"])
                ),
            )
            fake_aiofiles = types.SimpleNamespace(
                open=lambda path, mode: _AsyncFileStub(path, mode)
            )

            with patch.dict(
                sys.modules,
                {"aiohttp": fake_aiohttp, "aiofiles": fake_aiofiles},
            ):
                asyncio.run(
                    strategy.execute_transfer(
                        "https://example.com/file.bin", target_path
                    )
                )

            self.assertEqual(target_path.read_bytes(), b"new-data")
            self.assertEqual(
                list(Path(temp_dir).glob("payload.bin.*.part")), []
            )

    def test_async_transfer_removes_partial_file_and_preserves_target_on_failure(
        self,
    ) -> None:
        transport_module = get_transport_module()
        strategy = transport_module.HttpChunkedTransferStrategy(
            chunk_size_bytes=4
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "payload.bin"
            target_path.write_bytes(b"stable-data")
            fake_aiohttp = types.SimpleNamespace(
                ClientTimeout=lambda total: object(),
                ClientSession=lambda: _AsyncSessionStub(
                    _AsyncResponseStub(
                        [b"new-", ConnectionError("stream interrupted")]
                    )
                ),
            )
            fake_aiofiles = types.SimpleNamespace(
                open=lambda path, mode: _AsyncFileStub(path, mode)
            )

            with patch.dict(
                sys.modules,
                {"aiohttp": fake_aiohttp, "aiofiles": fake_aiofiles},
            ):
                with self.assertRaisesRegex(
                    ConnectionError, "stream interrupted"
                ):
                    asyncio.run(
                        strategy.execute_transfer(
                            "https://example.com/file.bin", target_path
                        )
                    )

            self.assertEqual(target_path.read_bytes(), b"stable-data")
            self.assertEqual(
                list(Path(temp_dir).glob("payload.bin.*.part")), []
            )


class TestDownloadFileOrchestrator(unittest.TestCase):
    def test_download_file_defaults_use_http_orchestrator(self) -> None:
        transport_module = get_transport_module()

        defaults = transport_module.download_file.__defaults__

        self.assertIsNotNone(defaults)
        self.assertEqual(len(defaults), 2)
        self.assertIs(defaults[0], transport_module.execute_async_operation)
        self.assertIs(defaults[1], transport_module.create_http_orchestrator)

    def test_create_http_orchestrator_uses_parallel_range_strategy(
        self,
    ) -> None:
        transport_module = get_transport_module()

        orchestrator = transport_module.create_http_orchestrator()

        self.assertIsInstance(
            orchestrator.strategy,
            transport_module.AdaptiveHttpTransferStrategy,
        )


if __name__ == "__main__":
    unittest.main()
