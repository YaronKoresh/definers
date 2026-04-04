import asyncio
import io
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from definers.media.web_transfer import ZipExtractTransferStrategy


class FakeZipNetworkResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.raise_for_status_called = False

    async def __aenter__(self) -> "FakeZipNetworkResponse":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True

    async def read(self) -> bytes:
        return self.payload


class FakeZipClientSession:
    def __init__(self, payload: bytes):
        self.payload = payload

    async def __aenter__(self) -> "FakeZipClientSession":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def get(self, source_uri: str, timeout: object) -> FakeZipNetworkResponse:
        assert source_uri == "https://example.com/archive.zip"
        assert getattr(timeout, "total") == 20
        return FakeZipNetworkResponse(self.payload)


class FakeZipUrlResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self) -> "FakeZipUrlResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def read(self, _chunk_size_bytes: int = -1) -> bytes:
        return self.payload


def build_zip_payload(file_name: str, file_content: str) -> bytes:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w") as archive_context:
        archive_context.writestr(file_name, file_content)
    return archive_buffer.getvalue()


def test_zip_extract_transfer_strategy_extracts_archive_async(
    monkeypatch, tmp_path: Path
) -> None:
    payload = build_zip_payload("nested/payload.txt", "hello zip")
    fake_aiohttp = ModuleType("aiohttp")
    fake_aiohttp.ClientTimeout = lambda total: SimpleNamespace(total=total)
    fake_aiohttp.ClientSession = lambda: FakeZipClientSession(payload)
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    strategy = ZipExtractTransferStrategy(request_timeout_seconds=20)
    target_node = tmp_path / "extract"

    result = asyncio.run(
        strategy.execute_transfer(
            "https://example.com/archive.zip", target_node
        )
    )

    assert result is True
    assert (target_node / "nested" / "payload.txt").read_text(
        encoding="utf-8"
    ) == "hello zip"


def test_zip_extract_transfer_strategy_raises_on_invalid_archive_sync(
    monkeypatch, tmp_path: Path
) -> None:
    strategy = ZipExtractTransferStrategy()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda source_uri, timeout: FakeZipUrlResponse(b"not-a-zip"),
    )

    with pytest.raises(zipfile.BadZipFile):
        strategy._execute_transfer_sync(
            "https://example.com/archive.zip", tmp_path / "invalid"
        )


def test_zip_extract_transfer_strategy_rejects_path_traversal_sync(
    monkeypatch, tmp_path: Path
) -> None:
    payload = build_zip_payload("../outside.txt", "blocked")
    strategy = ZipExtractTransferStrategy()
    target_node = tmp_path / "extract"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda source_uri, timeout: FakeZipUrlResponse(payload),
    )

    with pytest.raises(
        ValueError, match="Archive member escapes target directory"
    ):
        strategy._execute_transfer_sync(
            "https://example.com/archive.zip", target_node
        )

    assert not (tmp_path / "outside.txt").exists()
    assert list(target_node.rglob("*")) == []


def test_zip_extract_transfer_strategy_rejects_path_traversal_async(
    monkeypatch, tmp_path: Path
) -> None:
    payload = build_zip_payload("..\\outside.txt", "blocked")
    fake_aiohttp = ModuleType("aiohttp")
    fake_aiohttp.ClientTimeout = lambda total: SimpleNamespace(total=total)
    fake_aiohttp.ClientSession = lambda: FakeZipClientSession(payload)
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    strategy = ZipExtractTransferStrategy(request_timeout_seconds=20)
    target_node = tmp_path / "extract"

    with pytest.raises(
        ValueError, match="Archive member escapes target directory"
    ):
        asyncio.run(
            strategy.execute_transfer(
                "https://example.com/archive.zip", target_node
            )
        )

    assert not (tmp_path / "outside.txt").exists()
    assert list(target_node.rglob("*")) == []
