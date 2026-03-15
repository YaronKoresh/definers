import asyncio
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import IO

from definers.media.web_transfer import HttpChunkedTransferStrategy


class FakeAsyncFile:
    def __init__(self, target_node: Path):
        self.target_node = target_node
        self.handle: IO[bytes] | None = None

    async def __aenter__(self) -> "FakeAsyncFile":
        self.handle = open(self.target_node, "wb")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        if self.handle is not None:
            self.handle.close()
        return False

    async def write(self, data_chunk: bytes) -> None:
        if self.handle is None:
            raise RuntimeError("persistent storage is not open")
        self.handle.write(data_chunk)


class FakeChunkStream:
    def __init__(self, payload: bytes):
        self.payload = payload

    async def iter_chunked(self, chunk_size_bytes: int):
        for index in range(0, len(self.payload), chunk_size_bytes):
            yield self.payload[index : index + chunk_size_bytes]


class FakeNetworkResponse:
    def __init__(self, payload: bytes):
        self.content = FakeChunkStream(payload)
        self.raise_for_status_called = False

    async def __aenter__(self) -> "FakeNetworkResponse":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True


class FakeClientSession:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.response = FakeNetworkResponse(payload)

    async def __aenter__(self) -> "FakeClientSession":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def get(self, source_uri: str, timeout: object) -> FakeNetworkResponse:
        assert source_uri == "https://example.com/data.bin"
        assert getattr(timeout, "total") == 15
        return self.response


class FakeUrlResponse:
    def __init__(self, payload: bytes):
        self.payload_stream = io.BytesIO(payload)

    def __enter__(self) -> "FakeUrlResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def read(self, chunk_size_bytes: int = -1) -> bytes:
        return self.payload_stream.read(chunk_size_bytes)


def test_http_chunked_transfer_strategy_executes_async_download(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"abcdef"
    fake_aiohttp = ModuleType("aiohttp")
    fake_aiohttp.ClientTimeout = lambda total: SimpleNamespace(total=total)
    fake_aiohttp.ClientSession = lambda: FakeClientSession(payload)
    fake_aiofiles = ModuleType("aiofiles")
    fake_aiofiles.open = lambda target_node, mode: FakeAsyncFile(target_node)
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setitem(sys.modules, "aiofiles", fake_aiofiles)
    strategy = HttpChunkedTransferStrategy(
        chunk_size_bytes=2, request_timeout_seconds=15
    )
    target_node = tmp_path / "nested" / "payload.bin"

    result = asyncio.run(
        strategy.execute_transfer("https://example.com/data.bin", target_node)
    )

    assert result is True
    assert target_node.read_bytes() == payload


def test_http_chunked_transfer_strategy_executes_sync_download(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"sync-payload"
    strategy = HttpChunkedTransferStrategy(chunk_size_bytes=4)
    target_node = tmp_path / "sync" / "payload.bin"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda source_uri, timeout: FakeUrlResponse(payload),
    )

    strategy._execute_transfer_sync("https://example.com/data.bin", target_node)

    assert target_node.read_bytes() == payload
