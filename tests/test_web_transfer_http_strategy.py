import asyncio
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import IO

import definers.media.web_transfer as web_transfer_module
from definers.media.web_transfer import (
    AdaptiveHttpTransferStrategy,
    HttpChunkedTransferStrategy,
    HttpTransferCapabilities,
    HttpTransferPolicy,
    ParallelHttpRangeTransferStrategy,
    ParallelProcessHttpRangeTransferStrategy,
    create_http_transfer_strategy,
    http_transfer_capabilities,
    http_transfer_policy,
)
from definers.system.download_activity import (
    bind_download_activity_scope,
    clear_download_activity_scope,
    create_download_activity_scope,
    get_download_activity_snapshot,
)


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
    def __init__(self, payload: bytes, *, read_bufsize: int | None = None):
        self.payload = payload
        self.response = FakeNetworkResponse(payload)
        self.read_bufsize = read_bufsize

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


class FakeRangeResponse(FakeUrlResponse):
    def __init__(
        self,
        payload: bytes,
        *,
        start_index: int,
        end_index: int,
        total_bytes: int,
    ):
        super().__init__(payload[start_index : end_index + 1])
        self.headers = {
            "Content-Range": f"bytes {start_index}-{end_index}/{total_bytes}"
        }


def test_http_chunked_transfer_strategy_executes_async_download(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"abcdef"
    fake_aiohttp = ModuleType("aiohttp")
    fake_aiohttp.ClientTimeout = lambda total: SimpleNamespace(total=total)
    fake_aiohttp.ClientSession = lambda **kwargs: FakeClientSession(
        payload, **kwargs
    )
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


def test_http_chunked_transfer_strategy_passes_aiohttp_read_bufsize(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"abcdef"
    session_kwargs: list[int | None] = []

    def build_session(**kwargs):
        session_kwargs.append(kwargs.get("read_bufsize"))
        return FakeClientSession(payload, **kwargs)

    fake_aiohttp = ModuleType("aiohttp")
    fake_aiohttp.ClientTimeout = lambda total: SimpleNamespace(total=total)
    fake_aiohttp.ClientSession = build_session
    fake_aiofiles = ModuleType("aiofiles")
    fake_aiofiles.open = lambda target_node, mode: FakeAsyncFile(target_node)
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setitem(sys.modules, "aiofiles", fake_aiofiles)
    strategy = HttpChunkedTransferStrategy(
        chunk_size_bytes=524288,
        request_timeout_seconds=15,
    )

    result = asyncio.run(
        strategy.execute_transfer(
            "https://example.com/data.bin",
            tmp_path / "nested" / "payload.bin",
        )
    )

    assert result is True
    assert session_kwargs == [4194304]


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


def test_http_chunked_transfer_strategy_reports_transfer_activity(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"sync-payload"
    strategy = HttpChunkedTransferStrategy(chunk_size_bytes=4)
    target_node = tmp_path / "sync" / "payload.bin"
    fake_response = FakeUrlResponse(payload)
    fake_response.headers = {"Content-Length": str(len(payload))}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda source_uri, timeout: fake_response,
    )

    scope_id = create_download_activity_scope()
    with bind_download_activity_scope(scope_id):
        strategy._execute_transfer_sync(
            "https://example.com/data.bin",
            target_node,
        )
    activity_snapshot = get_download_activity_snapshot(scope_id)
    clear_download_activity_scope(scope_id)

    assert activity_snapshot is not None
    assert activity_snapshot.phase == "transfer"
    assert activity_snapshot.item_label == "payload.bin"
    assert activity_snapshot.bytes_downloaded == len(payload)
    assert activity_snapshot.bytes_total == len(payload)


def test_parallel_http_range_transfer_strategy_reports_target_name(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"parallel-payload"
    strategy = ParallelHttpRangeTransferStrategy(
        chunk_size_bytes=4,
        max_workers=2,
        min_parallel_size_bytes=1,
        part_size_bytes=8,
    )
    target_node = tmp_path / "parallel" / "payload.bin"

    monkeypatch.setattr(
        strategy, "_probe_remote_file", lambda source_uri: (len(payload), True)
    )

    def fake_urlopen(request, timeout):
        range_header = request.headers.get("Range") or request.headers.get(
            "range"
        )
        assert range_header is not None
        byte_range = range_header.removeprefix("bytes=")
        start_text, end_text = byte_range.split("-", 1)
        return FakeRangeResponse(
            payload,
            start_index=int(start_text),
            end_index=int(end_text),
            total_bytes=len(payload),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    scope_id = create_download_activity_scope()
    with bind_download_activity_scope(scope_id):
        strategy._execute_transfer_sync(
            "https://example.com/data.bin",
            target_node,
        )
    activity_snapshot = get_download_activity_snapshot(scope_id)
    clear_download_activity_scope(scope_id)

    assert target_node.read_bytes() == payload
    assert activity_snapshot is not None
    assert activity_snapshot.item_label == "payload.bin"
    assert activity_snapshot.bytes_downloaded == len(payload)
    assert activity_snapshot.bytes_total == len(payload)


def test_parallel_http_range_transfer_strategy_uses_mmap_offset_writer(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"parallel-payload"
    strategy = ParallelHttpRangeTransferStrategy(
        chunk_size_bytes=4,
        max_workers=2,
        min_parallel_size_bytes=1,
        part_size_bytes=8,
    )
    target_node = tmp_path / "parallel-mmap" / "payload.bin"
    mmap_calls: list[tuple[int, int, int]] = []

    monkeypatch.setattr(
        strategy, "_probe_remote_file", lambda source_uri: (len(payload), True)
    )

    def fake_urlopen(request, timeout):
        range_header = request.headers.get("Range") or request.headers.get(
            "range"
        )
        assert range_header is not None
        byte_range = range_header.removeprefix("bytes=")
        start_text, end_text = byte_range.split("-", 1)
        return FakeRangeResponse(
            payload,
            start_index=int(start_text),
            end_index=int(end_text),
            total_bytes=len(payload),
        )

    def fake_mmap_writer(
        response,
        staging_node: Path,
        *,
        start_index: int,
        total_bytes: int,
        chunk_size_bytes: int,
        on_chunk,
    ) -> None:
        mmap_calls.append((start_index, total_bytes, chunk_size_bytes))
        web_transfer_module._write_response_range_to_staging_via_file(
            response,
            staging_node,
            start_index=start_index,
            total_bytes=total_bytes,
            chunk_size_bytes=chunk_size_bytes,
            on_chunk=on_chunk,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        web_transfer_module,
        "_download_mmap_min_size_bytes",
        lambda: 1,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_write_response_range_to_staging_via_mmap",
        fake_mmap_writer,
    )

    strategy._execute_transfer_sync("https://example.com/data.bin", target_node)

    assert target_node.read_bytes() == payload
    assert mmap_calls == [
        (0, len(payload), 4),
        (8, len(payload), 4),
    ]


def test_parallel_http_range_transfer_strategy_uses_higher_default_limits(
    monkeypatch,
) -> None:
    monkeypatch.setattr(web_transfer_module.os, "cpu_count", lambda: 8)
    monkeypatch.delenv("DEFINERS_DOWNLOAD_MAX_WORKERS", raising=False)
    monkeypatch.delenv("DEFINERS_DOWNLOAD_CHUNK_SIZE_BYTES", raising=False)
    monkeypatch.delenv(
        "DEFINERS_DOWNLOAD_MIN_PARALLEL_SIZE_BYTES",
        raising=False,
    )
    monkeypatch.delenv("DEFINERS_DOWNLOAD_PART_SIZE_BYTES", raising=False)

    strategy = ParallelHttpRangeTransferStrategy()

    assert strategy.max_workers == 96
    assert strategy.chunk_size_bytes == 1048576
    assert strategy.min_parallel_size_bytes == 262144
    assert strategy.part_size_bytes == 1048576


def test_parallel_http_range_transfer_strategy_honors_env_overrides(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEFINERS_DOWNLOAD_MAX_WORKERS", "21")
    monkeypatch.setenv("DEFINERS_DOWNLOAD_CHUNK_SIZE_BYTES", "65536")
    monkeypatch.setenv(
        "DEFINERS_DOWNLOAD_MIN_PARALLEL_SIZE_BYTES",
        "131072",
    )
    monkeypatch.setenv("DEFINERS_DOWNLOAD_PART_SIZE_BYTES", "262144")

    strategy = ParallelHttpRangeTransferStrategy()

    assert strategy.max_workers == 21
    assert strategy.chunk_size_bytes == 65536
    assert strategy.min_parallel_size_bytes == 131072
    assert strategy.part_size_bytes == 262144


def test_http_transfer_capabilities_report_all_parallel_modes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        web_transfer_module,
        "_parallel_download_workers",
        lambda: 48,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_download_process_workers",
        lambda: 6,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_download_max_multiplexed_streams",
        lambda: 24,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_http2_runtime_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_http3_runtime_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_download_enable_multiplexing",
        lambda: True,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_download_enable_http3",
        lambda: True,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_download_http_protocol",
        lambda: "auto",
    )

    capabilities = http_transfer_capabilities()

    assert capabilities.protocol_preference == "auto"
    assert capabilities.http_range_requests is True
    assert capabilities.parallel_connections is True
    assert capabilities.separate_process_workers is True
    assert capabilities.http2_multiplexing is True
    assert capabilities.http2_runtime_ready is True
    assert capabilities.http3_multiplexing is True
    assert capabilities.http3_runtime_ready is True
    assert capabilities.quic_udp is True
    assert capabilities.max_parallel_connections == 48
    assert capabilities.max_process_workers == 6
    assert capabilities.max_multiplexed_streams == 24


def test_http_transfer_capabilities_disable_process_workers_in_daemon_runtime(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "DEFINERS_DOWNLOAD_ENABLE_PROCESS_WORKERS",
        raising=False,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_parallel_download_workers",
        lambda: 48,
    )
    monkeypatch.setattr(
        web_transfer_module.multiprocessing,
        "current_process",
        lambda: type("Proc", (), {"daemon": True})(),
    )

    capabilities = http_transfer_capabilities()

    assert capabilities.separate_process_workers is False
    assert capabilities.max_process_workers == 1


def test_download_process_workers_env_override_reenables_process_pool(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEFINERS_DOWNLOAD_ENABLE_PROCESS_WORKERS", "1")
    monkeypatch.setenv("DEFINERS_DOWNLOAD_PROCESS_WORKERS", "7")
    monkeypatch.setattr(
        web_transfer_module.multiprocessing,
        "current_process",
        lambda: type("Proc", (), {"daemon": True})(),
    )

    assert web_transfer_module._download_process_workers() == 7


def test_http_transfer_policy_uses_threaded_base_strategy_in_restricted_runtime(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        web_transfer_module,
        "http_transfer_capabilities",
        lambda: HttpTransferCapabilities(
            protocol_preference="http1",
            http_range_requests=True,
            parallel_connections=True,
            separate_process_workers=False,
            http2_multiplexing=False,
            http2_runtime_ready=False,
            http3_multiplexing=False,
            http3_runtime_ready=False,
            quic_udp=False,
            max_parallel_connections=48,
            max_process_workers=1,
            max_multiplexed_streams=24,
        ),
    )

    policy = http_transfer_policy()

    assert isinstance(policy, HttpTransferPolicy)
    assert policy.runtime_class == "restricted"
    assert policy.base_strategy_name == "http1-range-threaded"
    assert policy.strategy_names == ("http1-range-threaded",)


def test_create_http_transfer_strategy_uses_threaded_base_when_process_pool_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        web_transfer_module,
        "http_transfer_capabilities",
        lambda: HttpTransferCapabilities(
            protocol_preference="http1",
            http_range_requests=True,
            parallel_connections=True,
            separate_process_workers=False,
            http2_multiplexing=False,
            http2_runtime_ready=False,
            http3_multiplexing=False,
            http3_runtime_ready=False,
            quic_udp=False,
            max_parallel_connections=48,
            max_process_workers=1,
            max_multiplexed_streams=24,
        ),
    )

    strategy = create_http_transfer_strategy()

    assert isinstance(strategy, ParallelHttpRangeTransferStrategy)
    assert not isinstance(strategy, ParallelProcessHttpRangeTransferStrategy)


def test_create_http_transfer_strategy_prefers_http3_when_requested(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        web_transfer_module,
        "http_transfer_capabilities",
        lambda: HttpTransferCapabilities(
            protocol_preference="http3",
            http_range_requests=True,
            parallel_connections=True,
            separate_process_workers=True,
            http2_multiplexing=True,
            http2_runtime_ready=True,
            http3_multiplexing=True,
            http3_runtime_ready=True,
            quic_udp=True,
            max_parallel_connections=48,
            max_process_workers=6,
            max_multiplexed_streams=24,
        ),
    )

    strategy = create_http_transfer_strategy()

    assert isinstance(strategy, AdaptiveHttpTransferStrategy)
    assert [entry.strategy_name for entry in strategy.strategies] == [
        "http3-quic",
        "http2-multiplex",
        "http1-range-process",
    ]


def test_parallel_process_http_range_transfer_strategy_merges_part_files(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"parallel-process-payload"
    strategy = ParallelProcessHttpRangeTransferStrategy(
        chunk_size_bytes=4,
        max_workers=4,
        min_parallel_size_bytes=1,
        part_size_bytes=4,
        process_workers=2,
        min_process_size_bytes=1,
    )
    target_node = tmp_path / "parallel-process" / "payload.bin"

    monkeypatch.setattr(
        strategy,
        "_probe_remote_file",
        lambda source_uri: (len(payload), True),
    )

    def fake_download_range_part_file(
        source_uri: str,
        target_part_path: str,
        start_index: int,
        end_index: int,
        request_timeout_seconds: float,
        chunk_size_bytes: int,
    ) -> int:
        del source_uri, request_timeout_seconds, chunk_size_bytes
        part_payload = payload[start_index : end_index + 1]
        Path(target_part_path).write_bytes(part_payload)
        return len(part_payload)

    class FakeFuture:
        def __init__(self, value: int):
            self._value = value

        def result(self) -> int:
            return self._value

    class FakeProcessPoolExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self) -> "FakeProcessPoolExecutor":
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        def submit(self, function, *args):
            return FakeFuture(function(*args))

    monkeypatch.setattr(
        web_transfer_module,
        "_download_range_part_file",
        fake_download_range_part_file,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "ProcessPoolExecutor",
        FakeProcessPoolExecutor,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "as_completed",
        lambda futures: list(futures),
    )

    strategy._execute_transfer_sync("https://example.com/data.bin", target_node)

    assert target_node.read_bytes() == payload


def test_write_payload_to_path_prefers_mmap_for_large_payload(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"payload-through-mmap"
    target_node = tmp_path / "mmap" / "payload.bin"
    captured_calls: list[tuple[Path, bytes]] = []

    def fake_mmap_writer(
        resolved_target_node: Path,
        resolved_payload: bytes | bytearray | memoryview,
    ) -> None:
        payload_bytes = bytes(resolved_payload)
        captured_calls.append((resolved_target_node, payload_bytes))
        resolved_target_node.parent.mkdir(parents=True, exist_ok=True)
        resolved_target_node.write_bytes(payload_bytes)

    monkeypatch.setattr(
        web_transfer_module,
        "_download_mmap_min_size_bytes",
        lambda: 1,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_write_payload_to_path_via_mmap",
        fake_mmap_writer,
    )

    web_transfer_module._write_payload_to_path(target_node, payload)

    assert captured_calls == [(target_node, payload)]
    assert target_node.read_bytes() == payload


def test_merge_part_nodes_prefers_mmap_for_large_total_payload(
    monkeypatch, tmp_path: Path
) -> None:
    part_directory = tmp_path / "parts"
    part_directory.mkdir(parents=True, exist_ok=True)
    part_nodes = [
        part_directory / "000000.part",
        part_directory / "000001.part",
    ]
    part_nodes[0].write_bytes(b"hello ")
    part_nodes[1].write_bytes(b"world")
    staging_node = tmp_path / "payload.bin"
    captured_calls: list[tuple[tuple[Path, ...], Path]] = []

    def fake_mmap_merge(
        resolved_part_nodes: list[Path],
        resolved_staging_node: Path,
    ) -> None:
        captured_calls.append(
            (tuple(resolved_part_nodes), resolved_staging_node)
        )
        resolved_staging_node.write_bytes(
            b"".join(
                part_node.read_bytes() for part_node in resolved_part_nodes
            )
        )

    monkeypatch.setattr(
        web_transfer_module,
        "_download_mmap_min_size_bytes",
        lambda: 1,
    )
    monkeypatch.setattr(
        web_transfer_module,
        "_merge_part_nodes_via_mmap",
        fake_mmap_merge,
    )

    web_transfer_module._merge_part_nodes(part_nodes, staging_node)

    assert captured_calls == [(tuple(part_nodes), staging_node)]
    assert staging_node.read_bytes() == b"hello world"
