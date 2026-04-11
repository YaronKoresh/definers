from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import logging
import math
import mmap
import multiprocessing
import os
import shutil
import ssl
import tempfile
import threading
import urllib.request
import zipfile
from collections.abc import Callable
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urlsplit

from definers.media.transfer.api import (
    add_to_path_windows,
    broadcast_path_change,
    download_and_unzip,
    download_file,
    execute_async_operation,
    extract_text,
    google_drive_download,
    linked_url,
    validate_network_url,
)
from definers.media.transfer.orchestrators import (
    ResourceRetrievalOrchestrator,
    TransferExecutionPolicy,
    create_http_orchestrator,
    create_zip_orchestrator,
)
from definers.media.transfer.policy import (
    HttpTransferCapabilities,
    HttpTransferPolicy,
    create_http_transfer_strategy,
    http_transfer_capabilities,
    http_transfer_policy,
)
from definers.system import log
from definers.system.download_activity import report_download_activity


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized_value = str(raw_value).strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    return default


def _read_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return max(int(str(raw_value).strip()), 1)
    except Exception:
        return default


def _parallel_download_workers() -> int:
    cpu_count = max(int(os.cpu_count() or 1), 1)
    return _read_positive_int_env(
        "DEFINERS_DOWNLOAD_MAX_WORKERS",
        cpu_count * 12,
    )


def _download_chunk_size_bytes() -> int:
    return _read_positive_int_env(
        "DEFINERS_DOWNLOAD_CHUNK_SIZE_BYTES",
        1024 * 1024,
    )


def _parallel_download_min_size_bytes() -> int:
    return _read_positive_int_env(
        "DEFINERS_DOWNLOAD_MIN_PARALLEL_SIZE_BYTES",
        256 * 1024,
    )


def _parallel_download_part_size_bytes() -> int:
    return _read_positive_int_env(
        "DEFINERS_DOWNLOAD_PART_SIZE_BYTES",
        1024 * 1024,
    )


def _download_mmap_min_size_bytes() -> int:
    return _read_positive_int_env(
        "DEFINERS_DOWNLOAD_MMAP_MIN_SIZE_BYTES",
        4 * 1024 * 1024,
    )


def _download_enable_process_workers() -> bool:
    return _read_bool_env(
        "DEFINERS_DOWNLOAD_ENABLE_PROCESS_WORKERS",
        False,
    )


def _download_process_workers() -> int:
    requested_workers = _read_positive_int_env(
        "DEFINERS_DOWNLOAD_PROCESS_WORKERS",
        max(int(os.cpu_count() or 1), 1),
    )
    if _download_enable_process_workers():
        return requested_workers
    with contextlib.suppress(Exception):
        if getattr(multiprocessing.current_process(), "daemon", False):
            return 1
    return min(requested_workers, _parallel_download_workers())


def _download_min_process_size_bytes() -> int:
    return _read_positive_int_env(
        "DEFINERS_DOWNLOAD_MIN_PROCESS_SIZE_BYTES",
        8 * 1024 * 1024,
    )


def _download_enable_multiplexing() -> bool:
    return _read_bool_env(
        "DEFINERS_DOWNLOAD_ENABLE_MULTIPLEXING",
        True,
    )


def _download_enable_http3() -> bool:
    return _read_bool_env("DEFINERS_DOWNLOAD_ENABLE_HTTP3", True)


def _download_http_protocol() -> str:
    normalized_value = (
        str(os.getenv("DEFINERS_DOWNLOAD_HTTP_PROTOCOL", "auto"))
        .strip()
        .lower()
    )
    if normalized_value in {"auto", "http1", "http2", "http3"}:
        return normalized_value
    return "auto"


def _download_max_multiplexed_streams() -> int:
    return _read_positive_int_env(
        "DEFINERS_DOWNLOAD_MAX_MULTIPLEXED_STREAMS",
        32,
    )


def _content_length(value: object) -> int | None:
    try:
        return max(int(str(value).strip()), 0)
    except Exception:
        return None


def _header_value(response: object, header_name: str) -> str | None:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if callable(getter):
        for candidate_name in (
            header_name,
            header_name.lower(),
            header_name.upper(),
        ):
            value = getter(candidate_name)
            if value is None:
                continue
            normalized_value = str(value).strip()
            if normalized_value:
                return normalized_value
    if isinstance(headers, dict):
        normalized_header_name = header_name.strip().lower()
        for current_name, current_value in headers.items():
            if str(current_name).strip().lower() != normalized_header_name:
                continue
            normalized_value = str(current_value).strip()
            if normalized_value:
                return normalized_value
    return None


def _content_range_length(response: object) -> int | None:
    return _content_range_length_from_value(
        _header_value(response, "Content-Range")
    )


def _aiohttp_read_bufsize_bytes(chunk_size_bytes: int) -> int:
    return max(4 * 1024 * 1024, max(int(chunk_size_bytes), 1) * 8)


def _create_aiohttp_client_session(
    aiohttp_module: Any,
    *,
    chunk_size_bytes: int,
) -> Any:
    try:
        return aiohttp_module.ClientSession(
            read_bufsize=_aiohttp_read_bufsize_bytes(chunk_size_bytes)
        )
    except TypeError:
        return aiohttp_module.ClientSession()


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _http2_runtime_ready() -> bool:
    return _download_enable_multiplexing() and _module_available("httpx")


def _http3_runtime_ready() -> bool:
    return (
        _download_enable_multiplexing()
        and _download_enable_http3()
        and _module_available("aioquic")
    )


class TransferStrategyUnsupportedError(RuntimeError):
    pass


class NetworkTransferStrategy(Protocol):
    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool: ...


def _transfer_item_label(source_uri: str, target_node: Path) -> str:
    target_name = str(target_node.name).strip()
    if target_name:
        return target_name
    source_name = Path(urlsplit(source_uri).path).name.strip()
    return source_name or source_uri


def _transfer_progress(
    source_uri: str,
    target_node: Path,
    *,
    detail: str,
    phase: str,
    bytes_downloaded: int | None = None,
    bytes_total: int | None = None,
    completed: int | None = None,
    total: int | None = None,
) -> None:
    report_download_activity(
        _transfer_item_label(source_uri, target_node),
        detail=detail,
        phase=phase,
        completed=completed,
        total=total,
        bytes_downloaded=bytes_downloaded,
        bytes_total=bytes_total,
    )


def _normalize_http_version(value: object) -> str:
    normalized_value = str(value or "").strip().upper()
    if normalized_value in {"HTTP/2", "HTTP/3", "HTTP/1.1", "HTTP/1.0"}:
        return normalized_value
    return normalized_value


def _content_range_length_from_value(value: str | None) -> int | None:
    if not value or "/" not in value:
        return None
    total_length_text = value.rsplit("/", 1)[-1].strip()
    if total_length_text == "*":
        return None
    return _content_length(total_length_text)


def _header_pairs_value(
    headers: tuple[tuple[bytes, bytes], ...] | list[tuple[bytes, bytes]],
    header_name: str,
) -> str | None:
    normalized_header_name = str(header_name).strip().lower().encode("ascii")
    for current_name, current_value in headers:
        if bytes(current_name).strip().lower() != normalized_header_name:
            continue
        normalized_value = bytes(current_value).decode("utf-8", errors="ignore")
        normalized_value = normalized_value.strip()
        if normalized_value:
            return normalized_value
    return None


class HttpChunkedTransferStrategy:
    strategy_name = "http1-chunked"

    def __init__(
        self,
        chunk_size_bytes: int | None = None,
        request_timeout_seconds: float = 30,
    ):
        self.chunk_size_bytes = (
            _download_chunk_size_bytes()
            if chunk_size_bytes is None
            else max(int(chunk_size_bytes), 1)
        )
        self.request_timeout_seconds = float(request_timeout_seconds)

    def _create_staging_target(self, target_node: Path) -> Path:
        target_node.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, staging_path = tempfile.mkstemp(
            prefix=f"{target_node.name}.",
            suffix=".part",
            dir=target_node.parent,
        )
        os.close(file_descriptor)
        return Path(staging_path)

    def _commit_staging_target(
        self, staging_node: Path, target_node: Path
    ) -> None:
        target_node.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_node, target_node)

    def _cleanup_staging_target(self, staging_node: Path) -> None:
        staging_node.unlink(missing_ok=True)

    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool:
        resolved_target_node = Path(target_node)
        try:
            import aiofiles
            import aiohttp
        except ImportError:
            await asyncio.to_thread(
                self._execute_transfer_sync,
                source_uri,
                resolved_target_node,
            )
            return True
        staging_node = self._create_staging_target(resolved_target_node)
        try:
            request_timeout = aiohttp.ClientTimeout(
                total=self.request_timeout_seconds
            )
            async with _create_aiohttp_client_session(
                aiohttp,
                chunk_size_bytes=self.chunk_size_bytes,
            ) as session:
                async with session.get(
                    source_uri, timeout=request_timeout
                ) as network_response:
                    network_response.raise_for_status()
                    total_bytes = _content_length(
                        getattr(
                            getattr(network_response, "headers", {}),
                            "get",
                            lambda _key, _default=None: None,
                        )("Content-Length")
                    )
                    async with aiofiles.open(
                        staging_node, "wb"
                    ) as persistent_storage:
                        downloaded_bytes = 0
                        if hasattr(network_response, "content") and hasattr(
                            network_response.content,
                            "iter_chunked",
                        ):
                            async for (
                                data_chunk
                            ) in network_response.content.iter_chunked(
                                self.chunk_size_bytes
                            ):
                                await persistent_storage.write(data_chunk)
                                downloaded_bytes += len(data_chunk)
                                _transfer_progress(
                                    source_uri,
                                    resolved_target_node,
                                    detail="Streaming artifact bytes.",
                                    phase="transfer",
                                    bytes_downloaded=downloaded_bytes,
                                    bytes_total=total_bytes,
                                )
                        else:
                            payload = await network_response.read()
                            await persistent_storage.write(payload)
                            _transfer_progress(
                                source_uri,
                                resolved_target_node,
                                detail="Streaming artifact bytes.",
                                phase="transfer",
                                bytes_downloaded=len(payload),
                                bytes_total=total_bytes,
                            )
            await asyncio.to_thread(
                self._commit_staging_target,
                staging_node,
                resolved_target_node,
            )
        except Exception:
            await asyncio.to_thread(self._cleanup_staging_target, staging_node)
            raise
        return True

    def _execute_transfer_sync(
        self, source_uri: str, target_node: Path
    ) -> None:
        resolved_target_node = Path(target_node)
        staging_node = self._create_staging_target(resolved_target_node)
        try:
            with urllib.request.urlopen(
                source_uri, timeout=self.request_timeout_seconds
            ) as response:
                total_bytes = _content_length(
                    getattr(
                        getattr(response, "headers", {}),
                        "get",
                        lambda _key, _default=None: None,
                    )("Content-Length")
                )
                with open(staging_node, "wb") as persistent_storage:
                    downloaded_bytes = 0
                    while True:
                        data_chunk = response.read(self.chunk_size_bytes)
                        if not data_chunk:
                            break
                        persistent_storage.write(data_chunk)
                        downloaded_bytes += len(data_chunk)
                        _transfer_progress(
                            source_uri,
                            resolved_target_node,
                            detail="Streaming artifact bytes.",
                            phase="transfer",
                            bytes_downloaded=downloaded_bytes,
                            bytes_total=total_bytes,
                        )
            self._commit_staging_target(staging_node, resolved_target_node)
        except Exception:
            self._cleanup_staging_target(staging_node)
            raise


def _multipart_part_directory(staging_node: Path) -> Path:
    return staging_node.parent / f"{staging_node.name}.parts"


def _multipart_part_node(part_directory: Path, part_index: int) -> Path:
    return part_directory / f"{part_index:06d}.part"


def _planned_byte_ranges(
    total_bytes: int,
    part_size_bytes: int,
) -> list[tuple[int, int]]:
    byte_ranges: list[tuple[int, int]] = []
    for start_index in range(0, total_bytes, part_size_bytes):
        end_index = min(start_index + part_size_bytes, total_bytes) - 1
        byte_ranges.append((start_index, end_index))
    return byte_ranges


def _should_use_mmap_write(total_bytes: int) -> bool:
    return total_bytes >= _download_mmap_min_size_bytes() and total_bytes > 0


def _write_payload_to_path_via_mmap(
    target_node: Path,
    payload: bytes | bytearray | memoryview,
) -> None:
    payload_view = (
        payload if isinstance(payload, memoryview) else memoryview(payload)
    )
    target_node.parent.mkdir(parents=True, exist_ok=True)
    with open(target_node, "w+b") as persistent_storage:
        persistent_storage.truncate(len(payload_view))
        if len(payload_view) == 0:
            return
        with mmap.mmap(
            persistent_storage.fileno(),
            len(payload_view),
            access=mmap.ACCESS_WRITE,
        ) as mapped_storage:
            mapped_storage[:] = payload_view
            mapped_storage.flush()


def _write_payload_to_path(
    target_node: Path,
    payload: bytes | bytearray | memoryview,
) -> None:
    payload_view = (
        payload if isinstance(payload, memoryview) else memoryview(payload)
    )
    if _should_use_mmap_write(len(payload_view)):
        try:
            _write_payload_to_path_via_mmap(target_node, payload_view)
            return
        except Exception:
            pass
    target_node.parent.mkdir(parents=True, exist_ok=True)
    with open(target_node, "wb") as persistent_storage:
        persistent_storage.write(payload_view)


def _merge_part_nodes_via_mmap(
    part_nodes: list[Path],
    staging_node: Path,
) -> None:
    total_bytes = sum(
        max(int(part_node.stat().st_size), 0)
        for part_node in part_nodes
        if part_node.exists()
    )
    staging_node.parent.mkdir(parents=True, exist_ok=True)
    with open(staging_node, "w+b") as persistent_storage:
        persistent_storage.truncate(total_bytes)
        if total_bytes == 0:
            return
        with mmap.mmap(
            persistent_storage.fileno(),
            total_bytes,
            access=mmap.ACCESS_WRITE,
        ) as mapped_storage:
            current_offset = 0
            for part_node in part_nodes:
                part_size = (
                    max(int(part_node.stat().st_size), 0)
                    if part_node.exists()
                    else 0
                )
                if part_size == 0:
                    continue
                with open(part_node, "rb") as part_storage:
                    if _should_use_mmap_write(part_size):
                        with mmap.mmap(
                            part_storage.fileno(),
                            part_size,
                            access=mmap.ACCESS_READ,
                        ) as mapped_part:
                            mapped_storage[
                                current_offset : current_offset + part_size
                            ] = mapped_part
                    else:
                        mapped_storage[
                            current_offset : current_offset + part_size
                        ] = part_storage.read()
                current_offset += part_size
            mapped_storage.flush()


def _merge_part_nodes(part_nodes: list[Path], staging_node: Path) -> None:
    total_bytes = sum(
        max(int(part_node.stat().st_size), 0)
        for part_node in part_nodes
        if part_node.exists()
    )
    if _should_use_mmap_write(total_bytes):
        try:
            _merge_part_nodes_via_mmap(part_nodes, staging_node)
            return
        except Exception:
            pass
    with open(staging_node, "wb") as persistent_storage:
        for part_node in part_nodes:
            with open(part_node, "rb") as part_storage:
                shutil.copyfileobj(part_storage, persistent_storage)


def _write_response_range_to_staging_via_file(
    response: object,
    staging_node: Path,
    *,
    start_index: int,
    total_bytes: int,
    chunk_size_bytes: int,
    on_chunk: Callable[[int], None] | None = None,
) -> None:
    del total_bytes
    with open(staging_node, "r+b") as persistent_storage:
        persistent_storage.seek(start_index)
        while True:
            data_chunk = response.read(chunk_size_bytes)
            if not data_chunk:
                break
            persistent_storage.write(data_chunk)
            if on_chunk is not None:
                on_chunk(len(data_chunk))


def _write_response_range_to_staging_via_mmap(
    response: object,
    staging_node: Path,
    *,
    start_index: int,
    total_bytes: int,
    chunk_size_bytes: int,
    on_chunk: Callable[[int], None] | None = None,
) -> None:
    with open(staging_node, "r+b") as persistent_storage:
        with mmap.mmap(
            persistent_storage.fileno(),
            total_bytes,
            access=mmap.ACCESS_WRITE,
        ) as mapped_storage:
            current_offset = start_index
            while True:
                data_chunk = response.read(chunk_size_bytes)
                if not data_chunk:
                    break
                next_offset = current_offset + len(data_chunk)
                mapped_storage[current_offset:next_offset] = data_chunk
                current_offset = next_offset
                if on_chunk is not None:
                    on_chunk(len(data_chunk))
            mapped_storage.flush()


def _write_response_range_to_staging(
    response: object,
    staging_node: Path,
    *,
    start_index: int,
    total_bytes: int,
    chunk_size_bytes: int,
    on_chunk: Callable[[int], None] | None = None,
) -> None:
    if _should_use_mmap_write(total_bytes):
        try:
            _write_response_range_to_staging_via_mmap(
                response,
                staging_node,
                start_index=start_index,
                total_bytes=total_bytes,
                chunk_size_bytes=chunk_size_bytes,
                on_chunk=on_chunk,
            )
            return
        except Exception:
            pass
    _write_response_range_to_staging_via_file(
        response,
        staging_node,
        start_index=start_index,
        total_bytes=total_bytes,
        chunk_size_bytes=chunk_size_bytes,
        on_chunk=on_chunk,
    )


def _cleanup_part_nodes(part_directory: Path, part_nodes: list[Path]) -> None:
    for part_node in part_nodes:
        part_node.unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        part_directory.rmdir()


def _download_range_part_file(
    source_uri: str,
    target_part_path: str,
    start_index: int,
    end_index: int,
    request_timeout_seconds: float,
    chunk_size_bytes: int,
) -> int:
    request = urllib.request.Request(
        source_uri,
        headers={"Range": f"bytes={start_index}-{end_index}"},
    )
    written_bytes = 0
    with urllib.request.urlopen(
        request, timeout=request_timeout_seconds
    ) as response:
        if _content_range_length(response) is None:
            raise RuntimeError(
                "Remote server did not honor the byte range request"
            )
        with open(target_part_path, "wb") as persistent_storage:
            while True:
                data_chunk = response.read(chunk_size_bytes)
                if not data_chunk:
                    break
                persistent_storage.write(data_chunk)
                written_bytes += len(data_chunk)
    return written_bytes


@dataclass(frozen=True, slots=True)
class HttpRemoteProbeResult:
    total_bytes: int | None
    supports_ranges: bool
    protocol: str


@dataclass(frozen=True, slots=True)
class Http3ResponsePayload:
    status_code: int
    headers: tuple[tuple[bytes, bytes], ...]
    body: bytes


class ParallelHttpRangeTransferStrategy(HttpChunkedTransferStrategy):
    strategy_name = "http1-range-threaded"

    def __init__(
        self,
        chunk_size_bytes: int | None = None,
        request_timeout_seconds: float = 30,
        *,
        max_workers: int | None = None,
        min_parallel_size_bytes: int | None = None,
        part_size_bytes: int | None = None,
    ):
        super().__init__(
            chunk_size_bytes=chunk_size_bytes,
            request_timeout_seconds=request_timeout_seconds,
        )
        self.max_workers = (
            _parallel_download_workers()
            if max_workers is None
            else max(int(max_workers), 1)
        )
        resolved_min_parallel_size_bytes = (
            _parallel_download_min_size_bytes()
            if min_parallel_size_bytes is None
            else max(int(min_parallel_size_bytes), 1)
        )
        resolved_part_size_bytes = (
            _parallel_download_part_size_bytes()
            if part_size_bytes is None
            else max(int(part_size_bytes), 1)
        )
        self.min_parallel_size_bytes = max(resolved_min_parallel_size_bytes, 1)
        self.part_size_bytes = max(
            resolved_part_size_bytes,
            self.chunk_size_bytes,
        )

    def _probe_remote_file(
        self,
        source_uri: str,
    ) -> tuple[int | None, bool]:
        total_bytes = None
        supports_ranges = False
        try:
            request = urllib.request.Request(source_uri, method="HEAD")
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_seconds
            ) as response:
                total_bytes = _content_length(
                    _header_value(response, "Content-Length")
                )
                supports_ranges = (
                    str(_header_value(response, "Accept-Ranges") or "")
                    .strip()
                    .lower()
                    == "bytes"
                )
        except Exception:
            total_bytes = None
            supports_ranges = False
        if total_bytes is not None and supports_ranges:
            return total_bytes, True
        try:
            request = urllib.request.Request(
                source_uri,
                headers={"Range": "bytes=0-0"},
            )
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_seconds
            ) as response:
                range_total = _content_range_length(response)
                if range_total is not None:
                    return range_total, True
                if total_bytes is None:
                    total_bytes = _content_length(
                        _header_value(response, "Content-Length")
                    )
        except Exception:
            return total_bytes, False
        return total_bytes, False

    def _download_ranges(
        self,
        source_uri: str,
        staging_node: Path,
        *,
        target_node: Path,
        total_bytes: int,
    ) -> None:
        from definers.system.download_activity import (
            bind_download_activity_scope,
            current_download_activity_scope,
        )

        planned_parts = max(
            int(math.ceil(total_bytes / self.part_size_bytes)), 1
        )
        worker_count = max(1, min(self.max_workers, planned_parts))
        if worker_count <= 1:
            raise RuntimeError("parallel range download is not needed")
        byte_ranges = _planned_byte_ranges(total_bytes, self.part_size_bytes)
        downloaded_bytes = 0
        download_lock = threading.Lock()
        activity_scope_id = current_download_activity_scope()

        def download_range(start_index: int, end_index: int) -> None:
            def run_range_download() -> None:
                nonlocal downloaded_bytes

                request = urllib.request.Request(
                    source_uri,
                    headers={"Range": f"bytes={start_index}-{end_index}"},
                )
                with urllib.request.urlopen(
                    request, timeout=self.request_timeout_seconds
                ) as response:
                    if _content_range_length(response) is None:
                        raise RuntimeError(
                            "Remote server did not honor the byte range request"
                        )

                    def on_chunk(chunk_length: int) -> None:
                        nonlocal downloaded_bytes

                        with download_lock:
                            downloaded_bytes += chunk_length
                            progress_value = downloaded_bytes
                        _transfer_progress(
                            source_uri,
                            target_node,
                            detail="Streaming artifact bytes.",
                            phase="transfer",
                            bytes_downloaded=progress_value,
                            bytes_total=total_bytes,
                        )

                    _write_response_range_to_staging(
                        response,
                        staging_node,
                        start_index=start_index,
                        total_bytes=total_bytes,
                        chunk_size_bytes=self.chunk_size_bytes,
                        on_chunk=on_chunk,
                    )

            scope_context = (
                bind_download_activity_scope(activity_scope_id)
                if activity_scope_id is not None
                else contextlib.nullcontext()
            )
            with scope_context:
                run_range_download()

        with open(staging_node, "wb") as persistent_storage:
            persistent_storage.truncate(total_bytes)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(download_range, start_index, end_index)
                for start_index, end_index in byte_ranges
            ]
            for future in futures:
                future.result()

    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool:
        return await asyncio.to_thread(
            self._execute_transfer_sync, source_uri, target_node
        )

    def _execute_transfer_sync(
        self, source_uri: str, target_node: Path
    ) -> None:
        total_bytes, supports_ranges = self._probe_remote_file(source_uri)
        if (
            total_bytes is None
            or not supports_ranges
            or total_bytes < self.min_parallel_size_bytes
            or self.max_workers <= 1
        ):
            super()._execute_transfer_sync(source_uri, target_node)
            return
        staging_node = self._create_staging_target(target_node)
        try:
            self._download_ranges(
                source_uri,
                staging_node,
                target_node=target_node,
                total_bytes=total_bytes,
            )
            self._commit_staging_target(staging_node, target_node)
        except Exception:
            self._cleanup_staging_target(staging_node)
            raise


class ParallelProcessHttpRangeTransferStrategy(
    ParallelHttpRangeTransferStrategy
):
    strategy_name = "http1-range-process"

    def __init__(
        self,
        chunk_size_bytes: int | None = None,
        request_timeout_seconds: float = 30,
        *,
        max_workers: int | None = None,
        min_parallel_size_bytes: int | None = None,
        part_size_bytes: int | None = None,
        process_workers: int | None = None,
        min_process_size_bytes: int | None = None,
    ):
        super().__init__(
            chunk_size_bytes=chunk_size_bytes,
            request_timeout_seconds=request_timeout_seconds,
            max_workers=max_workers,
            min_parallel_size_bytes=min_parallel_size_bytes,
            part_size_bytes=part_size_bytes,
        )
        self.process_workers = (
            _download_process_workers()
            if process_workers is None
            else max(int(process_workers), 1)
        )
        self.min_process_size_bytes = (
            _download_min_process_size_bytes()
            if min_process_size_bytes is None
            else max(int(min_process_size_bytes), 1)
        )

    def _use_process_workers(self, total_bytes: int) -> bool:
        return (
            self.process_workers > 1
            and self.max_workers > 1
            and total_bytes >= self.min_process_size_bytes
        )

    def _download_ranges(
        self,
        source_uri: str,
        staging_node: Path,
        *,
        target_node: Path,
        total_bytes: int,
    ) -> None:
        if not self._use_process_workers(total_bytes):
            super()._download_ranges(
                source_uri,
                staging_node,
                target_node=target_node,
                total_bytes=total_bytes,
            )
            return
        byte_ranges = _planned_byte_ranges(total_bytes, self.part_size_bytes)
        worker_count = max(1, min(self.process_workers, len(byte_ranges)))
        if worker_count <= 1:
            super()._download_ranges(
                source_uri,
                staging_node,
                target_node=target_node,
                total_bytes=total_bytes,
            )
            return
        part_directory = _multipart_part_directory(staging_node)
        part_directory.mkdir(parents=True, exist_ok=True)
        part_nodes = [
            _multipart_part_node(part_directory, index)
            for index in range(len(byte_ranges))
        ]
        downloaded_bytes = 0
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _download_range_part_file,
                        source_uri,
                        str(part_nodes[index]),
                        start_index,
                        end_index,
                        self.request_timeout_seconds,
                        self.chunk_size_bytes,
                    ): index
                    for index, (start_index, end_index) in enumerate(
                        byte_ranges
                    )
                }
                for future in as_completed(future_map):
                    downloaded_bytes += future.result()
                    _transfer_progress(
                        source_uri,
                        target_node,
                        detail="Streaming artifact bytes.",
                        phase="transfer",
                        bytes_downloaded=downloaded_bytes,
                        bytes_total=total_bytes,
                    )
            _merge_part_nodes(part_nodes, staging_node)
        except Exception:
            _cleanup_part_nodes(part_directory, part_nodes)
            raise
        _cleanup_part_nodes(part_directory, part_nodes)


class AdaptiveHttpTransferStrategy:
    strategy_name = "adaptive-http"

    def __init__(self, strategies: list[NetworkTransferStrategy]):
        self.strategies = tuple(strategies)

    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool:
        last_error: Exception | None = None
        for strategy in self.strategies:
            try:
                return await strategy.execute_transfer(source_uri, target_node)
            except Exception as error:
                last_error = error
                strategy_label = getattr(
                    strategy,
                    "strategy_name",
                    type(strategy).__name__,
                )
                if isinstance(error, TransferStrategyUnsupportedError):
                    logging.getLogger(__name__).debug(
                        "Transfer strategy %s is unavailable for %s: %s",
                        strategy_label,
                        source_uri,
                        error,
                    )
                    continue
                logging.getLogger(__name__).warning(
                    "Transfer strategy %s failed for %s: %s",
                    strategy_label,
                    source_uri,
                    error,
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError("No HTTP transfer strategies are configured.")


class Http2MultiplexedRangeTransferStrategy(ParallelHttpRangeTransferStrategy):
    strategy_name = "http2-multiplex"

    def __init__(
        self,
        chunk_size_bytes: int | None = None,
        request_timeout_seconds: float = 30,
        *,
        max_workers: int | None = None,
        min_parallel_size_bytes: int | None = None,
        part_size_bytes: int | None = None,
        max_streams: int | None = None,
    ):
        super().__init__(
            chunk_size_bytes=chunk_size_bytes,
            request_timeout_seconds=request_timeout_seconds,
            max_workers=max_workers,
            min_parallel_size_bytes=min_parallel_size_bytes,
            part_size_bytes=part_size_bytes,
        )
        self.max_streams = (
            _download_max_multiplexed_streams()
            if max_streams is None
            else max(int(max_streams), 1)
        )

    async def _probe_remote_file_http2(
        self,
        client: Any,
        source_uri: str,
    ) -> HttpRemoteProbeResult:
        total_bytes = None
        supports_ranges = False
        protocol = ""
        try:
            response = await client.head(source_uri)
            response.raise_for_status()
            protocol = _normalize_http_version(
                getattr(response, "http_version", "")
            )
            total_bytes = _content_length(
                _header_value(response, "Content-Length")
            )
            supports_ranges = (
                str(_header_value(response, "Accept-Ranges") or "")
                .strip()
                .lower()
                == "bytes"
            )
            if total_bytes is not None and supports_ranges:
                return HttpRemoteProbeResult(total_bytes, True, protocol)
        except Exception:
            total_bytes = None
            supports_ranges = False
        try:
            response = await client.get(
                source_uri,
                headers={"Range": "bytes=0-0"},
            )
            response.raise_for_status()
            protocol = _normalize_http_version(
                getattr(response, "http_version", protocol)
            )
            range_total = _content_range_length(response)
            if range_total is not None:
                return HttpRemoteProbeResult(range_total, True, protocol)
            if total_bytes is None:
                total_bytes = _content_length(
                    _header_value(response, "Content-Length")
                )
        except Exception:
            return HttpRemoteProbeResult(total_bytes, False, protocol)
        return HttpRemoteProbeResult(total_bytes, False, protocol)

    async def _download_single_http2(
        self,
        client: Any,
        source_uri: str,
        target_node: Path,
        *,
        total_bytes: int | None,
    ) -> bool:
        staging_node = self._create_staging_target(target_node)
        try:
            async with client.stream("GET", source_uri) as response:
                if (
                    _normalize_http_version(
                        getattr(response, "http_version", "")
                    )
                    != "HTTP/2"
                ):
                    raise TransferStrategyUnsupportedError(
                        "Remote server did not negotiate HTTP/2."
                    )
                response.raise_for_status()
                downloaded_bytes = 0
                with open(staging_node, "wb") as persistent_storage:
                    async for data_chunk in response.aiter_bytes(
                        self.chunk_size_bytes
                    ):
                        persistent_storage.write(data_chunk)
                        downloaded_bytes += len(data_chunk)
                        _transfer_progress(
                            source_uri,
                            target_node,
                            detail="Streaming artifact bytes over HTTP/2.",
                            phase="transfer",
                            bytes_downloaded=downloaded_bytes,
                            bytes_total=total_bytes,
                        )
            self._commit_staging_target(staging_node, target_node)
        except Exception:
            self._cleanup_staging_target(staging_node)
            raise
        return True

    async def _download_ranges_http2(
        self,
        client: Any,
        source_uri: str,
        target_node: Path,
        *,
        total_bytes: int,
    ) -> bool:
        byte_ranges = _planned_byte_ranges(total_bytes, self.part_size_bytes)
        worker_count = max(
            1,
            min(self.max_workers, self.max_streams, len(byte_ranges)),
        )
        if worker_count <= 1:
            return await self._download_single_http2(
                client,
                source_uri,
                target_node,
                total_bytes=total_bytes,
            )
        staging_node = self._create_staging_target(target_node)
        part_directory = _multipart_part_directory(staging_node)
        part_directory.mkdir(parents=True, exist_ok=True)
        part_nodes = [
            _multipart_part_node(part_directory, index)
            for index in range(len(byte_ranges))
        ]
        progress_lock = asyncio.Lock()
        concurrency_lock = asyncio.Semaphore(worker_count)
        downloaded_bytes = 0

        async def download_part(
            part_index: int,
            start_index: int,
            end_index: int,
        ) -> None:
            nonlocal downloaded_bytes
            async with concurrency_lock:
                async with client.stream(
                    "GET",
                    source_uri,
                    headers={"Range": f"bytes={start_index}-{end_index}"},
                ) as response:
                    if (
                        _normalize_http_version(
                            getattr(response, "http_version", "")
                        )
                        != "HTTP/2"
                    ):
                        raise TransferStrategyUnsupportedError(
                            "Remote server did not keep the transfer on HTTP/2."
                        )
                    response.raise_for_status()
                    if _content_range_length(response) is None:
                        raise TransferStrategyUnsupportedError(
                            "Remote server did not honor HTTP/2 range requests."
                        )
                    payload = bytearray()
                    async for data_chunk in response.aiter_bytes(
                        self.chunk_size_bytes
                    ):
                        payload.extend(data_chunk)
                        async with progress_lock:
                            downloaded_bytes += len(data_chunk)
                            progress_value = downloaded_bytes
                        _transfer_progress(
                            source_uri,
                            target_node,
                            detail="Multiplexing artifact bytes over HTTP/2.",
                            phase="transfer",
                            bytes_downloaded=progress_value,
                            bytes_total=total_bytes,
                        )
                    await asyncio.to_thread(
                        _write_payload_to_path,
                        part_nodes[part_index],
                        bytes(payload),
                    )

        try:
            await asyncio.gather(
                *(
                    download_part(part_index, start_index, end_index)
                    for part_index, (start_index, end_index) in enumerate(
                        byte_ranges
                    )
                )
            )
            await asyncio.to_thread(
                _merge_part_nodes,
                part_nodes,
                staging_node,
            )
            await asyncio.to_thread(
                self._commit_staging_target,
                staging_node,
                target_node,
            )
        except Exception:
            await asyncio.to_thread(self._cleanup_staging_target, staging_node)
            await asyncio.to_thread(
                _cleanup_part_nodes,
                part_directory,
                part_nodes,
            )
            raise
        await asyncio.to_thread(
            _cleanup_part_nodes,
            part_directory,
            part_nodes,
        )
        return True

    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool:
        if not _download_enable_multiplexing():
            raise TransferStrategyUnsupportedError(
                "HTTP/2 multiplexing is disabled."
            )
        try:
            import httpx
        except Exception as error:
            raise TransferStrategyUnsupportedError(
                "HTTP/2 runtime is unavailable."
            ) from error
        try:
            timeout = httpx.Timeout(self.request_timeout_seconds)
            limits = httpx.Limits(
                max_connections=1, max_keepalive_connections=1
            )
            async with httpx.AsyncClient(
                http2=True,
                follow_redirects=True,
                timeout=timeout,
                limits=limits,
            ) as client:
                probe = await self._probe_remote_file_http2(client, source_uri)
                if probe.protocol != "HTTP/2":
                    raise TransferStrategyUnsupportedError(
                        "Remote server did not negotiate HTTP/2."
                    )
                if (
                    probe.total_bytes is None
                    or not probe.supports_ranges
                    or probe.total_bytes < self.min_parallel_size_bytes
                    or self.max_workers <= 1
                ):
                    return await self._download_single_http2(
                        client,
                        source_uri,
                        target_node,
                        total_bytes=probe.total_bytes,
                    )
                return await self._download_ranges_http2(
                    client,
                    source_uri,
                    target_node,
                    total_bytes=probe.total_bytes,
                )
        except TransferStrategyUnsupportedError:
            raise
        except Exception as error:
            raise TransferStrategyUnsupportedError(
                "HTTP/2 transfer could not be completed."
            ) from error


class Http3MultiplexedRangeTransferStrategy(ParallelHttpRangeTransferStrategy):
    strategy_name = "http3-quic"

    def __init__(
        self,
        chunk_size_bytes: int | None = None,
        request_timeout_seconds: float = 30,
        *,
        max_workers: int | None = None,
        min_parallel_size_bytes: int | None = None,
        part_size_bytes: int | None = None,
        max_streams: int | None = None,
    ):
        super().__init__(
            chunk_size_bytes=chunk_size_bytes,
            request_timeout_seconds=request_timeout_seconds,
            max_workers=max_workers,
            min_parallel_size_bytes=min_parallel_size_bytes,
            part_size_bytes=part_size_bytes,
        )
        self.max_streams = (
            _download_max_multiplexed_streams()
            if max_streams is None
            else max(int(max_streams), 1)
        )

    async def _request_http3(
        self,
        client: Any,
        method: str,
        source_uri: str,
        headers: dict[str, str] | None = None,
    ) -> Http3ResponsePayload:
        return await asyncio.wait_for(
            client.request(method, source_uri, headers=headers),
            timeout=self.request_timeout_seconds,
        )

    async def _probe_remote_file_http3(
        self,
        client: Any,
        source_uri: str,
    ) -> HttpRemoteProbeResult:
        total_bytes = None
        supports_ranges = False
        try:
            response = await self._request_http3(client, "HEAD", source_uri)
            if response.status_code >= 400:
                raise RuntimeError(str(response.status_code))
            total_bytes = _content_length(
                _header_pairs_value(response.headers, "content-length")
            )
            supports_ranges = (
                str(
                    _header_pairs_value(response.headers, "accept-ranges") or ""
                )
                .strip()
                .lower()
                == "bytes"
            )
            if total_bytes is not None and supports_ranges:
                return HttpRemoteProbeResult(total_bytes, True, "HTTP/3")
        except Exception:
            total_bytes = None
            supports_ranges = False
        try:
            response = await self._request_http3(
                client,
                "GET",
                source_uri,
                headers={"range": "bytes=0-0"},
            )
            if response.status_code >= 400:
                raise RuntimeError(str(response.status_code))
            range_total = _content_range_length_from_value(
                _header_pairs_value(response.headers, "content-range")
            )
            if range_total is not None:
                return HttpRemoteProbeResult(range_total, True, "HTTP/3")
            if total_bytes is None:
                total_bytes = _content_length(
                    _header_pairs_value(response.headers, "content-length")
                )
        except Exception:
            return HttpRemoteProbeResult(total_bytes, False, "HTTP/3")
        return HttpRemoteProbeResult(total_bytes, False, "HTTP/3")

    async def _download_single_http3(
        self,
        client: Any,
        source_uri: str,
        target_node: Path,
        *,
        total_bytes: int,
    ) -> bool:
        staging_node = self._create_staging_target(target_node)
        try:
            response = await self._request_http3(client, "GET", source_uri)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP/3 transfer failed with status {response.status_code}."
                )
            await asyncio.to_thread(
                _write_payload_to_path,
                staging_node,
                response.body,
            )
            _transfer_progress(
                source_uri,
                target_node,
                detail="Streaming artifact bytes over HTTP/3.",
                phase="transfer",
                bytes_downloaded=len(response.body),
                bytes_total=total_bytes,
            )
            await asyncio.to_thread(
                self._commit_staging_target,
                staging_node,
                target_node,
            )
        except Exception:
            await asyncio.to_thread(self._cleanup_staging_target, staging_node)
            raise
        return True

    async def _download_ranges_http3(
        self,
        client: Any,
        source_uri: str,
        target_node: Path,
        *,
        total_bytes: int,
    ) -> bool:
        byte_ranges = _planned_byte_ranges(total_bytes, self.part_size_bytes)
        worker_count = max(
            1,
            min(self.max_workers, self.max_streams, len(byte_ranges)),
        )
        if worker_count <= 1:
            return await self._download_single_http3(
                client,
                source_uri,
                target_node,
                total_bytes=total_bytes,
            )
        staging_node = self._create_staging_target(target_node)
        part_directory = _multipart_part_directory(staging_node)
        part_directory.mkdir(parents=True, exist_ok=True)
        part_nodes = [
            _multipart_part_node(part_directory, index)
            for index in range(len(byte_ranges))
        ]
        concurrency_lock = asyncio.Semaphore(worker_count)
        progress_lock = asyncio.Lock()
        downloaded_bytes = 0

        async def download_part(
            part_index: int,
            start_index: int,
            end_index: int,
        ) -> None:
            nonlocal downloaded_bytes
            async with concurrency_lock:
                response = await self._request_http3(
                    client,
                    "GET",
                    source_uri,
                    headers={"range": f"bytes={start_index}-{end_index}"},
                )
                if response.status_code < 200 or response.status_code >= 300:
                    raise RuntimeError(
                        f"HTTP/3 range transfer failed with status {response.status_code}."
                    )
                if (
                    _content_range_length_from_value(
                        _header_pairs_value(response.headers, "content-range")
                    )
                    is None
                ):
                    raise TransferStrategyUnsupportedError(
                        "Remote server did not honor HTTP/3 range requests."
                    )
                await asyncio.to_thread(
                    _write_payload_to_path,
                    part_nodes[part_index],
                    response.body,
                )
                async with progress_lock:
                    downloaded_bytes += len(response.body)
                    progress_value = downloaded_bytes
                _transfer_progress(
                    source_uri,
                    target_node,
                    detail="Multiplexing artifact bytes over HTTP/3.",
                    phase="transfer",
                    bytes_downloaded=progress_value,
                    bytes_total=total_bytes,
                )

        try:
            await asyncio.gather(
                *(
                    download_part(part_index, start_index, end_index)
                    for part_index, (start_index, end_index) in enumerate(
                        byte_ranges
                    )
                )
            )
            await asyncio.to_thread(
                _merge_part_nodes,
                part_nodes,
                staging_node,
            )
            await asyncio.to_thread(
                self._commit_staging_target,
                staging_node,
                target_node,
            )
        except Exception:
            await asyncio.to_thread(self._cleanup_staging_target, staging_node)
            await asyncio.to_thread(
                _cleanup_part_nodes,
                part_directory,
                part_nodes,
            )
            raise
        await asyncio.to_thread(
            _cleanup_part_nodes,
            part_directory,
            part_nodes,
        )
        return True

    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool:
        if not _download_enable_multiplexing() or not _download_enable_http3():
            raise TransferStrategyUnsupportedError(
                "HTTP/3 support is disabled."
            )
        try:
            from aioquic.asyncio.client import connect
            from aioquic.asyncio.protocol import QuicConnectionProtocol
            from aioquic.h3.connection import H3_ALPN, H3Connection
            from aioquic.h3.events import DataReceived, HeadersReceived
            from aioquic.quic.configuration import QuicConfiguration
        except Exception as error:
            raise TransferStrategyUnsupportedError(
                "HTTP/3 runtime is unavailable."
            ) from error

        class _Http3Client(QuicConnectionProtocol):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self._http = H3Connection(self._quic)
                self._headers: dict[int, list[tuple[bytes, bytes]]] = {}
                self._bodies: dict[int, bytearray] = {}
                self._waiters: dict[
                    int, asyncio.Future[Http3ResponsePayload]
                ] = {}

            async def request(
                self,
                method: str,
                request_url: str,
                headers: dict[str, str] | None = None,
            ) -> Http3ResponsePayload:
                parsed_url = urlsplit(request_url)
                path = parsed_url.path or "/"
                if parsed_url.query:
                    path = f"{path}?{parsed_url.query}"
                stream_id = self._quic.get_next_available_stream_id()
                self._headers[stream_id] = []
                self._bodies[stream_id] = bytearray()
                waiter = asyncio.get_running_loop().create_future()
                self._waiters[stream_id] = waiter
                request_headers = [
                    (b":method", method.encode("ascii")),
                    (
                        b":scheme",
                        (parsed_url.scheme or "https").encode("ascii"),
                    ),
                    (b":authority", parsed_url.netloc.encode("utf-8")),
                    (b":path", path.encode("utf-8")),
                    (b"user-agent", b"definers/aioquic"),
                ]
                if headers is not None:
                    request_headers.extend(
                        (
                            str(key).strip().lower().encode("ascii"),
                            str(value).encode("utf-8"),
                        )
                        for key, value in headers.items()
                    )
                self._http.send_headers(
                    stream_id=stream_id,
                    headers=request_headers,
                    end_stream=True,
                )
                self.transmit()
                return await asyncio.shield(waiter)

            def quic_event_received(self, event: object) -> None:
                for http_event in self._http.handle_event(event):
                    if isinstance(http_event, HeadersReceived):
                        self._headers.setdefault(
                            http_event.stream_id, []
                        ).extend(tuple(http_event.headers))
                        if http_event.stream_ended:
                            self._complete(http_event.stream_id)
                    elif isinstance(http_event, DataReceived):
                        self._bodies.setdefault(
                            http_event.stream_id, bytearray()
                        ).extend(http_event.data)
                        if http_event.stream_ended:
                            self._complete(http_event.stream_id)

            def _complete(self, stream_id: int) -> None:
                waiter = self._waiters.pop(stream_id, None)
                if waiter is None or waiter.done():
                    return
                headers = tuple(self._headers.pop(stream_id, ()))
                body = bytes(self._bodies.pop(stream_id, bytearray()))
                status_text = _header_pairs_value(headers, ":status") or "0"
                waiter.set_result(
                    Http3ResponsePayload(
                        status_code=int(status_text),
                        headers=headers,
                        body=body,
                    )
                )

        parsed_url = urlsplit(source_uri)
        host = str(parsed_url.hostname or "").strip()
        if parsed_url.scheme.lower() != "https" or not host:
            raise TransferStrategyUnsupportedError(
                "HTTP/3 requires an HTTPS origin."
            )
        configuration = QuicConfiguration(
            is_client=True,
            alpn_protocols=H3_ALPN,
        )
        configuration.verify_mode = ssl.CERT_REQUIRED
        with contextlib.suppress(Exception):
            import certifi

            configuration.load_verify_locations(certifi.where())
        try:
            async with connect(
                host,
                int(parsed_url.port or 443),
                configuration=configuration,
                create_protocol=_Http3Client,
                wait_connected=True,
            ) as client:
                probe = await self._probe_remote_file_http3(client, source_uri)
                if probe.protocol != "HTTP/3":
                    raise TransferStrategyUnsupportedError(
                        "Remote server did not negotiate HTTP/3."
                    )
                if probe.total_bytes is None:
                    raise TransferStrategyUnsupportedError(
                        "HTTP/3 probe could not determine the remote size."
                    )
                if (
                    not probe.supports_ranges
                    or probe.total_bytes < self.min_parallel_size_bytes
                    or self.max_workers <= 1
                ):
                    return await self._download_single_http3(
                        client,
                        source_uri,
                        target_node,
                        total_bytes=probe.total_bytes,
                    )
                return await self._download_ranges_http3(
                    client,
                    source_uri,
                    target_node,
                    total_bytes=probe.total_bytes,
                )
        except TransferStrategyUnsupportedError:
            raise
        except Exception as error:
            raise TransferStrategyUnsupportedError(
                "HTTP/3 transfer could not be completed."
            ) from error


class ZipExtractTransferStrategy:
    def __init__(
        self,
        chunk_size_bytes: int = 262144,
        request_timeout_seconds: float = 60,
        download_strategy: NetworkTransferStrategy | None = None,
    ):
        self.chunk_size_bytes = chunk_size_bytes
        self.request_timeout_seconds = request_timeout_seconds
        self.download_strategy = download_strategy

    def _create_archive_staging_target(self, target_node: Path) -> Path:
        target_node.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, staging_path = tempfile.mkstemp(
            prefix="archive.",
            suffix=".zip",
            dir=target_node.parent,
        )
        os.close(file_descriptor)
        return Path(staging_path)

    def _resolve_archive_member_path(
        self, resolved_target_root: Path, archive_member_name: str
    ) -> Path:
        normalized_member_name = str(archive_member_name).replace("\\", "/")
        member_path = PurePosixPath(normalized_member_name)
        member_parts = tuple(
            part for part in member_path.parts if part not in ("", ".")
        )
        if (
            member_path.is_absolute()
            or not member_parts
            or any(part == ".." or part.endswith(":") for part in member_parts)
        ):
            raise ValueError("Archive member escapes target directory.")
        destination_node = resolved_target_root.joinpath(
            *member_parts
        ).resolve()
        try:
            destination_node.relative_to(resolved_target_root)
        except ValueError as error:
            value_error = ValueError("Archive member escapes target directory.")
            value_error.__cause__ = error
            raise value_error
        return destination_node

    def _extract_archive(
        self, archive_context: zipfile.ZipFile, target_node: Path
    ) -> None:
        target_node.mkdir(parents=True, exist_ok=True)
        resolved_target_root = target_node.resolve()
        archive_members = archive_context.infolist()
        for member_index, archive_member in enumerate(archive_members, start=1):
            report_download_activity(
                str(target_node.name or resolved_target_root.name),
                detail=archive_member.filename,
                phase="extract",
                completed=member_index,
                total=len(archive_members),
            )
            destination_node = self._resolve_archive_member_path(
                resolved_target_root, archive_member.filename
            )
            if archive_member.is_dir():
                destination_node.mkdir(parents=True, exist_ok=True)
                continue
            destination_node.parent.mkdir(parents=True, exist_ok=True)
            with archive_context.open(archive_member) as archive_member_stream:
                with open(destination_node, "wb") as persistent_storage:
                    shutil.copyfileobj(
                        archive_member_stream, persistent_storage
                    )

    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool:
        if self.download_strategy is not None:
            archive_node = self._create_archive_staging_target(target_node)
            try:
                await self.download_strategy.execute_transfer(
                    source_uri,
                    archive_node,
                )
                with zipfile.ZipFile(archive_node) as archive_context:
                    self._extract_archive(archive_context, target_node)
            finally:
                archive_node.unlink(missing_ok=True)
            return True
        try:
            import aiohttp
        except ImportError:
            await asyncio.to_thread(
                self._execute_transfer_sync, source_uri, target_node
            )
            return True
        target_node.mkdir(parents=True, exist_ok=True)
        request_timeout = aiohttp.ClientTimeout(
            total=self.request_timeout_seconds
        )
        async with _create_aiohttp_client_session(
            aiohttp,
            chunk_size_bytes=self.chunk_size_bytes,
        ) as session:
            async with session.get(
                source_uri, timeout=request_timeout
            ) as network_response:
                network_response.raise_for_status()
                total_bytes = _content_length(
                    getattr(
                        getattr(network_response, "headers", {}),
                        "get",
                        lambda _key, _default=None: None,
                    )("Content-Length")
                )
                memory_buffer = io.BytesIO()
                if hasattr(network_response, "content") and hasattr(
                    network_response.content, "iter_chunked"
                ):
                    downloaded_bytes = 0
                    async for (
                        data_chunk
                    ) in network_response.content.iter_chunked(
                        self.chunk_size_bytes
                    ):
                        memory_buffer.write(data_chunk)
                        downloaded_bytes += len(data_chunk)
                        _transfer_progress(
                            source_uri,
                            target_node,
                            detail="Downloading archive bytes.",
                            phase="artifact",
                            bytes_downloaded=downloaded_bytes,
                            bytes_total=total_bytes,
                        )
                else:
                    payload = await network_response.read()
                    memory_buffer.write(payload)
                    _transfer_progress(
                        source_uri,
                        target_node,
                        detail="Downloading archive bytes.",
                        phase="artifact",
                        bytes_downloaded=len(payload),
                        bytes_total=total_bytes,
                    )
                memory_buffer.seek(0)
                with zipfile.ZipFile(memory_buffer) as archive_context:
                    self._extract_archive(archive_context, target_node)
        return True

    def _execute_transfer_sync(
        self, source_uri: str, target_node: Path
    ) -> None:
        target_node.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            source_uri, timeout=self.request_timeout_seconds
        ) as response:
            payload = response.read()
        _transfer_progress(
            source_uri,
            target_node,
            detail="Downloading archive bytes.",
            phase="artifact",
            bytes_downloaded=len(payload),
            bytes_total=_content_length(
                getattr(
                    getattr(response, "headers", {}),
                    "get",
                    lambda _key, _default=None: None,
                )("Content-Length")
            ),
        )
        memory_buffer = io.BytesIO(payload)
        with zipfile.ZipFile(memory_buffer) as archive_context:
            self._extract_archive(archive_context, target_node)
