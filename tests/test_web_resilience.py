import asyncio
import io
import zipfile
from pathlib import Path

from definers.capabilities import CircuitBreaker
from definers.media.web_transfer import (
    HttpChunkedTransferStrategy,
    ResourceRetrievalOrchestrator,
    download_file,
    ZipExtractTransferStrategy,
    execute_async_operation,
)


class FlakyTransferStrategy:
    def __init__(self, successful_attempt: int):
        self.successful_attempt = successful_attempt
        self.calls = 0

    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool:
        self.calls += 1
        if self.calls < self.successful_attempt:
            raise RuntimeError("transient")
        return True


class AlwaysFailTransferStrategy:
    def __init__(self):
        self.calls = 0

    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool:
        self.calls += 1
        raise RuntimeError("permanent")


def test_orchestrator_retries_until_success() -> None:
    strategy = FlakyTransferStrategy(successful_attempt=3)
    orchestrator = ResourceRetrievalOrchestrator(
        strategy=strategy, max_retries=3, base_delay_seconds=0
    )
    result = asyncio.run(
        orchestrator.process("https://example.com", "target.bin")
    )
    assert result is True
    assert strategy.calls == 3


def test_orchestrator_returns_false_after_retry_exhaustion() -> None:
    strategy = AlwaysFailTransferStrategy()
    orchestrator = ResourceRetrievalOrchestrator(
        strategy=strategy, max_retries=2, base_delay_seconds=0
    )
    result = asyncio.run(
        orchestrator.process("https://example.com", "target.bin")
    )
    assert result is False
    assert strategy.calls == 2


def test_orchestrator_open_circuit_blocks_subsequent_call() -> None:
    strategy = AlwaysFailTransferStrategy()
    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=120, clock=lambda: 0
    )
    orchestrator = ResourceRetrievalOrchestrator(
        strategy=strategy,
        circuit_breaker=breaker,
        max_retries=1,
        base_delay_seconds=0,
    )
    first_result = asyncio.run(
        orchestrator.process("https://example.com", "target.bin")
    )
    second_result = asyncio.run(
        orchestrator.process("https://example.com", "target.bin")
    )
    assert first_result is False
    assert second_result is False
    assert strategy.calls == 1


def test_execute_async_operation_inside_running_loop() -> None:

    async def probe() -> int:
        return execute_async_operation(asyncio.sleep(0, result=42))

    assert asyncio.run(probe()) == 42


def test_download_file_returns_destination_on_success(monkeypatch) -> None:

    async def fake_process(
        self, source_uri: str, target_node: str | Path
    ) -> bool:
        return True

    monkeypatch.setattr(ResourceRetrievalOrchestrator, "process", fake_process)
    assert (
        download_file("https://example.com", "artifact.bin") == "artifact.bin"
    )


def test_http_strategy_sync_fallback_writer(tmp_path) -> None:
    payload = b"abcdef"

    class FakeResponse:
        def __init__(self, content: bytes):
            self._stream = io.BytesIO(content)

        def read(self, size: int = -1) -> bytes:
            return self._stream.read(size)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    strategy = HttpChunkedTransferStrategy(chunk_size_bytes=2)
    target_file = tmp_path / "nested" / "target.bin"
    import urllib.request

    original_urlopen = urllib.request.urlopen
    urllib.request.urlopen = lambda *_args, **_kwargs: FakeResponse(payload)
    try:
        strategy._execute_transfer_sync("https://example.com", target_file)
    finally:
        urllib.request.urlopen = original_urlopen
    assert target_file.read_bytes() == payload


def test_zip_strategy_sync_fallback_extract(tmp_path) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, mode="w") as archive_context:
        archive_context.writestr("a.txt", "hello")
    archive_payload = archive_bytes.getvalue()

    class FakeResponse:
        def __init__(self, content: bytes):
            self._content = content

        def read(self, _size: int = -1) -> bytes:
            return self._content

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    strategy = ZipExtractTransferStrategy()
    target_directory = tmp_path / "extract"
    import urllib.request

    original_urlopen = urllib.request.urlopen
    urllib.request.urlopen = lambda *_args, **_kwargs: FakeResponse(
        archive_payload
    )
    try:
        strategy._execute_transfer_sync(
            "https://example.com/archive.zip", target_directory
        )
    finally:
        urllib.request.urlopen = original_urlopen
    assert (target_directory / "a.txt").read_text(encoding="utf-8") == "hello"
