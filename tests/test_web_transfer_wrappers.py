import asyncio
import sys
from types import ModuleType

from definers.media import web_transfer


class FakeOrchestrator:
    def __init__(self, result: bool):
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def process(self, source_uri: str, target_node: str) -> bool:
        self.calls.append((source_uri, target_node))
        return self.result


def test_download_file_returns_destination_on_success() -> None:
    orchestrator = FakeOrchestrator(True)

    result = web_transfer.download_file(
        "https://example.com/file.bin",
        "artifact.bin",
        executor=web_transfer.execute_async_operation,
        orchestrator_factory=lambda: orchestrator,
    )

    assert result == "artifact.bin"
    assert orchestrator.calls == [
        ("https://example.com/file.bin", "artifact.bin")
    ]


def test_download_file_returns_none_on_failed_transfer() -> None:
    orchestrator = FakeOrchestrator(False)

    result = web_transfer.download_file(
        "https://example.com/file.bin",
        "artifact.bin",
        executor=web_transfer.execute_async_operation,
        orchestrator_factory=lambda: orchestrator,
    )

    assert result is None


def test_download_and_unzip_returns_false_on_failed_transfer() -> None:
    orchestrator = FakeOrchestrator(False)

    result = web_transfer.download_and_unzip(
        "https://example.com/file.zip",
        "extract-here",
        executor=web_transfer.execute_async_operation,
        orchestrator_factory=lambda: orchestrator,
    )

    assert result is False
    assert orchestrator.calls == [
        ("https://example.com/file.zip", "extract-here")
    ]


def test_google_drive_download_suppresses_downloader_failures(
    monkeypatch,
) -> None:
    calls: list[tuple[str, BaseException]] = []

    def fail_download(**kwargs) -> None:
        raise RuntimeError("network fault")

    fake_module = ModuleType("googledrivedownloader")
    fake_module.download_file_from_google_drive = fail_download
    monkeypatch.setitem(sys.modules, "googledrivedownloader", fake_module)
    monkeypatch.setattr(
        web_transfer,
        "log",
        lambda message, error: calls.append((message, error)),
    )

    result = web_transfer.google_drive_download("file-id", "archive.zip")

    assert result is None
    assert len(calls) == 1
    assert calls[0][0] == "google_drive_download failed"
    assert str(calls[0][1]) == "network fault"
