import asyncio
import logging

import pytest

import definers.shared_kernel.resilience as resilience_module
from definers.shared_kernel.resilience import with_retry


class FakeLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int, int, str]] = []

    def warning(
        self,
        message: str,
        attempt: int,
        total: int,
        error: BaseException,
    ) -> None:
        self.messages.append((message, attempt, total, str(error)))


class RecordingDelayStrategy:
    def __init__(self, delays: list[float]) -> None:
        self.delays = delays
        self.calls: list[int] = []

    def delay_for_attempt(self, attempt_index: int) -> float:
        self.calls.append(attempt_index)
        return self.delays[attempt_index]


def test_with_retry_eventual_success_logs_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_sleeps: list[float] = []
    attempts = 0
    fake_logger = FakeLogger()
    original_get_logger = logging.getLogger

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)

    def fake_get_logger(_name: str | None = None):
        if _name is None:
            return original_get_logger()
        return fake_logger

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(resilience_module.logging, "getLogger", fake_get_logger)

    @with_retry(max_retries=3, delay=0, retry_on=ValueError)
    async def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError(f"temporary {attempts}")
        return "done"

    assert asyncio.run(flaky_operation()) == "done"
    assert attempts == 3
    assert recorded_sleeps == [0.0, 0.0]
    assert fake_logger.messages == [
        ("Retry attempt %d/%d failed: %s", 1, 3, "temporary 1"),
        ("Retry attempt %d/%d failed: %s", 2, 3, "temporary 2"),
    ]


def test_with_retry_reraises_non_retryable_exception_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    @with_retry(max_retries=4, delay=0, retry_on=ValueError)
    async def fatal_operation() -> str:
        raise TypeError("fatal")

    with pytest.raises(TypeError, match="fatal"):
        asyncio.run(fatal_operation())

    assert recorded_sleeps == []


def test_with_retry_uses_custom_delay_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_sleeps: list[float] = []
    attempts = 0
    delay_strategy = RecordingDelayStrategy([0.25, 0.5])

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    @with_retry(
        max_retries=3,
        retry_on=RuntimeError,
        delay_strategy=delay_strategy,
    )
    async def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("retry")
        return "done"

    assert asyncio.run(flaky_operation()) == "done"
    assert delay_strategy.calls == [0, 1]
    assert recorded_sleeps == [0.25, 0.5]


def test_with_retry_preserves_wrapped_metadata() -> None:
    @with_retry(max_retries=1, delay=0)
    async def named_operation() -> str:
        return "ok"

    assert named_operation.__name__ == "named_operation"
    assert named_operation.__wrapped__.__name__ == "named_operation"
    assert asyncio.run(named_operation()) == "ok"