import asyncio

import pytest

from definers.shared_kernel.resilience import (
    ExponentialBackoffDelay,
    RetryPolicy,
    execute_with_retry_async,
)


def test_retry_policy_rejects_non_positive_retry_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RetryPolicy(max_retries=0)


def test_exponential_backoff_delay_grows_and_caps_without_jitter() -> None:
    strategy = ExponentialBackoffDelay(
        base_delay=0.5,
        multiplier=2,
        max_delay=3,
        jitter_ratio=0,
    )

    assert strategy.delay_for_attempt(0) == 0.5
    assert strategy.delay_for_attempt(1) == 1.0
    assert strategy.delay_for_attempt(2) == 2.0
    assert strategy.delay_for_attempt(3) == 3.0
    assert strategy.delay_for_attempt(4) == 3.0


def test_exponential_backoff_delay_applies_jitter_with_expected_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_windows: list[tuple[float, float]] = []

    def fake_uniform(lower: float, upper: float) -> float:
        recorded_windows.append((lower, upper))
        return 0.25

    monkeypatch.setattr("random.uniform", fake_uniform)
    strategy = ExponentialBackoffDelay(
        base_delay=2,
        multiplier=2,
        max_delay=10,
        jitter_ratio=0.1,
    )

    assert strategy.delay_for_attempt(2) == 8.25
    assert recorded_windows == [(-0.8, 0.8)]


def test_execute_with_retry_async_retries_with_expected_delays_and_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_sleeps: list[float] = []
    recorded_callbacks: list[tuple[int, int, str]] = []
    attempts = 0

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)

    async def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError(f"boom {attempts}")
        return "ok"

    def on_retry(attempt: int, total: int, error: BaseException) -> None:
        recorded_callbacks.append((attempt, total, str(error)))

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    result = asyncio.run(
        execute_with_retry_async(
            flaky_operation,
            retry_policy=RetryPolicy(
                max_retries=3,
                delay_strategy=ExponentialBackoffDelay(
                    base_delay=0.5,
                    multiplier=2,
                    max_delay=10,
                ),
                retry_on=ValueError,
            ),
            on_retry=on_retry,
        )
    )

    assert result == "ok"
    assert attempts == 3
    assert recorded_sleeps == [0.5, 1.0]
    assert recorded_callbacks == [
        (1, 3, "boom 1"),
        (2, 3, "boom 2"),
    ]


def test_execute_with_retry_async_reraises_final_retryable_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)

    async def always_fail() -> str:
        raise RuntimeError("down")

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="down"):
        asyncio.run(
            execute_with_retry_async(
                always_fail,
                retry_policy=RetryPolicy(
                    max_retries=2,
                    delay_strategy=ExponentialBackoffDelay(base_delay=0),
                    retry_on=RuntimeError,
                ),
            )
        )

    assert recorded_sleeps == [0.0]


def test_execute_with_retry_async_does_not_retry_non_retryable_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)

    async def fatal_operation() -> str:
        raise TypeError("fatal")

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(TypeError, match="fatal"):
        asyncio.run(
            execute_with_retry_async(
                fatal_operation,
                retry_policy=RetryPolicy(
                    max_retries=3,
                    delay_strategy=ExponentialBackoffDelay(base_delay=0),
                    retry_on=ValueError,
                ),
            )
        )

    assert recorded_sleeps == []
