import asyncio

import pytest

from definers.capabilities import (
    CircuitBreaker,
    CircuitBreakerOpenException,
    CircuitState,
    ExponentialBackoffDelay,
    with_retry,
)


def test_circuit_breaker_closed_success() -> None:
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

    def stable_operation() -> str:
        return "success"

    assert breaker.execute(stable_operation) == "success"
    snapshot = breaker.snapshot()
    assert snapshot.state == CircuitState.CLOSED
    assert snapshot.failure_count == 0


def test_circuit_breaker_open_failure() -> None:
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

    def unstable_operation() -> str:
        raise ValueError("fail")

    with pytest.raises(ValueError):
        breaker.execute(unstable_operation)
    with pytest.raises(ValueError):
        breaker.execute(unstable_operation)
    with pytest.raises(CircuitBreakerOpenException):
        breaker.execute(unstable_operation)
    assert breaker.snapshot().state == CircuitState.OPEN


def test_circuit_breaker_half_open_then_close() -> None:
    clock_samples = [0.0, 0.0, 5.0]

    def monotonic_clock() -> float:
        if not clock_samples:
            return 5.0
        return clock_samples.pop(0)

    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=2, clock=monotonic_clock
    )

    def failing_operation() -> str:
        raise RuntimeError("down")

    def healthy_operation() -> str:
        return "ok"

    with pytest.raises(RuntimeError):
        breaker.execute(failing_operation)
    assert breaker.snapshot().state == CircuitState.OPEN
    assert breaker.execute(healthy_operation) == "ok"
    assert breaker.snapshot().state == CircuitState.CLOSED


def test_circuit_breaker_half_open_failure_reopens_immediately() -> None:
    clock_samples = [0.0, 10.0, 10.0]

    def monotonic_clock() -> float:
        if not clock_samples:
            return 10.0
        return clock_samples.pop(0)

    breaker = CircuitBreaker(
        failure_threshold=5, recovery_timeout=2, clock=monotonic_clock
    )

    def failing_operation() -> str:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        breaker.execute(failing_operation)
    assert breaker.snapshot().state == CircuitState.OPEN
    with pytest.raises(RuntimeError):
        breaker.execute(failing_operation)
    assert breaker.snapshot().state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_execute_async_opens_circuit_on_threshold() -> None:
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10)

    async def failing_operation() -> str:
        raise RuntimeError("network")

    with pytest.raises(RuntimeError):
        await breaker.execute_async(failing_operation)
    with pytest.raises(RuntimeError):
        await breaker.execute_async(failing_operation)
    with pytest.raises(CircuitBreakerOpenException):
        await breaker.execute_async(failing_operation)


def test_with_retry_eventual_success() -> None:
    attempts = {"count": 0}

    @with_retry(max_retries=3, delay=0)
    async def unstable_task() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("temporary")
        return "done"

    assert asyncio.run(unstable_task()) == "done"
    assert attempts["count"] == 3


def test_with_retry_rethrows_unhandled_exception() -> None:

    @with_retry(max_retries=3, delay=0, retry_on=ValueError)
    async def non_retryable_failure() -> None:
        raise TypeError("fatal")

    with pytest.raises(TypeError):
        asyncio.run(non_retryable_failure())


def test_exponential_backoff_delay_with_zero_jitter() -> None:
    policy = ExponentialBackoffDelay(
        base_delay=0.5, multiplier=2, max_delay=3, jitter_ratio=0
    )
    assert policy.delay_for_attempt(0) == 0.5
    assert policy.delay_for_attempt(1) == 1.0
    assert policy.delay_for_attempt(2) == 2.0
    assert policy.delay_for_attempt(3) == 3.0


def test_with_retry_validation_errors() -> None:
    with pytest.raises(ValueError):
        with_retry(max_retries=0)
    with pytest.raises(ValueError):
        ExponentialBackoffDelay(base_delay=-1)
