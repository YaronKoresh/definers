import asyncio

from definers.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenException,
    CircuitState,
)


class ScriptedClock:
    def __init__(self, values: list[float], fallback: float) -> None:
        self.values = values
        self.fallback = fallback

    def __call__(self) -> float:
        if self.values:
            return self.values.pop(0)
        return self.fallback


def test_async_circuit_resets_failures_after_success() -> None:
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10)

    async def failing_operation() -> str:
        raise ValueError("transient")

    async def healthy_operation() -> str:
        return "ok"

    async def run_test() -> None:
        try:
            await breaker.execute_async(failing_operation)
        except ValueError as error:
            assert "transient" in str(error)
        else:
            raise AssertionError("Expected ValueError")

        assert breaker.snapshot().failure_count == 1
        assert await breaker.execute_async(healthy_operation) == "ok"
        assert breaker.snapshot().state == CircuitState.CLOSED
        assert breaker.snapshot().failure_count == 0

    asyncio.run(run_test())


def test_async_circuit_opens_at_threshold_and_blocks_subsequent_calls() -> None:
    clock = ScriptedClock([1.0, 2.0, 3.0], fallback=3.0)
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=5,
        clock=clock,
    )

    async def failing_operation() -> str:
        raise RuntimeError("down")

    async def run_test() -> None:
        try:
            await breaker.execute_async(failing_operation)
        except RuntimeError as error:
            assert "down" in str(error)
        else:
            raise AssertionError("Expected RuntimeError")

        try:
            await breaker.execute_async(failing_operation)
        except RuntimeError as error:
            assert "down" in str(error)
        else:
            raise AssertionError("Expected RuntimeError")

        try:
            await breaker.execute_async(failing_operation)
        except CircuitBreakerOpenException as error:
            assert "OPEN" in str(error)
        else:
            raise AssertionError("Expected CircuitBreakerOpenException")

        snapshot = breaker.snapshot()
        assert snapshot.state == CircuitState.OPEN
        assert snapshot.failure_count == 2
        assert snapshot.opened_at == 2.0

    asyncio.run(run_test())


def test_async_circuit_transitions_to_half_open_then_closes_on_success() -> (
    None
):
    clock = ScriptedClock([0.0, 0.0, 3.1], fallback=3.1)
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=3,
        clock=clock,
    )

    async def failing_operation() -> str:
        raise RuntimeError("down")

    async def healthy_operation() -> str:
        return "healthy"

    async def run_test() -> None:
        try:
            await breaker.execute_async(failing_operation)
        except RuntimeError as error:
            assert "down" in str(error)
        else:
            raise AssertionError("Expected RuntimeError")

        assert breaker.snapshot().state == CircuitState.OPEN
        assert await breaker.execute_async(healthy_operation) == "healthy"

        snapshot = breaker.snapshot()
        assert snapshot.state == CircuitState.CLOSED
        assert snapshot.failure_count == 0

    asyncio.run(run_test())


def test_async_circuit_half_open_failure_reopens_with_new_timestamp() -> None:
    clock = ScriptedClock([0.0, 0.0, 5.0, 5.0], fallback=5.0)
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=2,
        clock=clock,
    )

    async def failing_operation() -> str:
        raise RuntimeError("still down")

    async def run_test() -> None:
        try:
            await breaker.execute_async(failing_operation)
        except RuntimeError as error:
            assert "still down" in str(error)
        else:
            raise AssertionError("Expected RuntimeError")

        try:
            await breaker.execute_async(failing_operation)
        except RuntimeError as error:
            assert "still down" in str(error)
        else:
            raise AssertionError("Expected RuntimeError")

        snapshot = breaker.snapshot()
        assert snapshot.state == CircuitState.OPEN
        assert snapshot.failure_count == 1
        assert snapshot.opened_at == 5.0

    asyncio.run(run_test())
