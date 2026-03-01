import asyncio
import functools
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Protocol, TypeAlias, TypeVar

T = TypeVar("T")
ExceptionType: TypeAlias = type[BaseException] | tuple[type[BaseException], ...]


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenException(Exception):
    pass


@dataclass(slots=True)
class CircuitSnapshot:
    state: CircuitState
    failure_count: int
    opened_at: float


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60,
        clock: Callable[[], float] | None = None,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be greater than 0")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitState.CLOSED
        self._clock = clock or time.monotonic
        self._state_lock = Lock()

    def snapshot(self) -> CircuitSnapshot:
        with self._state_lock:
            return CircuitSnapshot(
                state=self.state,
                failure_count=self.failure_count,
                opened_at=self.last_failure_time,
            )

    def _transition_open(self) -> None:
        self.state = CircuitState.OPEN
        self.last_failure_time = self._clock()

    def _transition_closed(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    def _allow_call(self) -> None:
        if self.state != CircuitState.OPEN:
            return
        elapsed_seconds = self._clock() - self.last_failure_time
        if elapsed_seconds > self.recovery_timeout:
            self.state = CircuitState.HALF_OPEN
            return
        raise CircuitBreakerOpenException("Circuit breaker is OPEN")

    def _record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self._transition_closed()
            return
        self.failure_count = 0

    def _record_failure(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self._transition_open()
            return
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self._transition_open()

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        with self._state_lock:
            self._allow_call()
        try:
            result = func(*args, **kwargs)
            with self._state_lock:
                self._record_success()
            return result
        except Exception:
            with self._state_lock:
                self._record_failure()
            raise

    async def execute_async(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        with self._state_lock:
            self._allow_call()
        try:
            result = await func(*args, **kwargs)
            with self._state_lock:
                self._record_success()
            return result
        except Exception:
            with self._state_lock:
                self._record_failure()
            raise


class RetryDelayStrategy(Protocol):
    def delay_for_attempt(self, attempt_index: int) -> float: ...


@dataclass(slots=True)
class ExponentialBackoffDelay:
    base_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 60.0
    jitter_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if self.max_delay < self.base_delay:
            raise ValueError(
                "max_delay must be greater than or equal to base_delay"
            )
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")

    def delay_for_attempt(self, attempt_index: int) -> float:
        if attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        raw_delay = self.base_delay * self.multiplier**attempt_index
        bounded_delay = min(raw_delay, self.max_delay)
        if self.jitter_ratio == 0:
            return bounded_delay
        jitter_window = bounded_delay * self.jitter_ratio
        randomized_delay = bounded_delay + random.uniform(
            -jitter_window, jitter_window
        )
        return max(0.0, randomized_delay)


def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    retry_on: ExceptionType = Exception,
    delay_strategy: RetryDelayStrategy | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    active_delay_strategy = delay_strategy or ExponentialBackoffDelay(
        base_delay=delay
    )

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: BaseException | None = None
            for attempt_index in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retry_on as current_exception:
                    last_exception = current_exception
                    if attempt_index == max_retries - 1:
                        break
                    logging.getLogger(__name__).warning(
                        "Retry attempt %d/%d failed: %s",
                        attempt_index + 1,
                        max_retries,
                        current_exception,
                    )
                    await asyncio.sleep(
                        active_delay_strategy.delay_for_attempt(attempt_index)
                    )
            if last_exception is None:
                raise RuntimeError(
                    "retry loop exited without result or captured exception"
                )
            raise last_exception

        return wrapper

    return decorator
