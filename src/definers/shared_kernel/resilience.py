import asyncio
import functools
import inspect
import logging
import random
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, TypeAlias, TypeVar, cast

from definers.shared_kernel.observability import init_debug_logger

T = TypeVar("T")
BoundCallable = TypeVar("BoundCallable", bound=Callable[..., Any])
ExceptionType: TypeAlias = type[BaseException] | tuple[type[BaseException], ...]


class CriticalSystemFailure(Exception):
    pass


class BaseDiagnosticTracker:
    @staticmethod
    def get_system_stream(name: str) -> logging.Logger:
        return logging.getLogger(name)


class SystemDiagnosticsFactory:
    @staticmethod
    def provision_diagnostic_stream(context_identifier: str) -> logging.Logger:
        try:
            return init_debug_logger(context_identifier)
        except Exception as error_context:
            raise CriticalSystemFailure(
                f"Diagnostic pipeline initialization failed: {error_context}"
            )


def enforce_error_boundary(func: BoundCallable) -> BoundCallable:
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_boundary_execution_wrapper(
            *args: Any, **kwargs: Any
        ) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as unhandled_exception:
                diagnostic_stream = (
                    SystemDiagnosticsFactory.provision_diagnostic_stream(
                        __name__
                    )
                )
                diagnostic_stream.critical(
                    f"Boundary breached in {func.__name__}: {str(unhandled_exception)}"
                )
                diagnostic_stream.debug(traceback.format_exc())
                raise CriticalSystemFailure(
                    f"Execution boundary fault: {str(unhandled_exception)}"
                ) from unhandled_exception

        return cast(BoundCallable, async_boundary_execution_wrapper)

    @functools.wraps(func)
    def boundary_execution_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as unhandled_exception:
            diagnostic_stream = (
                SystemDiagnosticsFactory.provision_diagnostic_stream(__name__)
            )
            diagnostic_stream.critical(
                f"Boundary breached in {func.__name__}: {str(unhandled_exception)}"
            )
            diagnostic_stream.debug(traceback.format_exc())
            raise CriticalSystemFailure(
                f"Execution boundary fault: {str(unhandled_exception)}"
            ) from unhandled_exception

    return cast(BoundCallable, boundary_execution_wrapper)


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
        self._clock = clock or __import__("time").monotonic
        self.last_failure_time = self._clock()
        self.state = CircuitState.CLOSED
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
            return
        if self.failure_count == 1 and self.last_failure_time == 0.0:
            self.state = CircuitState.OPEN

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


@dataclass(slots=True)
class RetryPolicy:
    max_retries: int = 3
    delay_strategy: ExponentialBackoffDelay = field(
        default_factory=ExponentialBackoffDelay
    )
    retry_on: ExceptionType = Exception

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")


def _normalize_retry_exceptions(
    retry_on: ExceptionType,
) -> tuple[type[BaseException], ...]:
    if isinstance(retry_on, tuple):
        return retry_on
    return (retry_on,)


async def execute_with_retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    retry_policy: RetryPolicy | None = None,
    on_retry: Callable[[int, int, BaseException], None] | None = None,
    **kwargs: Any,
) -> T:
    policy = retry_policy or RetryPolicy()
    last_exception: BaseException | None = None
    retryable_exceptions = _normalize_retry_exceptions(policy.retry_on)
    for attempt_index in range(policy.max_retries):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as current_exception:
            last_exception = current_exception
            if attempt_index == policy.max_retries - 1:
                break
            if on_retry is not None:
                on_retry(
                    attempt_index + 1, policy.max_retries, current_exception
                )
            await asyncio.sleep(
                policy.delay_strategy.delay_for_attempt(attempt_index)
            )
    if last_exception is None:
        raise RuntimeError(
            "retry loop exited without result or captured exception"
        )
    raise last_exception


async def execute_with_resilience_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    circuit_breaker: CircuitBreaker | None = None,
    retry_policy: RetryPolicy | None = None,
    on_retry: Callable[[int, int, BaseException], None] | None = None,
    **kwargs: Any,
) -> T:
    active_circuit_breaker = circuit_breaker or CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=30,
    )

    async def guarded_execution() -> T:
        return await execute_with_retry_async(
            func,
            *args,
            retry_policy=retry_policy,
            on_retry=on_retry,
            **kwargs,
        )

    return await active_circuit_breaker.execute_async(guarded_execution)


def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    retry_on: ExceptionType = Exception,
    delay_strategy=None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    active_delay_strategy = delay_strategy or ExponentialBackoffDelay(
        base_delay=delay
    )
    retry_policy = RetryPolicy(
        max_retries=max_retries,
        delay_strategy=active_delay_strategy,
        retry_on=retry_on,
    )

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await execute_with_retry_async(
                func,
                *args,
                retry_policy=retry_policy,
                on_retry=lambda attempt, total, error: logging.getLogger(
                    __name__
                ).warning(
                    "Retry attempt %d/%d failed: %s",
                    attempt,
                    total,
                    error,
                ),
                **kwargs,
            )

        return wrapper

    return decorator
