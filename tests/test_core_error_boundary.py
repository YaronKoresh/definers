import asyncio
import logging

from src.definers._core import CriticalSystemFailure, enforce_error_boundary


@enforce_error_boundary
def stable_function(value: int) -> int:
    return value * 2


@enforce_error_boundary
def unstable_function() -> None:
    raise ValueError("sync-fault")


@enforce_error_boundary
async def stable_async_function(value: int) -> int:
    await asyncio.sleep(0)
    return value + 1


@enforce_error_boundary
async def unstable_async_function() -> None:
    await asyncio.sleep(0)
    raise RuntimeError("async-fault")


def test_enforce_error_boundary_sync_success() -> None:
    assert stable_function(4) == 8


def test_enforce_error_boundary_sync_failure() -> None:
    try:
        unstable_function()
        raise AssertionError("Expected CriticalSystemFailure")
    except CriticalSystemFailure:
        pass


def test_enforce_error_boundary_async_success() -> None:
    assert asyncio.run(stable_async_function(4)) == 5


def test_enforce_error_boundary_async_failure() -> None:
    try:
        asyncio.run(unstable_async_function())
        raise AssertionError("Expected CriticalSystemFailure")
    except CriticalSystemFailure:
        pass


def test_enforce_error_boundary_preserves_function_name() -> None:
    assert stable_function.__name__ == "stable_function"
    assert stable_async_function.__name__ == "stable_async_function"


def test_logger_pipeline_remains_accessible() -> None:
    logger = logging.getLogger("src.definers._core")
    assert isinstance(logger, logging.Logger)
