import asyncio

import pytest

from definers.media.web_transfer import execute_async_operation


def test_execute_async_operation_runs_without_active_loop() -> None:
    assert execute_async_operation(asyncio.sleep(0, result=7)) == 7


def test_execute_async_operation_bridges_running_loop() -> None:
    async def probe() -> int:
        return execute_async_operation(asyncio.sleep(0, result=11))

    assert asyncio.run(probe()) == 11


def test_execute_async_operation_reraises_runner_failure() -> None:
    async def fail() -> None:
        raise RuntimeError("bridge failed")

    async def probe() -> None:
        execute_async_operation(fail())

    with pytest.raises(RuntimeError, match="bridge failed"):
        asyncio.run(probe())