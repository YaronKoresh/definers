import functools
import inspect
import logging
import traceback
from collections.abc import Callable
from typing import Any, TypeVar, cast

T = TypeVar("T", bound=Callable[..., Any])


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
            diagnostic_stream = BaseDiagnosticTracker.get_system_stream(
                context_identifier
            )
            diagnostic_stream.setLevel(logging.DEBUG)
            terminal_output_bridge = logging.StreamHandler()
            structured_message_schema = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            terminal_output_bridge.setFormatter(structured_message_schema)
            if not diagnostic_stream.handlers:
                diagnostic_stream.addHandler(terminal_output_bridge)
            return diagnostic_stream
        except Exception as error_context:
            raise CriticalSystemFailure(
                f"Diagnostic pipeline initialization failed: {error_context}"
            )


def enforce_error_boundary(func: T) -> T:
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_boundary_execution_wrapper(
            *args: Any,
            **kwargs: Any,
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

        return cast(T, async_boundary_execution_wrapper)

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

    return cast(T, boundary_execution_wrapper)
