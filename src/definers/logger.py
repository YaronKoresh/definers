import logging
from typing import ClassVar, Final

from definers.core import enforce_error_boundary


class UnifiedLoggingSystem:
    DIAGNOSTIC_LEVEL: ClassVar[int] = logging.DEBUG
    MESSAGE_SCHEMA: Final[str] = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    @staticmethod
    @enforce_error_boundary
    def construct_default_diagnostic_pipeline(
        context_scope_identifier: str,
    ) -> logging.Logger:
        diagnostic_stream = logging.getLogger(context_scope_identifier)
        diagnostic_stream.setLevel(UnifiedLoggingSystem.DIAGNOSTIC_LEVEL)
        if not diagnostic_stream.handlers:
            terminal_output_bridge = logging.StreamHandler()
            structured_message_schema = logging.Formatter(
                UnifiedLoggingSystem.MESSAGE_SCHEMA
            )
            terminal_output_bridge.setFormatter(structured_message_schema)
            diagnostic_stream.addHandler(terminal_output_bridge)
        return diagnostic_stream


def init_logger() -> logging.Logger:
    return UnifiedLoggingSystem.construct_default_diagnostic_pipeline(__name__)
