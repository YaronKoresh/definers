import logging

from definers.observability import (
    MESSAGE_SCHEMA,
    init_logger as _init_logger,
)

DEFAULT_LOGGER_NAME = __name__
DIAGNOSTIC_LEVEL = logging.DEBUG


def init_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: str | int | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    effective_level = DIAGNOSTIC_LEVEL if level is None else level
    return _init_logger(
        name,
        level=effective_level,
        log_file=log_file,
        default_level=DIAGNOSTIC_LEVEL,
    )


logger = init_logger()


__all__ = [
    "DEFAULT_LOGGER_NAME",
    "DIAGNOSTIC_LEVEL",
    "MESSAGE_SCHEMA",
    "init_logger",
    "logger",
]
