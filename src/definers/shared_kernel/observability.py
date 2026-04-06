import logging
import os
import sys
from datetime import datetime
from threading import RLock

MESSAGE_SCHEMA = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOGGER_LOCK = RLock()


def _parse_level(value: str | int | None, default_level: int) -> int:
    if value is None:
        return default_level
    if isinstance(value, int):
        return value
    normalized_value = str(value).strip().upper()
    if normalized_value.isdigit():
        return int(normalized_value)
    return getattr(logging, normalized_value, default_level)


def init_logger(
    name: str,
    level: str | int | None = None,
    log_file: str | None = None,
    *,
    enable_console: bool = True,
    propagate: bool = False,
    default_level: int = logging.INFO,
) -> logging.Logger:
    environment_level = os.environ.get("DEFINERS_LOG_LEVEL") or os.environ.get(
        "LOGLEVEL"
    )
    final_level = _parse_level(level or environment_level, default_level)
    resolved_log_file = log_file or os.environ.get("DEFINERS_LOG_FILE")
    formatter = logging.Formatter(MESSAGE_SCHEMA)

    with _LOGGER_LOCK:
        active_logger = logging.getLogger(name)
        active_logger.propagate = propagate
        active_logger.setLevel(final_level)

        console_handler: logging.Handler | None = None
        file_handlers: dict[str, logging.FileHandler] = {}
        for handler in list(active_logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler_key = os.path.normcase(
                    os.path.abspath(os.path.expanduser(handler.baseFilename))
                )
                if handler_key in file_handlers:
                    active_logger.removeHandler(handler)
                    handler.close()
                    continue
                file_handlers[handler_key] = handler
            elif isinstance(handler, logging.StreamHandler):
                stream = getattr(handler, "stream", None)
                if console_handler is None and stream in {
                    None,
                    sys.stderr,
                    sys.stdout,
                }:
                    console_handler = handler
                elif stream in {None, sys.stderr, sys.stdout}:
                    active_logger.removeHandler(handler)
                    handler.close()
                    continue
            handler.setFormatter(formatter)
            handler.setLevel(final_level)

        if enable_console and console_handler is None:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(final_level)
            active_logger.addHandler(console_handler)

        if resolved_log_file:
            normalized_log_file = os.path.normcase(
                os.path.abspath(os.path.expanduser(resolved_log_file))
            )
            if normalized_log_file not in file_handlers:
                file_handler = logging.FileHandler(
                    normalized_log_file,
                    encoding="utf-8",
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(final_level)
                active_logger.addHandler(file_handler)

        for handler in active_logger.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(final_level)

        return active_logger


def init_debug_logger(name: str) -> logging.Logger:
    return init_logger(name, level=logging.DEBUG, default_level=logging.DEBUG)


def log_message(
    active_logger: logging.Logger,
    subject: str,
    data: str | int | None = None,
    status: bool | str | None = None,
) -> None:
    if data is None:
        data = "No data provided"

    now = datetime.now().time()
    payload = str(data)

    print("   \n" + "-" * 30 + "\n")

    if status is True:
        active_logger.info(f"[{now}] SUCCESS - {subject}\n\n{payload}")
        return
    if status is False:
        active_logger.error(f"[{now}] ERROR - {subject}\n\n{payload}")
        return
    if isinstance(status, str) and status.strip():
        active_logger.info(f"[{now}] {status.strip()} - {subject}\n\n{payload}")
        return
    active_logger.info(f"[{now}] {subject}\n\n{payload}")


def catch_exception(
    active_logger: logging.Logger,
    error: object,
    message: str | None = None,
    reraise: bool = False,
) -> None:

    print("   \n" + "-" * 30 + "\n")

    if message:
        active_logger.error(str(message))
    if isinstance(error, BaseException):
        active_logger.exception(error)
    else:
        active_logger.error(str(error))
    if reraise and isinstance(error, BaseException):
        raise error
