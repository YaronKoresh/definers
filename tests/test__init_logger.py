import logging
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from definers.logger import init_logger


def _clear_logger(name: str) -> None:
    active_logger = logging.getLogger(name)
    for handler in list(active_logger.handlers):
        active_logger.removeHandler(handler)
        handler.close()


class TestInitLogger(unittest.TestCase):
    def test_returns_logger_instance(self):
        logger = init_logger()
        self.assertIsInstance(logger, logging.Logger)

    def test_logger_level_is_debug(self):
        logger = init_logger()
        self.assertEqual(logger.level, logging.DEBUG)

    def test_logger_has_handlers(self):
        logger = init_logger()
        self.assertTrue(len(logger.handlers) > 0)

    def test_handler_is_streamhandler(self):
        logger = init_logger()
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

    def test_handler_has_formatter(self):
        logger = init_logger()
        self.assertIsInstance(logger.handlers[0].formatter, logging.Formatter)

    def test_no_duplicate_handlers_on_multiple_calls(self):
        logger = init_logger()
        initial_handler_count = len(logger.handlers)
        init_logger()
        self.assertEqual(len(logger.handlers), initial_handler_count)

    def test_formatter_format_string(self):
        logger = init_logger()
        expected_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.assertEqual(logger.handlers[0].formatter._fmt, expected_format)

    def test_repeated_file_logger_initialization_does_not_duplicate_handlers(
        self,
    ):
        logger_name = f"definers.test.logger.{uuid4().hex}"
        file_descriptor, log_path = tempfile.mkstemp(suffix=".log")
        os.close(file_descriptor)

        try:
            logger = init_logger(name=logger_name, log_file=log_path)
            logger = init_logger(name=logger_name, log_file=log_path)

            stream_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
            ]
            file_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
            ]

            self.assertEqual(len(stream_handlers), 1)
            self.assertEqual(len(file_handlers), 1)
        finally:
            _clear_logger(logger_name)
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_concurrent_logger_initialization_is_idempotent(self):
        logger_name = f"definers.test.concurrent.{uuid4().hex}"

        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                loggers = list(
                    executor.map(
                        lambda _: init_logger(name=logger_name), range(32)
                    )
                )

            logger = loggers[0]
            stream_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
            ]

            self.assertTrue(
                all(active_logger is logger for active_logger in loggers)
            )
            self.assertEqual(len(stream_handlers), 1)
        finally:
            _clear_logger(logger_name)


if __name__ == "__main__":
    unittest.main()
