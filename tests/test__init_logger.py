import logging
import unittest

from definers import _init_logger


class TestInitLogger(unittest.TestCase):

    def test_returns_logger_instance(self):
        logger = _init_logger()
        self.assertIsInstance(logger, logging.Logger)

    def test_logger_level_is_debug(self):
        logger = _init_logger()
        self.assertEqual(logger.level, logging.DEBUG)

    def test_logger_has_handlers(self):
        logger = _init_logger()
        self.assertTrue(len(logger.handlers) > 0)

    def test_handler_is_streamhandler(self):
        logger = _init_logger()
        self.assertIsInstance(
            logger.handlers[0], logging.StreamHandler
        )

    def test_handler_has_formatter(self):
        logger = _init_logger()
        self.assertIsInstance(
            logger.handlers[0].formatter, logging.Formatter
        )

    def test_no_duplicate_handlers_on_multiple_calls(self):
        logger = _init_logger()
        initial_handler_count = len(logger.handlers)
        _init_logger()
        self.assertEqual(len(logger.handlers), initial_handler_count)

    def test_formatter_format_string(self):
        logger = _init_logger()
        expected_format = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.assertEqual(
            logger.handlers[0].formatter._fmt, expected_format
        )


if __name__ == "__main__":
    unittest.main()
