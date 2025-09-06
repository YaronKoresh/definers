import unittest
from unittest.mock import patch, MagicMock
from definers import catch, _init_logger

class TestCatch(unittest.TestCase):

    @patch('definers.logger.exception')
    def test_catch_logs_exception(self, mock_logger_exception):
        test_exception = ValueError("This is a test exception")
        catch(test_exception)
        mock_logger_exception.assert_called_once_with(test_exception)

    def test_catch_with_non_exception_object(self):
        with patch('definers.logger.exception') as mock_logger_exception:
            non_exception = "just a string"
            catch(non_exception)
            mock_logger_exception.assert_called_once_with(non_exception)

    def test_catch_with_none(self):
        with patch('definers.logger.exception') as mock_logger_exception:
            catch(None)
            mock_logger_exception.assert_called_once_with(None)

    def test_catch_integration_with_real_logger(self):
        logger = _init_logger()
        with patch.object(logger, 'exception') as mock_method:
            test_exception = RuntimeError("Integration test")
            with patch('definers.logger', logger):
                 catch(test_exception)
            mock_method.assert_called_once_with(test_exception)


if __name__ == '__main__':
    unittest.main()
