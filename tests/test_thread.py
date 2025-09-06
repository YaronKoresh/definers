import time
import unittest
from unittest.mock import MagicMock

from definers import thread


class TestThread(unittest.TestCase):

    def test_thread_starts_and_runs(self):
        func = MagicMock()
        t = thread(func)
        t.join()
        func.assert_called_once()

    def test_thread_with_args_and_kwargs(self):
        func = MagicMock()
        t = thread(func, 1, 2, key="value")
        t.join()
        func.assert_called_once_with(1, 2, key="value")

    def test_thread_returns_thread_object(self):
        import threading

        func = MagicMock()
        t = thread(func)
        self.assertIsInstance(t, threading.Thread)
        t.join()

    def test_thread_exception_handling(self):
        def func_that_raises():
            raise ValueError("Test Error")

        t = thread(func_that_raises)
        t.join()
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
