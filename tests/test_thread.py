import unittest
from unittest.mock import patch
import threading
import time
from queue import Queue
from definers import thread

class TestThread(unittest.TestCase):

    def test_thread_starts_and_runs(self):
        event = threading.Event()
        def target_func():
            event.set()

        t = thread(target_func)
        self.assertIsInstance(t, threading.Thread)
        self.assertTrue(t.is_alive())
        
        event_was_set = event.wait(timeout=1)
        self.assertTrue(event_was_set, "Target function did not run within timeout")
        
        t.join()
        self.assertFalse(t.is_alive())

    def test_thread_with_args_and_kwargs(self):
        q = Queue()
        def target_func_with_args(arg1, kwarg1=None):
            q.put((arg1, kwarg1))

        t = thread(target_func_with_args, "test_arg", kwarg1="test_kwarg")
        
        try:
            result = q.get(timeout=1)
            self.assertEqual(result, ("test_arg", "test_kwarg"))
        except Empty:
            self.fail("Target function with args did not execute or put result in queue.")
        
        t.join()

    @patch('definers.catch')
    @patch('threading.Thread', side_effect=Exception("Test Exception"))
    def test_thread_exception_handling(self, mock_thread, mock_catch):
        
        thread(lambda: None)
        mock_catch.assert_called_once()
        self.assertIsInstance(mock_catch.call_args[0][0], Exception)
        self.assertEqual(str(mock_catch.call_args[0][0]), "Test Exception")

if __name__ == '__main__':
    unittest.main()
