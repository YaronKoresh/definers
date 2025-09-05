import unittest
from unittest.mock import Mock
import threading
import time
from queue import Queue
from definers import wait, thread

class TestWait(unittest.TestCase):

    def test_wait_joins_single_thread(self):
        mock_thread = Mock(spec=threading.Thread)
        wait(mock_thread)
        mock_thread.join.assert_called_once()

    def test_wait_joins_multiple_threads(self):
        mock_thread1 = Mock(spec=threading.Thread)
        mock_thread2 = Mock(spec=threading.Thread)
        mock_thread3 = Mock(spec=threading.Thread)

        wait(mock_thread1, mock_thread2, mock_thread3)

        mock_thread1.join.assert_called_once()
        mock_thread2.join.assert_called_once()
        mock_thread3.join.assert_called_once()

    def test_wait_with_real_threads(self):
        q1 = Queue()
        q2 = Queue()

        def func1():
            time.sleep(0.01)
            q1.put("done")

        def func2():
            time.sleep(0.02)
            q2.put("done")

        t1 = thread(func1)
        t2 = thread(func2)

        self.assertTrue(t1.is_alive())
        self.assertTrue(t2.is_alive())

        wait(t1, t2)

        self.assertFalse(t1.is_alive())
        self.assertFalse(t2.is_alive())
        self.assertEqual(q1.get_nowait(), "done")
        self.assertEqual(q2.get_nowait(), "done")
        
    def test_wait_no_threads(self):
        try:
            wait()
        except Exception as e:
            self.fail(f"wait() raised an exception with no arguments: {e}")

if __name__ == '__main__':
    unittest.main()
