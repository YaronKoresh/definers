import multiprocessing
import unittest

from definers.system import cores


class TestCores(unittest.TestCase):
    def test_returns_int(self):
        result = cores()
        self.assertIsInstance(result, int)

    def test_at_least_one(self):
        result = cores()
        self.assertGreaterEqual(result, 1)

    def test_matches_cpu_count(self):
        result = cores()
        self.assertEqual(result, multiprocessing.cpu_count())


if __name__ == "__main__":
    unittest.main()
