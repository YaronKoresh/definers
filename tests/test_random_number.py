import unittest
from unittest.mock import patch

from definers import random_number


class TestRandomNumber(unittest.TestCase):
    def test_returns_int(self):
        result = random_number()
        self.assertIsInstance(result, int)

    def test_default_range(self):
        for _ in range(100):
            result = random_number()
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 100)

    def test_custom_range(self):
        for _ in range(100):
            result = random_number(min=10, max=20)
            self.assertGreaterEqual(result, 10)
            self.assertLessEqual(result, 20)

    def test_single_value_range(self):
        result = random_number(min=5, max=5)
        self.assertEqual(result, 5)


if __name__ == "__main__":
    unittest.main()
