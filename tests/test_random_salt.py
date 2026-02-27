import unittest

from definers import random_salt


class TestRandomSalt(unittest.TestCase):
    def test_returns_int(self):
        result = random_salt(4)
        self.assertIsInstance(result, int)

    def test_positive_result(self):
        result = random_salt(8)
        self.assertGreaterEqual(result, 0)

    def test_different_sizes(self):
        r1 = random_salt(1)
        self.assertIsInstance(r1, int)
        r2 = random_salt(16)
        self.assertIsInstance(r2, int)


if __name__ == "__main__":
    unittest.main()
