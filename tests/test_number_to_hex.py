import unittest
from definers import number_to_hex


class TestNumberToHex(unittest.TestCase):
    def test_zero(self):
        result = number_to_hex(0)
        self.assertIsNotNone(result)

    def test_positive(self):
        result = number_to_hex(255)
        self.assertIsNotNone(result)

    def test_returns_value(self):
        result = number_to_hex(16)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
