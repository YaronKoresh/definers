import unittest
from definers import string_to_bytes


class TestStringToBytes(unittest.TestCase):
    def test_basic_string(self):
        result = string_to_bytes("hello")
        self.assertEqual(result, b"hello")

    def test_empty_string(self):
        result = string_to_bytes("")
        self.assertEqual(result, b"")

    def test_returns_bytes(self):
        result = string_to_bytes("test")
        self.assertIsInstance(result, bytes)

    def test_unicode(self):
        result = string_to_bytes("héllo")
        self.assertEqual(result, "héllo".encode())


if __name__ == "__main__":
    unittest.main()
