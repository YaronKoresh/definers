import unittest
from definers import string_to_sha3_512


class TestStringToSha3512(unittest.TestCase):
    def test_returns_string(self):
        result = string_to_sha3_512("hello")
        self.assertIsInstance(result, str)

    def test_deterministic(self):
        r1 = string_to_sha3_512("test")
        r2 = string_to_sha3_512("test")
        self.assertEqual(r1, r2)

    def test_different_inputs(self):
        r1 = string_to_sha3_512("hello")
        r2 = string_to_sha3_512("world")
        self.assertNotEqual(r1, r2)

    def test_with_salt(self):
        r1 = string_to_sha3_512("test", salt_num=123)
        r2 = string_to_sha3_512("test", salt_num=456)
        self.assertNotEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
