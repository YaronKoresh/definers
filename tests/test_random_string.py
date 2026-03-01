import unittest

from definers import random_string


class TestRandomString(unittest.TestCase):
    def test_default_length_range(self):
        result = random_string()
        self.assertGreaterEqual(len(result), 50)
        self.assertLessEqual(len(result), 60)

    def test_custom_length_range(self):
        result = random_string(min_len=10, max_len=20)
        self.assertGreaterEqual(len(result), 10)
        self.assertLessEqual(len(result), 20)

    def test_returns_string(self):
        result = random_string()
        self.assertIsInstance(result, str)

    def test_only_valid_chars(self):
        import string

        valid = set(string.ascii_letters + string.digits + string.punctuation)
        result = random_string()
        for ch in result:
            self.assertIn(ch, valid)


if __name__ == "__main__":
    unittest.main()
