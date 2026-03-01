import unittest
from definers import strip_nikud


class TestStripNikud(unittest.TestCase):
    def test_ascii_unchanged(self):
        self.assertEqual(strip_nikud("hello"), "hello")

    def test_empty_string(self):
        self.assertEqual(strip_nikud(""), "")

    def test_hebrew_without_nikud(self):
        self.assertEqual(strip_nikud("שלום"), "שלום")

    def test_returns_string(self):
        result = strip_nikud("test")
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
