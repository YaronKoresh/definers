import unittest

from definers import camel_case


class TestCamelCase(unittest.TestCase):
    def test_simple_words(self):
        result = camel_case("hello world")
        self.assertEqual(result, "HelloWorld")

    def test_single_word(self):
        result = camel_case("hello")
        self.assertEqual(result, "Hello")

    def test_empty_string(self):
        result = camel_case("")
        self.assertEqual(result, "")

    def test_already_camel(self):
        result = camel_case("hello World")
        self.assertEqual(result, "HelloWorld")


if __name__ == "__main__":
    unittest.main()
