import unittest

from definers.ml import check_parameter


class TestCheckParameter(unittest.TestCase):
    def test_none_is_false(self):
        self.assertFalse(check_parameter(None))

    def test_valid_string(self):
        self.assertTrue(check_parameter("hello"))

    def test_valid_number(self):
        self.assertTrue(check_parameter(42))

    def test_empty_list_is_false(self):
        self.assertFalse(check_parameter([]))

    def test_non_empty_list(self):
        self.assertTrue(check_parameter([1, 2, 3]))

    def test_list_with_empty_string(self):
        self.assertFalse(check_parameter(["  "]))


if __name__ == "__main__":
    unittest.main()
