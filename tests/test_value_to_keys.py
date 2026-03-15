import unittest

from definers.audio import value_to_keys


class TestValueToKeys(unittest.TestCase):
    def test_single_match(self):
        d = {"a": 1, "b": 2, "c": 3}
        result = value_to_keys(d, 2)
        self.assertEqual(result, ["b"])

    def test_multiple_matches(self):
        d = {"a": 1, "b": 1, "c": 2}
        result = value_to_keys(d, 1)
        self.assertEqual(sorted(result), ["a", "b"])

    def test_no_match(self):
        d = {"a": 1, "b": 2}
        result = value_to_keys(d, 99)
        self.assertEqual(result, [])

    def test_empty_dict(self):
        result = value_to_keys({}, "x")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
