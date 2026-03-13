import re
import unittest

from definers import regex_utils


class TestRegexUtils(unittest.TestCase):
    def test_escape_basic(self):
        self.assertEqual(regex_utils.escape("a.b*c"), re.escape("a.b*c"))

    def test_compile_simple(self):
        r = regex_utils.compile(r"\d+")
        self.assertTrue(r.fullmatch("123"))

    def test_compile_nested_quantifier_reject(self):
        with self.assertRaises(ValueError):
            regex_utils.compile(r"(a+)+")

    def test_compile_long_pattern_reject(self):
        long_pat = "a" * (regex_utils.MAX_PATTERN_LENGTH + 1)
        with self.assertRaises(ValueError):
            regex_utils.compile(long_pat)

    def test_sub_wrapper(self):
        out = regex_utils.sub(r"x", "y", "xoxo")
        self.assertEqual(out, "yoyo")

    def test_escape_and_compile(self):
        pat = regex_utils.escape_and_compile(r"^foo{}bar$", "baz")
        self.assertTrue(pat.fullmatch("foobazbar"))

    def test_escape_and_compile_complexity(self):
        with self.assertRaises(ValueError):
            regex_utils.escape_and_compile(r"(foo{})+", "+")

    def test_fullmatch(self):
        self.assertTrue(regex_utils.fullmatch(r"abc", "abc"))
        self.assertFalse(regex_utils.fullmatch(r"abc", "abcd"))

    def test_sub_with_flags(self):
        self.assertEqual(
            regex_utils.sub(r"a", "b", "A", flags=re.IGNORECASE), "b"
        )

    def test_compile_performance_guard(self):

        huge = "x" * (regex_utils.MAX_PATTERN_LENGTH * 10)
        with self.assertRaises(ValueError):
            regex_utils.compile(huge)

    def test_escape_handles_long_input(self):
        long_input = "[" * 10000 + "]" * 10000
        escaped = regex_utils.escape(long_input)
        self.assertTrue(len(escaped) >= len(long_input))


if __name__ == "__main__":
    unittest.main()
