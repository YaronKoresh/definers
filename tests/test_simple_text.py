import unittest

from definers import simple_text


class TestSimpleText(unittest.TestCase):

    def test_with_extra_spaces_and_tabs(self):
        prompt = "this   is a  \t test"
        expected = "this is a test"
        self.assertEqual(simple_text(prompt), expected)

    def test_with_multiple_newlines(self):
        prompt = "line one\n\n\nline two"
        expected = "line one\nline two"
        self.assertEqual(simple_text(prompt), expected)

    def test_with_punctuation(self):
        prompt = "Hello, world! This is a test."
        expected = "hello world this is a test"
        self.assertEqual(simple_text(prompt), expected)

    def test_with_multiple_hyphens(self):
        prompt = "a long---dashed--word"
        expected = "a long-dashed-word"
        self.assertEqual(simple_text(prompt), expected)

    def test_with_mixed_case_and_whitespace(self):
        prompt = "  Some Mixed Case Text  "
        expected = "some mixed case text"
        self.assertEqual(simple_text(prompt), expected)

    def test_with_space_hyphen_space(self):
        prompt = "word - another"
        expected = "word-another"
        self.assertEqual(simple_text(prompt), expected)

    def test_empty_string(self):
        prompt = ""
        expected = ""
        self.assertEqual(simple_text(prompt), expected)

    def test_string_with_only_whitespace(self):
        prompt = "   \t\n   "
        expected = ""
        self.assertEqual(simple_text(prompt), expected)

    def test_complex_string(self):
        prompt = (
            "  Here IS --- a COMPLEX,,, \t text!!\n\nTo test...  "
        )
        expected = "here is-a complex text\nto test"
        self.assertEqual(simple_text(prompt), expected)

    def test_no_changes_needed(self):
        prompt = "this is a clean string"
        expected = "this is a clean string"
        self.assertEqual(simple_text(prompt), expected)


if __name__ == "__main__":
    unittest.main()
