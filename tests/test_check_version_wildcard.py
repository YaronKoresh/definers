import unittest
from definers import check_version_wildcard

class TestCheckVersionWildcard(unittest.TestCase):

    def test_exact_match(self):
        self.assertTrue(check_version_wildcard("1.2.3", "1.2.3"))

    def test_exact_no_match(self):
        self.assertFalse(check_version_wildcard("1.2.3", "1.2.4"))

    def test_wildcard_patch(self):
        self.assertTrue(check_version_wildcard("1.2.*", "1.2.3"))
        self.assertTrue(check_version_wildcard("1.2.*", "1.2.10"))
        self.assertFalse(check_version_wildcard("1.2.*", "1.3.0"))

    def test_wildcard_minor(self):
        self.assertTrue(check_version_wildcard("1.*.3", "1.2.3"))
        self.assertTrue(check_version_wildcard("1.*.3", "1.10.3"))
        self.assertFalse(check_version_wildcard("1.*.3", "1.2.4"))

    def test_wildcard_major(self):
        self.assertTrue(check_version_wildcard("*.2.3", "1.2.3"))
        self.assertTrue(check_version_wildcard("*.2.3", "10.2.3"))
        self.assertFalse(check_version_wildcard("*.2.3", "1.3.3"))

    def test_multiple_wildcards(self):
        self.assertTrue(check_version_wildcard("1.*.*", "1.2.3"))
        self.assertTrue(check_version_wildcard("*.*.3", "1.2.3"))
        self.assertTrue(check_version_wildcard("*.*.*", "10.20.30"))
        self.assertFalse(check_version_wildcard("1.*.*", "2.0.0"))

    def test_wildcard_in_middle_of_segment(self):
        self.assertTrue(check_version_wildcard("1.1*.0", "1.12.0"))
        self.assertTrue(check_version_wildcard("1.1*.0", "1.1.0"))
        self.assertFalse(check_version_wildcard("1.1*.0", "1.2.0"))

    def test_empty_strings(self):
        self.assertTrue(check_version_wildcard("", ""))
        self.assertFalse(check_version_wildcard("1.0.0", ""))
        self.assertFalse(check_version_wildcard("", "1.0.0"))

    def test_with_text_and_special_chars(self):
        self.assertTrue(check_version_wildcard("2.5.0-alpha", "2.5.0-alpha"))
        self.assertTrue(check_version_wildcard("2.5.*-alpha", "2.5.0-alpha"))
        self.assertFalse(check_version_wildcard("2.5.*-beta", "2.5.0-alpha"))

if __name__ == '__main__':
    unittest.main()
