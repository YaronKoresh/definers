import unittest
from definers import lang_code_to_name

class TestLangCodeToName(unittest.TestCase):

    def test_valid_common_code(self):
        self.assertEqual(lang_code_to_name('en'), 'english')

    def test_valid_another_common_code(self):
        self.assertEqual(lang_code_to_name('es'), 'spanish')

    def test_valid_less_common_code(self):
        self.assertEqual(lang_code_to_name('zu'), 'zulu')

    def test_valid_code_with_hyphen(self):
        self.assertEqual(lang_code_to_name('zh-CN'), 'chinese (simplified)')

    def test_invalid_code(self):
        with self.assertRaises(KeyError):
            lang_code_to_name('xx')

    def test_empty_string_code(self):
        with self.assertRaises(KeyError):
            lang_code_to_name('')

if __name__ == '__main__':
    unittest.main()
