import unittest
from unittest.mock import patch

import definers._text as text_module


class TestTranslateWithCode(unittest.TestCase):
    def test_empty_string(self):

        result = text_module.translate_with_code("", "en")
        self.assertEqual(result, "")

    def test_whitespace_only(self):
        result = text_module.translate_with_code("   ", "en")
        self.assertEqual(result, "   ")

    @patch.object(text_module, "ai_translate", return_value="translated text")
    def test_plain_text_is_translated(self, mock_translate):
        result = text_module.translate_with_code("hello world", "fr")
        mock_translate.assert_called_once()
        self.assertEqual(result, "translated text")

    @patch.object(text_module, "ai_translate", return_value="translated")
    def test_code_block_is_not_translated(self, mock_translate):
        text = "before\n```python\nprint('hi')\n```\nafter"
        result = text_module.translate_with_code(text, "es")
        self.assertIn("```python\nprint('hi')\n```", result)
        for call in mock_translate.call_args_list:
            self.assertNotIn("print", call[0][0])

    @patch.object(
        text_module, "ai_translate", side_effect=lambda t, lang: t.upper()
    )
    def test_text_outside_code_is_translated(self, mock_translate):
        text = "intro\n```\ncode block\n```\noutro"
        result = text_module.translate_with_code(text, "de")
        self.assertIn("INTRO", result)
        self.assertIn("OUTRO", result)
        self.assertIn("```\ncode block\n```", result)

    @patch.object(
        text_module, "ai_translate", side_effect=lambda t, lang: t.upper()
    )
    def test_multiple_code_blocks_preserved(self, mock_translate):
        text = "text1\n```a\n```\ntext2\n```b\n```"
        result = text_module.translate_with_code(text, "ja")
        self.assertIn("```a\n```", result)
        self.assertIn("```b\n```", result)


if __name__ == "__main__":
    unittest.main()
