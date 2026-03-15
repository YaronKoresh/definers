import unittest
from unittest.mock import patch

import gradio as gr

import definers.presentation.chat_handlers as presentation_chat_handlers
import definers.text as text
from definers.constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH
from definers.presentation.chat_handlers import get_chat_response


class TestChatValidation(unittest.TestCase):
    def setUp(self):
        patchers = [
            patch.object(text, "language", return_value="en"),
            patch.object(
                text,
                "ai_translate",
                side_effect=lambda txt, lang="en": txt,
            ),
            patch.object(text, "simple_text", side_effect=lambda txt: txt),
            patch.object(
                presentation_chat_handlers,
                "answer",
                side_effect=lambda history: history[-1]["content"],
            ),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_get_chat_response_rejects_too_long(self):
        msg = {"text": "a" * (MAX_INPUT_LENGTH + 1), "files": []}
        with self.assertRaises(gr.Error):
            get_chat_response(msg, [])

    def test_get_chat_response_rejects_excess_spaces(self):
        msg = {
            "text": "word" + " " * (MAX_CONSECUTIVE_SPACES + 2) + "word",
            "files": [],
        }
        with self.assertRaises(gr.Error):
            get_chat_response(msg, [])

    def test_get_chat_response_returns_validated_text_for_short_prompt(self):
        msg = {"text": "ok", "files": []}

        self.assertEqual(get_chat_response(msg, []), "ok")


if __name__ == "__main__":
    unittest.main()
