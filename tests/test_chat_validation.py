import unittest

import gradio as gr

import definers._text as _text
from definers._chat import get_chat_response
from definers._constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH

_text.language = lambda txt: "en"
_text.ai_translate = lambda txt: txt
_text.simple_text = lambda txt: txt


class TestChatValidation(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
