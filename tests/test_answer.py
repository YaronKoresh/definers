import unittest
from unittest.mock import MagicMock, patch

import definers.constants as constants
import definers.ml as ml


class TestAnswer(unittest.TestCase):
    def setUp(self):
        self.mock_sf = MagicMock()
        self.mock_sf.read.return_value = (MagicMock(), MagicMock())
        self.mock_image = MagicMock()
        self.mock_image.open.return_value = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model.generate.return_value = "A witty response."

    @patch("definers.constants.MODELS", new={"answer": None})
    @patch("definers.constants.PROCESSORS", new={})
    @patch("definers.constants.SYSTEM_MESSAGE", new="Mock System Message")
    def test_basic_text_history(self):
        with patch.dict(
            "sys.modules",
            {
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            constants.MODELS["answer"] = self.mock_model
            history = [
                {"role": "user", "content": "Hi there"},
                {"role": "assistant", "content": "Hello! How can I help?"},
                {"role": "user", "content": "Tell me a joke"},
            ]
            response = ml._answer(history, runtime=constants)
            self.assertEqual(response, "A witty response.")
            expected_prompt = "<|system|>Mock System Message<|end|><|user|>Hi there<|end|><|assistant|>Hello! How can I help?<|end|><|user|>Tell me a joke<|end|><|assistant|>"
            self.mock_model.generate.assert_called_once_with(
                prompt=expected_prompt, max_length=200, beam_width=16
            )

    @patch("definers.constants.MODELS", new={"answer": None})
    @patch("definers.constants.PROCESSORS", new={})
    @patch("definers.constants.iio_formats", new=["jpg"])
    def test_history_with_image(self):
        with patch.dict(
            "sys.modules",
            {
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            constants.MODELS["answer"] = self.mock_model
            history = [{"role": "user", "content": {"path": "test.jpg"}}]
            ml._answer(history, runtime=constants)
            self.mock_image.open.assert_called_once_with("test.jpg")
            self.mock_model.generate.assert_called_once()
            call_kwargs = self.mock_model.generate.call_args.kwargs
            self.assertIn("images", call_kwargs)
            self.assertEqual(len(call_kwargs["images"]), 1)

    @patch("definers.constants.MODELS", new={"answer": None})
    @patch("definers.constants.PROCESSORS", new={})
    @patch("definers.constants.common_audio_formats", new=["wav"])
    def test_history_with_audio(self):
        with patch.dict(
            "sys.modules",
            {
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            constants.MODELS["answer"] = self.mock_model
            history = [{"role": "user", "content": {"path": "test.wav"}}]
            ml._answer(history, runtime=constants)
            self.mock_sf.read.assert_called_once_with("test.wav")
            self.mock_model.generate.assert_called_once()
            call_kwargs = self.mock_model.generate.call_args.kwargs
            self.assertIn("audios", call_kwargs)
            self.assertEqual(len(call_kwargs["audios"]), 1)

    @patch("definers.constants.MODELS", new={"answer": None})
    @patch("definers.constants.PROCESSORS", new={})
    def test_empty_history(self):
        with patch.dict(
            "sys.modules",
            {
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            constants.MODELS["answer"] = self.mock_model
            history = []
            ml._answer(history, runtime=constants)
            self.mock_model.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
