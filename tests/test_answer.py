import unittest
from unittest.mock import MagicMock, patch

import definers
from definers import answer


class TestAnswer(unittest.TestCase):
    def setUp(self):
        self.mock_sf = MagicMock()
        self.mock_sf.read.return_value = (
            MagicMock(),
            MagicMock(),
        )
        self.mock_image = MagicMock()
        self.mock_image.open.return_value = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model.generate.return_value = "A witty response."

    @patch("definers.MODELS", new={"answer": None})
    @patch(
        "definers.SYSTEM_MESSAGE",
        new={"message": "Mock System Message"},
    )
    def test_basic_text_history(self):
        with patch.dict(
            "sys.modules",
            {
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            definers.MODELS["answer"] = self.mock_model
            history = [
                {"role": "user", "content": "Hi there"},
                {
                    "role": "assistant",
                    "content": "Hello! How can I help?",
                },
                {"role": "user", "content": "Tell me a joke"},
            ]
            response = answer(history)
            self.assertEqual(response, "A witty response.")
            self.mock_model.generate.assert_called_once_with(
                prompt="<|system|>message: Mock System Message\n<|end|>\n<|user|>Hi there<|end|><|assistant|>Hello! How can I help?<|user|>Tell me a joke<|end|><|assistant|>",
                max_length=200,
                beam_width=16,
            )

    @patch("definers.MODELS", new={"answer": None})
    @patch("definers.iio_formats", new=["jpg"])
    def test_history_with_image(self):
        with patch.dict(
            "sys.modules",
            {
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            definers.MODELS["answer"] = self.mock_model
            history = [
                {"role": "user", "content": {"path": "test.jpg"}}
            ]
            answer(history)
            self.mock_image.open.assert_called_once_with("test.jpg")
            self.mock_model.generate.assert_called_once()
            call_kwargs = self.mock_model.generate.call_args.kwargs
            self.assertIn("images", call_kwargs)
            self.assertEqual(len(call_kwargs["images"]), 1)

    @patch("definers.MODELS", new={"answer": None})
    @patch("definers.common_audio_formats", new=["wav"])
    def test_history_with_audio(self):
        with patch.dict(
            "sys.modules",
            {
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            definers.MODELS["answer"] = self.mock_model
            history = [
                {"role": "user", "content": {"path": "test.wav"}}
            ]
            answer(history)
            self.mock_sf.read.assert_called_once_with("test.wav")
            self.mock_model.generate.assert_called_once()
            call_kwargs = self.mock_model.generate.call_args.kwargs
            self.assertIn("audios", call_kwargs)
            self.assertEqual(len(call_kwargs["audios"]), 1)

    @patch("definers.MODELS", new={"answer": None})
    def test_empty_history(self):
        with patch.dict(
            "sys.modules",
            {
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            definers.MODELS["answer"] = self.mock_model
            history = []
            answer(history)
            self.mock_model.generate.assert_called_once()


if __name__ == "__main__":
    import definers

    unittest.main()
