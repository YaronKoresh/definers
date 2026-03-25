import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from definers.application_ml import answer


class TestAnswer(unittest.TestCase):
    def setUp(self):
        self.mock_sf = MagicMock()
        self.mock_sf.read.return_value = (MagicMock(), MagicMock())
        self.mock_librosa = MagicMock()
        self.mock_librosa.load.return_value = (MagicMock(), 16000)
        self.mock_image = MagicMock()
        self.mock_image.open.return_value = MagicMock()
        self.mock_audio_module = MagicMock()
        self.mock_audio_module.audio_preview.return_value = None
        self.mock_image_module = MagicMock(
            get_max_resolution=MagicMock(return_value=(1024, 1024)),
            image_resolution=MagicMock(return_value=(1024, 1024)),
            resize_image=MagicMock(return_value=("test.jpg", MagicMock())),
        )
        self.mock_system_module = MagicMock(
            get_ext=MagicMock(side_effect=lambda path: path.rsplit(".", 1)[-1]),
            read=MagicMock(return_value=""),
        )
        self.mock_model = MagicMock()
        self.mock_model.generate.return_value = "A witty response."
        self.runtime = SimpleNamespace(
            MODELS={"answer": self.mock_model},
            PROCESSORS={},
            SYSTEM_MESSAGE="Mock System Message",
            common_audio_formats=["wav"],
            iio_formats=["jpg"],
        )

    def test_basic_text_history(self):
        with patch.dict(
            "sys.modules",
            {
                "definers.audio": self.mock_audio_module,
                "definers.image": self.mock_image_module,
                "definers.system": self.mock_system_module,
                "librosa": self.mock_librosa,
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            history = [
                {"role": "user", "content": "Hi there"},
                {"role": "assistant", "content": "Hello! How can I help?"},
                {"role": "user", "content": "Tell me a joke"},
            ]
            response = answer(history, runtime=self.runtime)
            self.assertEqual(response, "A witty response.")
            expected_prompt = "<|system|>Mock System Message<|end|><|user|>Hi there<|end|><|assistant|>Hello! How can I help?<|end|><|user|>Tell me a joke<|end|><|assistant|>"
            self.mock_model.generate.assert_called_once_with(
                prompt=expected_prompt, max_length=200, beam_width=16
            )

    def test_history_with_image(self):
        with patch.dict(
            "sys.modules",
            {
                "definers.audio": self.mock_audio_module,
                "definers.image": self.mock_image_module,
                "definers.system": self.mock_system_module,
                "librosa": self.mock_librosa,
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            history = [{"role": "user", "content": {"path": "test.jpg"}}]
            answer(history, runtime=self.runtime)
            self.mock_image.open.assert_called_once_with("test.jpg")
            self.mock_model.generate.assert_called_once()
            call_kwargs = self.mock_model.generate.call_args.kwargs
            self.assertIn("images", call_kwargs)
            self.assertEqual(len(call_kwargs["images"]), 1)

    def test_history_with_audio(self):
        with patch.dict(
            "sys.modules",
            {
                "definers.audio": self.mock_audio_module,
                "definers.image": self.mock_image_module,
                "definers.system": self.mock_system_module,
                "librosa": self.mock_librosa,
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            history = [{"role": "user", "content": {"path": "test.wav"}}]
            answer(history, runtime=self.runtime)
            self.mock_sf.read.assert_called_once_with("test.wav")
            self.mock_model.generate.assert_called_once()
            call_kwargs = self.mock_model.generate.call_args.kwargs
            self.assertIn("audios", call_kwargs)
            self.assertEqual(len(call_kwargs["audios"]), 1)

    def test_empty_history(self):
        with patch.dict(
            "sys.modules",
            {
                "definers.audio": self.mock_audio_module,
                "definers.image": self.mock_image_module,
                "definers.system": self.mock_system_module,
                "librosa": self.mock_librosa,
                "soundfile": self.mock_sf,
                "PIL": MagicMock(Image=self.mock_image),
            },
        ):
            history = []
            answer(history, runtime=self.runtime)
            self.mock_model.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
