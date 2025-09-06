import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from definers import answer

class TestAnswer(unittest.TestCase):

    @patch('definers.MODELS')
    @patch('definers.merge_system_message', return_value="<|system|>SYS<|end|>")
    def test_empty_history(self, mock_merge, mock_models):
        mock_models["answer"].generate.return_value = "Empty response"
        history = []
        response = answer(history)
        self.assertEqual(response, "Empty response")
        mock_models["answer"].generate.assert_called_once_with(
            prompt='<|system|>SYS<|end|><|assistant|>',
            max_length=200,
            beam_width=16
        )

    @patch('definers.MODELS')
    @patch('definers.merge_system_message', return_value="")
    def test_simple_user_message(self, mock_merge, mock_models):
        mock_models["answer"].generate.return_value = "Hello there"
        history = [{"role": "user", "content": "Hi"}]
        answer(history)
        mock_models["answer"].generate.assert_called_once_with(
            prompt='<|user|>Hi<|end|><|assistant|>',
            max_length=200,
            beam_width=16
        )

    @patch('definers.MODELS')
    @patch('definers.merge_system_message', return_value="")
    def test_user_and_assistant_history(self, mock_merge, mock_models):
        mock_models["answer"].generate.return_value = "I can help"
        history = [
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "How?"},
            {"role": "user", "content": "With code"},
        ]
        answer(history)
        mock_models["answer"].generate.assert_called_once_with(
            prompt='<|user|>Help me<|end|><|assistant|>How?<|end|><|user|>With code<|end|><|assistant|>',
            max_length=200,
            beam_width=16
        )

    @patch('definers.MODELS')
    @patch('definers.Image.open')
    @patch('definers.iio_formats', new=["png"])
    @patch('definers.merge_system_message', return_value="")
    def test_history_with_image(self, mock_merge, mock_open, mock_models):
        mock_models["answer"].generate.return_value = "Nice image"
        mock_image = MagicMock()
        mock_open.return_value = mock_image
        history = [{"role": "user", "content": {"path": "test.png"}}]
        answer(history)
        mock_open.assert_called_once_with("test.png")
        mock_models["answer"].generate.assert_called_once_with(
            prompt='<|user|><|image_1|><|end|><|assistant|>',
            max_length=200,
            beam_width=16,
            images=[mock_image]
        )

    @patch('definers.MODELS')
    @patch('definers.sf.read')
    @patch('definers.common_audio_formats', new=["wav"])
    @patch('definers.merge_system_message', return_value="")
    def test_history_with_audio(self, mock_merge, mock_read, mock_models):
        mock_models["answer"].generate.return_value = "Nice audio"
        mock_audio_data = (np.array([0.1, 0.2]), 44100)
        mock_read.return_value = mock_audio_data
        history = [{"role": "user", "content": {"path": "test.wav"}}]
        answer(history)
        mock_read.assert_called_once_with("test.wav")
        mock_models["answer"].generate.assert_called_once_with(
            prompt='<|user|><|audio_1|><|end|><|assistant|>',
            max_length=200,
            beam_width=16,
            audios=[mock_audio_data]
        )

if __name__ == '__main__':
    unittest.main()