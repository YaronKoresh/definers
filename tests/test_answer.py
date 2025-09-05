import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Mocking PIL and soundfile before they are imported by definers
import sys
from PIL import Image
mock_image = MagicMock(spec=Image.Image)
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock(open=MagicMock(return_value=mock_image))

mock_sf = MagicMock()
mock_sf.read.return_value = (np.array([1, 2, 3]), 44100)
sys.modules['soundfile'] = mock_sf

from definers import answer

class TestAnswer(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.generate.return_value = "Mocked AI response"
        
        self.patcher_model = patch.dict('definers.MODELS', {'answer': self.mock_model})
        self.patcher_merge = patch('definers.merge_system_message', return_value="<|system|>Mock System Message<|end|>\n")
        self.patcher_log = patch('definers.log')

        self.patcher_model.start()
        self.mock_merge = self.patcher_merge.start()
        self.mock_log = self.patcher_log.start()

    def tearDown(self):
        self.patcher_model.stop()
        self.patcher_merge.stop()
        self.patcher_log.stop()

    def test_simple_text_history(self):
        history = [{"role": "user", "content": "Hello"}]
        response = answer(history)

        expected_prompt = "<|system|>Mock System Message<|end|>\n<|user|>Hello<|end|><|assistant|>"
        self.mock_model.generate.assert_called_once_with(prompt=expected_prompt, max_length=200, beam_width=16)
        self.assertEqual(response, "Mocked AI response")

    def test_multi_turn_history(self):
        history = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me a joke"}
        ]
        response = answer(history)

        expected_prompt = (
            "<|system|>Mock System Message<|end|>\n"
            "<|user|>Hi there<|end|><|end|>"
            "<|assistant|>Hello! How can I help?<|user|>Tell me a joke<|end|><|assistant|>"
        )
        self.mock_model.generate.assert_called_once_with(prompt=expected_prompt, max_length=200, beam_width=16)
        self.assertEqual(response, "Mocked AI response")

    @patch('definers.iio_formats', new=['jpg'])
    def test_history_with_image(self):
        history = [{"role": "user", "content": {"path": "test.jpg"}}]
        answer(history)

        expected_prompt = "<|system|>Mock System Message<|end|>\n<|user|><|image_1|><|end|><|assistant|>"
        
        generate_kwargs = self.mock_model.generate.call_args.kwargs
        self.assertEqual(generate_kwargs['prompt'], expected_prompt)
        self.assertIn('images', generate_kwargs)
        self.assertEqual(len(generate_kwargs['images']), 1)

    @patch('definers.common_audio_formats', new=['wav'])
    def test_history_with_audio(self):
        history = [{"role": "user", "content": {"path": "test.wav"}}]
        answer(history)

        expected_prompt = "<|system|>Mock System Message<|end|>\n<|user|><|audio_1|><|end|><|assistant|>"
        
        generate_kwargs = self.mock_model.generate.call_args.kwargs
        self.assertEqual(generate_kwargs['prompt'], expected_prompt)
        self.assertIn('audios', generate_kwargs)
        self.assertEqual(len(generate_kwargs['audios']), 1)
        self.assertEqual(generate_kwargs['audios'][0][1], 44100)

    @patch('definers.iio_formats', new=['png'])
    @patch('definers.common_audio_formats', new=['mp3'])
    def test_history_with_mixed_media_and_text(self):
        history = [
            {"role": "user", "content": "Describe this:"},
            {"role": "user", "content": ({"path": "image.png"}, {"path": "sound.mp3"})}
        ]
        answer(history)

        expected_prompt = (
            "<|system|>Mock System Message<|end|>\n"
            "<|user|>Describe this:<|end|>"
            "<|user|><|image_1|><|audio_1|><|end|><|assistant|>"
        )

        generate_kwargs = self.mock_model.generate.call_args.kwargs
        self.assertEqual(generate_kwargs['prompt'], expected_prompt)
        self.assertIn('images', generate_kwargs)
        self.assertEqual(len(generate_kwargs['images']), 1)
        self.assertIn('audios', generate_kwargs)
        self.assertEqual(len(generate_kwargs['audios']), 1)


if __name__ == '__main__':
    unittest.main()
