import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
from definers import upscale, MODELS

class TestUpscale(unittest.TestCase):

    @patch('definers.save_image', return_value="upscaled_image.png")
    @patch('definers.MODELS', {'upscale': MagicMock()})
    @patch('definers.manual_seed')
    @patch('PIL.Image.open')
    @patch('definers.random')
    @patch('definers.big_number', return_value=1000000)
    def test_successful_upscale_with_defaults(self, mock_big_number, mock_random, mock_open, mock_manual_seed, mock_save):
        mock_input_image = MagicMock(spec=Image.Image)
        mock_open.return_value = mock_input_image

        mock_upscaled_image = MagicMock(spec=Image.Image)
        MODELS['upscale'].upscale.return_value = mock_upscaled_image
        
        mock_random.randint.return_value = 12345

        result = upscale("test.png")

        mock_open.assert_called_once_with("test.png")
        mock_random.randint.assert_called_once_with(0, 1000000)
        mock_manual_seed.assert_called_once_with(12345)
        
        MODELS['upscale'].upscale.assert_called_once()
        
        mock_save.assert_called_once_with(mock_upscaled_image)
        self.assertEqual(result, "upscaled_image.png")

    @patch('definers.MODELS', {'upscale': MagicMock()})
    def test_invalid_upscale_factor_too_low(self, mock_models):
        result = upscale("test.png", upscale_factor=1)
        self.assertIsNone(result)
        mock_models['upscale'].upscale.assert_not_called()

    @patch('definers.MODELS', {'upscale': MagicMock()})
    def test_invalid_upscale_factor_too_high(self, mock_models):
        result = upscale("test.png", upscale_factor=5)
        self.assertIsNone(result)
        mock_models['upscale'].upscale.assert_not_called()

    @patch('definers.save_image')
    @patch('definers.MODELS', {'upscale': MagicMock()})
    @patch('definers.manual_seed')
    @patch('PIL.Image.open')
    @patch('definers.random')
    def test_custom_parameters_and_seed(self, mock_random, mock_open, mock_manual_seed, mock_models, mock_save):
        mock_open.return_value = MagicMock(spec=Image.Image)
        
        upscale(
            path="custom.png",
            upscale_factor=3,
            prompt="A cat",
            negative_prompt="A dog",
            seed=42,
            controlnet_scale=0.7,
            denoise_strength=0.3,
            solver="Euler"
        )
        
        mock_random.randint.assert_not_called()
        mock_manual_seed.assert_called_once_with(42)
        
        mock_models['upscale'].upscale.assert_called_once()
        call_args = mock_models['upscale'].upscale.call_args[1]
        
        self.assertEqual(call_args['prompt'], "A cat")
        self.assertEqual(call_args['negative_prompt'], "A dog")
        self.assertEqual(call_args['upscale_factor'], 3)
        self.assertEqual(call_args['controlnet_scale'], 0.7)
        self.assertEqual(call_args['denoise_strength'], 0.3)
        self.assertIn('Euler', str(call_args['solver_type']))

    @patch('definers.save_image')
    @patch('definers.MODELS', {'upscale': MagicMock()})
    @patch('PIL.Image.open', side_effect=FileNotFoundError("File not found"))
    def test_file_not_found(self, mock_open, mock_models, mock_save):
        with self.assertRaises(FileNotFoundError):
            upscale("non_existent.png")
        
        mock_models['upscale'].upscale.assert_not_called()
        mock_save.assert_not_called()

if __name__ == '__main__':
    unittest.main()
