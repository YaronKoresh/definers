import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from definers import MODELS, _negative_prompt_, upscale


class TestUpscale(unittest.TestCase):

    def setUp(self):
        self.image_path = "dummy_image.png"
        Image.new("RGB", (100, 100)).save(self.image_path)

        self.mock_upscaler = MagicMock()
        self.mock_upscaler.upscale.return_value = Image.new(
            "RGB", (200, 200)
        )
        MODELS["upscale"] = self.mock_upscaler

    def tearDown(self):
        import os

        if os.path.exists(self.image_path):
            os.remove(self.image_path)
        MODELS["upscale"] = None

    @patch("definers.save_image", return_value="upscaled_image.png")
    @patch("refiners.fluxion.utils.manual_seed")
    def test_successful_upscale_with_defaults(
        self, mock_manual_seed, mock_save_image
    ):
        result = upscale(self.image_path)

        mock_manual_seed.assert_called_once()
        self.mock_upscaler.upscale.assert_called_once()
        mock_save_image.assert_called_once()
        self.assertEqual(result, "upscaled_image.png")

    @patch("definers.save_image", return_value="upscaled_image.png")
    @patch("refiners.fluxion.utils.manual_seed")
    def test_custom_parameters_and_seed(
        self, mock_manual_seed, mock_save_image
    ):
        upscale(
            self.image_path,
            upscale_factor=3,
            prompt="Test Prompt",
            negative_prompt="Test Negative",
            seed=12345,
            num_inference_steps=30,
        )

        mock_manual_seed.assert_called_with(12345)
        self.mock_upscaler.upscale.assert_called_with(
            image=unittest.mock.ANY,
            prompt="Test Prompt",
            negative_prompt="Test Negative",
            upscale_factor=3,
            controlnet_scale=unittest.mock.ANY,
            controlnet_scale_decay=unittest.mock.ANY,
            condition_scale=unittest.mock.ANY,
            tile_size=(unittest.mock.ANY, unittest.mock.ANY),
            denoise_strength=unittest.mock.ANY,
            num_inference_steps=30,
            loras_scale=unittest.mock.ANY,
            solver_type=unittest.mock.ANY,
        )

    def test_invalid_upscale_factor_too_low(self):
        result = upscale(self.image_path, upscale_factor=1)
        self.assertIsNone(result)
        self.mock_upscaler.upscale.assert_not_called()

    def test_invalid_upscale_factor_too_high(self):
        result = upscale(self.image_path, upscale_factor=5)
        self.assertIsNone(result)
        self.mock_upscaler.upscale.assert_not_called()

    @patch(
        "PIL.Image.open",
        side_effect=FileNotFoundError("File not found"),
    )
    def test_file_not_found(self, mock_image_open):
        with self.assertRaises(FileNotFoundError):
            upscale("non_existent_file.png")
        self.mock_upscaler.upscale.assert_not_called()


if __name__ == "__main__":
    unittest.main()
