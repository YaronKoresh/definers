import unittest
from unittest.mock import MagicMock, patch

import torch

from definers import MODELS, init_upscale


class TestInitUpscale(unittest.TestCase):

    def setUp(self):
        self.mock_state_dict = {
            "model.0.weight": torch.randn(64, 3, 3, 3),
            "model.1.sub.22.RDB1.conv1.0.weight": torch.randn(1),
            "model.7.weight": torch.randn(64, 64, 3, 3),
            "model.9.weight": torch.randn(64, 64, 3, 3),
            "model.12.weight": torch.randn(3, 64, 3, 3),
        }

    @patch("definers.MODELS", {})
    @patch("definers.pillow_heif")
    @patch("definers.device", return_value="cpu")
    @patch("definers.dtype", return_value=torch.float32)
    @patch(
        "definers.hf_hub_download", return_value="mock/path/model.pth"
    )
    @patch("torch.load")
    def test_successful_initialization(
        self,
        mock_torch_load,
        mock_hf_hub_download,
        mock_dtype,
        mock_device,
        mock_pillow_heif,
    ):

        mock_torch_load.return_value = self.mock_state_dict

        init_upscale()

        self.assertIn("upscale", MODELS)
        self.assertIsNotNone(MODELS["upscale"])

        mock_pillow_heif.register_heif_opener.assert_called_once()
        self.assertTrue(mock_hf_hub_download.called)

        # We expect torch.load to be called for the ESRGAN model and negative embedding
        self.assertGreaterEqual(mock_torch_load.call_count, 1)

        # Check if the final model has the expected methods
        self.assertTrue(hasattr(MODELS["upscale"], "upscale"))
        self.assertTrue(hasattr(MODELS["upscale"], "to"))

    @patch(
        "definers.hf_hub_download",
        side_effect=Exception("Download failed"),
    )
    def test_hf_hub_download_fails(self, mock_hf_hub_download):
        with self.assertRaises(Exception) as context:
            init_upscale()
        self.assertEqual(str(context.exception), "Download failed")

    @patch(
        "definers.hf_hub_download", return_value="mock/path/model.pth"
    )
    @patch("torch.load", side_effect=IOError("Could not load file"))
    def test_torch_load_fails(
        self, mock_torch_load, mock_hf_hub_download
    ):
        with self.assertRaises(IOError) as context:
            init_upscale()
        self.assertEqual(
            str(context.exception), "Could not load file"
        )

    @patch("definers.MODELS", {})
    @patch("definers.pillow_heif")
    @patch("definers.device", return_value="cpu")
    @patch("definers.dtype", return_value=torch.float32)
    @patch(
        "definers.hf_hub_download", return_value="mock/path/model.pth"
    )
    @patch("torch.load")
    def test_model_assigned_correctly(
        self,
        mock_torch_load,
        mock_hf_hub_download,
        mock_dtype,
        mock_device,
        mock_pillow_heif,
    ):

        mock_torch_load.return_value = self.mock_state_dict

        init_upscale()

        self.assertIn("upscale", MODELS)
        self.assertIsNotNone(MODELS["upscale"])


if __name__ == "__main__":
    unittest.main()
