import unittest
from unittest.mock import patch

from definers import language


class TestLanguage(unittest.TestCase):

    @patch("langdetect.detect", return_value="en")
    def test_detect_english(self, mock_detect):
        result = language("This is a test sentence.")
        self.assertEqual(result, "en")
        mock_detect.assert_called_once_with(
            "This is a test sentence."
        )

    @patch("langdetect.detect", return_value="fr")
    def test_detect_french(self, mock_detect):
        result = language("Ceci est une phrase de test.")
        self.assertEqual(result, "fr")
        mock_detect.assert_called_once_with(
            "Ceci est une phrase de test."
        )

    @patch(
        "langdetect.detect", side_effect=Exception("Detection failed")
    )
    def test_language_detection_error(self, mock_detect):
        with self.assertRaises(Exception):
            language("!@#$%^&*")


if __name__ == "__main__":
    unittest.main()
