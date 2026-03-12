import unittest
from unittest.mock import patch

from definers import google_translate


class TestTextLogging(unittest.TestCase):
    @patch("definers.logger.info")
    def test_google_translate_logs_result(self, mock_info):

        res = google_translate("hello", lang="es")

        self.assertIsInstance(res, str)
        mock_info.assert_called_once_with(res)

    @patch("definers.logger.exception")
    def test_google_translate_handles_error(self, mock_exc):

        with patch("requests.get", side_effect=Exception("fail")):
            res = google_translate("hello", lang="es")
            self.assertEqual(res, "")
            mock_exc.assert_called_once()
