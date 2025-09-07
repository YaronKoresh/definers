import unittest
from unittest.mock import patch, MagicMock
from definers import _install_ffmpeg_windows, is_admin_windows

class TestInstallFfmpegWindows(unittest.TestCase):
    @patch("definers.is_admin_windows", return_value=True)
    @patch("definers.subprocess.run")
    def test_winget_install_succeeds(
        self, mock_run, mock_is_admin
    ):
        mock_run.return_value = MagicMock(
            check=True, stdout="Success", stderr=""
        )
        _install_ffmpeg_windows()
        mock_run.assert_called_once()
        self.assertIn("winget", mock_run.call_args[0][0])

    @patch("definers.is_admin_windows", return_value=False)
    @patch("builtins.print")
    @patch("definers.sys.exit", side_effect=SystemExit)
    def test_non_admin_exits(
        self, mock_exit, mock_print, mock_is_admin
    ):
        with self.assertRaises(SystemExit):
            _install_ffmpeg_windows()
        mock_print.assert_any_call(
            "[ERROR] This script requires Administrator privileges to run on Windows."
        )
        mock_exit.assert_called_once_with(1)

    @patch("definers.is_admin_windows", return_value=True)
    @patch(
        "definers.subprocess.run",
        side_effect=FileNotFoundError,
    )
    @patch("definers.requests.get")
    @patch("definers.zipfile.ZipFile")
    @patch("definers.shutil.move")
    @patch("definers.os.path.exists", return_value=False)
    @patch("definers.tempfile.gettempdir", return_value="/tmp")
    def test_manual_download_if_winget_fails(
        self,
        mock_gettempdir,
        mock_exists,
        mock_move,
        mock_zipfile,
        mock_requests_get,
        mock_run,
        mock_is_admin,
    ):
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"zip_data"]
        mock_requests_get.return_value.__enter__.return_value = (
            mock_response
        )
        _install_ffmpeg_windows()
        mock_requests_get.assert_called_once()
        mock_zipfile.assert_called_once()
        self.assertGreater(mock_move.call_count, 0)

if __name__ == "__main__":
    unittest.main()
