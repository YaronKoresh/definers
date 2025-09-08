import unittest
from unittest.mock import MagicMock, patch

from definers import _install_ffmpeg_windows, is_admin_windows


class TestInstallFfmpegWindows(unittest.TestCase):
    @patch("definers.is_admin_windows", return_value=True)
    @patch("definers.subprocess.run")
    def test_winget_install_succeeds(self, mock_run, mock_is_admin):
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
        side_effect=[
            FileNotFoundError,
            MagicMock(),
        ],  # Winget fails, setx succeeds
    )
    @patch("requests.get")
    @patch("zipfile.ZipFile")
    @patch("definers.shutil.move")
    @patch("definers.shutil.rmtree")
    @patch("definers.os.remove")
    @patch(
        "definers.os.path.exists", return_value=True
    )  # Mock existence for cleanup
    @patch("definers.os.listdir", return_value=["ffmpeg-build-123"])
    @patch("definers.tempfile.gettempdir", return_value="/tmp")
    def test_manual_download_if_winget_fails(
        self,
        mock_gettempdir,
        mock_listdir,
        mock_exists,
        mock_os_remove,
        mock_rmtree,
        mock_move,
        mock_zipfile,
        mock_requests_get,
        mock_run,
        mock_is_admin,
    ):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"zip_data"]
        mock_requests_get.return_value.__enter__.return_value = (
            mock_response
        )

        # Mock the context manager for zipfile.ZipFile
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = (
            mock_zip_instance
        )

        _install_ffmpeg_windows()

        # Assertions
        self.assertEqual(mock_run.call_count, 2)  # winget and setx
        mock_requests_get.assert_called_once()
        mock_zipfile.assert_called_once_with("/tmp/ffmpeg.zip", "r")
        mock_zip_instance.extractall.assert_called_once_with(
            "/tmp/ffmpeg_extracted"
        )
        mock_move.assert_called_once()


if __name__ == "__main__":
    unittest.main()
