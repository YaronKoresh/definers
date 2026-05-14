import unittest
from unittest.mock import MagicMock, patch

from definers.system import install_ffmpeg_windows


class TestInstallFfmpegWindows(unittest.TestCase):
    @patch("definers.system.is_admin_windows", return_value=True)
    @patch("subprocess.run")
    def test_winget_install_succeeds(self, mock_run, mock_is_admin):
        mock_run.return_value = MagicMock(
            check=True, stdout="Success", stderr=""
        )
        install_ffmpeg_windows()
        mock_run.assert_called_once()
        self.assertIn("winget", mock_run.call_args[0][0])

    @patch("definers.system.is_admin_windows", return_value=False)
    @patch("builtins.print")
    @patch("sys.exit", side_effect=SystemExit)
    def test_non_admin_exits(self, mock_exit, mock_print, mock_is_admin):
        with self.assertRaises(SystemExit):
            install_ffmpeg_windows()
        mock_print.assert_any_call(
            "[ERROR] This script requires Administrator privileges to run on Windows."
        )
        mock_exit.assert_called_once_with(1)

    @patch("definers.system.is_admin_windows", return_value=True)
    @patch("subprocess.run", side_effect=[FileNotFoundError, MagicMock()])
    @patch("definers.media.web_transfer.download_file")
    @patch("zipfile.ZipFile")
    @patch("shutil.move")
    @patch("shutil.rmtree")
    @patch("os.remove")
    @patch("os.path.exists", return_value=True)
    @patch("os.listdir", return_value=["ffmpeg-build-123"])
    @patch("tempfile.gettempdir", return_value="/tmp")
    def test_manual_download_if_winget_fails(
        self,
        mock_gettempdir,
        mock_listdir,
        mock_exists,
        mock_os_remove,
        mock_rmtree,
        mock_move,
        mock_zipfile,
        mock_download_file,
        mock_run,
        mock_is_admin,
    ):
        mock_download_file.return_value = "/tmp/ffmpeg.zip"
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
        install_ffmpeg_windows()
        self.assertEqual(mock_run.call_count, 2)
        mock_download_file.assert_called_once_with(
            "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
            "/tmp/ffmpeg.zip",
        )
        mock_zipfile.assert_called_once_with("/tmp/ffmpeg.zip", "r")
        mock_zip_instance.extractall.assert_called_once_with(
            "/tmp/ffmpeg_extracted"
        )
        mock_move.assert_called_once()


if __name__ == "__main__":
    unittest.main()
