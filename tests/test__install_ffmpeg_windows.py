import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, call, mock_open, patch

from definers import _install_ffmpeg_windows


class TestInstallFfmpegWindows(unittest.TestCase):

    @patch("definers.is_admin_windows", return_value=False)
    @patch("definers.sys.exit")
    @patch("builtins.print")
    def test_not_admin_exits(
        self, mock_print, mock_exit, mock_is_admin
    ):
        _install_ffmpeg_windows()
        mock_exit.assert_called_once_with(1)
        mock_print.assert_any_call(
            "[ERROR] This script requires Administrator privileges to run on Windows."
        )

    @patch("definers.is_admin_windows", return_value=True)
    @patch("subprocess.run")
    @patch("builtins.print")
    def test_winget_success(
        self, mock_print, mock_run, mock_is_admin
    ):
        mock_run.return_value = MagicMock(returncode=0)
        _install_ffmpeg_windows()
        mock_run.assert_called_once_with(
            [
                "winget",
                "install",
                "--id=Gyan.FFmpeg.Essentials",
                "-e",
                "--accept-source-agreements",
                "--accept-package-agreements",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        mock_print.assert_any_call(
            "[SUCCESS] FFmpeg has been installed via Winget."
        )

    @patch("definers.is_admin_windows", return_value=True)
    @patch("subprocess.run")
    @patch("requests.get")
    @patch("zipfile.ZipFile")
    @patch("os.path.exists", return_value=True)
    @patch("os.listdir", return_value=["ffmpeg-build-dir"])
    @patch("shutil.move")
    @patch("shutil.rmtree")
    @patch("os.remove")
    @patch("tempfile.gettempdir", return_value="/tmp")
    @patch("builtins.print")
    @patch("definers.sys.exit")
    def test_manual_install_success_after_winget_fail(
        self,
        mock_exit,
        mock_print,
        mock_gettempdir,
        mock_remove,
        mock_rmtree,
        mock_move,
        mock_listdir,
        mock_exists,
        mock_zipfile,
        mock_requests_get,
        mock_subprocess_run,
        mock_is_admin,
    ):
        mock_subprocess_run.side_effect = [
            subprocess.CalledProcessError(1, "winget"),
            MagicMock(returncode=0),
        ]

        zip_content = io.BytesIO()
        with zipfile.ZipFile(zip_content, "w") as zf:
            zf.writestr(
                "ffmpeg-build-dir/bin/ffmpeg.exe", b"ffmpeg_data"
            )
        zip_content.seek(0)

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [zip_content.read()]
        mock_requests_get.return_value.__enter__.return_value = (
            mock_response
        )

        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = (
            mock_zip_instance
        )

        with patch("builtins.open", mock_open()) as mock_file:
            _install_ffmpeg_windows()

        mock_requests_get.assert_called_once()
        mock_zipfile.assert_called_with("/tmp/ffmpeg.zip", "r")
        mock_zip_instance.extractall.assert_called_with(
            "/tmp/ffmpeg_extracted"
        )

        program_files = os.environ.get(
            "ProgramFiles", "C:\\Program Files"
        )
        ffmpeg_install_dir = os.path.join(program_files, "ffmpeg")

        mock_move.assert_called_once_with(
            os.path.join(
                "/tmp/ffmpeg_extracted", "ffmpeg-build-dir", "bin"
            ),
            ffmpeg_install_dir,
        )

        mock_subprocess_run.assert_called_with(
            ["setx", "/M", "PATH", f"%PATH%;{ffmpeg_install_dir}"],
            check=True,
        )

        mock_print.assert_any_call(
            "[SUCCESS] FFmpeg added to system PATH."
        )
        mock_remove.assert_called_with("/tmp/ffmpeg.zip")
        self.assertTrue(mock_rmtree.called)
        mock_exit.assert_not_called()

    @patch("definers.is_admin_windows", return_value=True)
    @patch("subprocess.run", side_effect=FileNotFoundError)
    @patch("requests.get", side_effect=Exception("Download failed"))
    @patch("builtins.print")
    @patch("definers.sys.exit")
    def test_manual_install_download_fails(
        self,
        mock_exit,
        mock_print,
        mock_requests_get,
        mock_subprocess_run,
        mock_is_admin,
    ):
        _install_ffmpeg_windows()
        mock_print.assert_any_call(
            "\n[ERROR] An error occurred during manual installation: Download failed"
        )
        mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
