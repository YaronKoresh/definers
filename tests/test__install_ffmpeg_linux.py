import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from definers import _install_ffmpeg_linux


class TestInstallFfmpegLinux(unittest.TestCase):

    @patch("os.geteuid", return_value=1000)
    @patch(
        "shutil.which",
        side_effect=lambda cmd: (
            "/usr/bin/apt" if cmd == "apt" else None
        ),
    )
    @patch("subprocess.run")
    @patch("builtins.print")
    def test_apt_install_as_non_root(
        self, mock_print, mock_run, mock_which, mock_geteuid
    ):
        mock_run.return_value = MagicMock(returncode=0)
        _install_ffmpeg_linux()
        mock_print.assert_any_call(
            "[WARN] This script needs sudo privileges to install packages."
        )
        mock_run.assert_any_call(["apt-get", "update"], check=True)
        mock_run.assert_any_call(
            ["apt-get", "install", "ffmpeg", "-y"], check=True
        )
        mock_print.assert_any_call(
            "\n[SUCCESS] FFmpeg installed successfully."
        )

    @patch("os.geteuid", return_value=0)
    @patch(
        "shutil.which",
        side_effect=lambda cmd: (
            "/usr/bin/dnf" if cmd == "dnf" else None
        ),
    )
    @patch("subprocess.run")
    @patch("builtins.print")
    def test_dnf_install_as_root(
        self, mock_print, mock_run, mock_which, mock_geteuid
    ):
        mock_run.return_value = MagicMock(returncode=0)
        _install_ffmpeg_linux()
        mock_run.assert_called_once_with(
            ["dnf", "install", "ffmpeg", "-y"], check=True
        )
        mock_print.assert_any_call(
            "\n[SUCCESS] FFmpeg installed successfully."
        )

    @patch("os.geteuid", return_value=0)
    @patch(
        "shutil.which",
        side_effect=lambda cmd: (
            "/usr/bin/pacman" if cmd == "pacman" else None
        ),
    )
    @patch("subprocess.run")
    @patch("builtins.print")
    def test_pacman_install_as_root(
        self, mock_print, mock_run, mock_which, mock_geteuid
    ):
        mock_run.return_value = MagicMock(returncode=0)
        _install_ffmpeg_linux()
        mock_run.assert_called_once_with(
            ["pacman", "-S", "ffmpeg", "--noconfirm"], check=True
        )
        mock_print.assert_any_call(
            "\n[SUCCESS] FFmpeg installed successfully."
        )

    @patch("os.geteuid", return_value=0)
    @patch("shutil.which", return_value=None)
    @patch("definers.sys.exit")
    @patch("builtins.print")
    def test_no_package_manager_found(
        self, mock_print, mock_exit, mock_which, mock_geteuid
    ):
        _install_ffmpeg_linux()
        mock_print.assert_any_call(
            "[ERROR] Could not detect a supported package manager (apt, dnf, pacman)."
        )
        mock_exit.assert_called_once_with(1)

    @patch("os.geteuid", return_value=0)
    @patch(
        "shutil.which",
        side_effect=lambda cmd: (
            "/usr/bin/apt" if cmd == "apt" else None
        ),
    )
    @patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "cmd"),
    )
    @patch("definers.sys.exit")
    @patch("builtins.print")
    def test_install_command_fails(
        self,
        mock_print,
        mock_exit,
        mock_run,
        mock_which,
        mock_geteuid,
    ):
        _install_ffmpeg_linux()
        mock_print.assert_any_call(
            "\n[ERROR] The installation command failed with exit code 1."
        )
        mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
