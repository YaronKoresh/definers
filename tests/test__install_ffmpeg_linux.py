import unittest
from unittest.mock import call, patch

from definers import _install_ffmpeg_linux


class TestInstallFfmpegLinux(unittest.TestCase):
    @patch("os.geteuid", return_value=0)
    @patch("definers.subprocess.run")
    @patch("shutil.which", side_effect=["apt-get", None])
    @patch("builtins.print")
    def test_apt_get_install(
        self, mock_print, mock_which, mock_run, mock_geteuid
    ):
        _install_ffmpeg_linux()
        mock_run.assert_has_calls(
            [
                call(["apt-get", "update"], check=True),
                call(
                    ["apt-get", "install", "ffmpeg", "-y"], check=True
                ),
            ]
        )
        mock_print.assert_any_call(
            "\n[SUCCESS] FFmpeg installed successfully."
        )

    @patch("os.geteuid", return_value=0)
    @patch("definers.subprocess.run")
    @patch("shutil.which", side_effect=[None, "dnf", None])
    @patch("builtins.print")
    def test_dnf_install(
        self, mock_print, mock_which, mock_run, mock_geteuid
    ):
        _install_ffmpeg_linux()
        mock_run.assert_called_once_with(
            ["dnf", "install", "ffmpeg", "-y"], check=True
        )
        mock_print.assert_any_call(
            "\n[SUCCESS] FFmpeg installed successfully."
        )

    @patch("os.geteuid", return_value=0)
    @patch("definers.subprocess.run")
    @patch("shutil.which", side_effect=[None, None, "pacman"])
    @patch("builtins.print")
    def test_pacman_install(
        self, mock_print, mock_which, mock_run, mock_geteuid
    ):
        _install_ffmpeg_linux()
        mock_run.assert_called_once_with(
            ["pacman", "-S", "ffmpeg", "--noconfirm"], check=True
        )
        mock_print.assert_any_call(
            "\n[SUCCESS] FFmpeg installed successfully."
        )

    @patch("os.geteuid", return_value=1000)
    @patch("shutil.which", return_value="/usr/bin/apt")
    @patch("subprocess.run")
    def test_permission_denied_triggers_exit(
        self, mock_subprocess_run, mock_which, mock_geteuid, capsys
    ):
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=13, cmd=["apt-get", "update"]
        )

        with pytest.raises(SystemExit) as excinfo:
            _install_ffmpeg_linux()

        assert excinfo.value.code == 1

        captured = capsys.readouterr()
        assert "[WARN] This script needs sudo privileges" in captured.out
        assert "[ERROR] The installation command failed" in captured.out

    @patch("os.geteuid", return_value=0)
    @patch("shutil.which", return_value=None)
    @patch("definers.sys.exit", side_effect=SystemExit)
    @patch("builtins.print")
    def test_no_package_manager_found(
        self, mock_print, mock_exit, mock_which, mock_geteuid
    ):
        with self.assertRaises(SystemExit):
            _install_ffmpeg_linux()
        mock_print.assert_any_call(
            "[ERROR] Could not detect a supported package manager (apt, dnf, pacman)."
        )
        mock_exit.assert_called_once_with(1)

    @patch("os.geteuid", return_value=0)
    @patch("shutil.which", return_value="apt-get")
    @patch(
        "definers.subprocess.run", side_effect=Exception("Test error")
    )
    @patch("definers.sys.exit", side_effect=SystemExit)
    @patch("builtins.print")
    def test_unexpected_error(
        self,
        mock_print,
        mock_exit,
        mock_run,
        mock_which,
        mock_geteuid,
    ):
        with self.assertRaises(SystemExit):
            _install_ffmpeg_linux()
        mock_print.assert_any_call(
            "\n[ERROR] An unexpected error occurred: Test error"
        )
        mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()