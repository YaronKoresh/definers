import unittest
from unittest.mock import call, patch

from definers import apt_install


class TestAptInstall(unittest.TestCase):

    @patch("definers.run")
    @patch("definers.post_install")
    @patch("definers.pre_install")
    def test_apt_install_calls_all_stages(
        self, mock_pre_install, mock_post_install, mock_run
    ):
        apt_install()

        mock_pre_install.assert_called_once()

        basic_apt = "build-essential gcc cmake swig gdebi git git-lfs wget curl libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev initramfs-tools libgirepository1.0-dev libdbus-1-dev libdbus-glib-1-dev libsecret-1-0 libmanette-0.2-0 libharfbuzz0b libharfbuzz-icu0 libenchant-2-2 libhyphen0 libwoff1 libgraphene-1.0-0 libxml2-dev libxmlsec1-dev"
        audio_apt = "libportaudio2 libasound2-dev sox libsox-fmt-all praat ffmpeg libavcodec-extra libavif-dev"
        visual_apt = "libopenblas-dev libgflags-dev libgles2 libgtk-3-0 libgtk-4-1 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libatspi2.0-0 libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-gl"

        expected_install_command = f"apt-get install -y { basic_apt } { audio_apt } { visual_apt }"

        expected_run_calls = [
            call("apt-get update"),
            call(expected_install_command),
        ]
        mock_run.assert_has_calls(expected_run_calls, any_order=True)

        mock_post_install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
