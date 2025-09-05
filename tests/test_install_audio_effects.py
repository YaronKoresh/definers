import unittest
from unittest.mock import patch, MagicMock
import os
import sys

from definers import install_audio_effects

class TestInstallAudioEffects(unittest.TestCase):

    @patch('definers.get_os_name', return_value='linux')
    @patch('definers.run')
    @patch('builtins.print')
    def test_install_on_linux(self, mock_print, mock_run, mock_get_os):
        install_audio_effects()
        mock_run.assert_any_call("apt-get update -y")
        mock_run.assert_any_call("apt-get install -y rubberband-cli fluidsynth fluid-soundfont-gm build-essential")
        mock_print.assert_any_call("\nInstalling Python packages with pip...")

    @patch('definers.get_os_name', return_value='windows')
    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    @patch('definers.download_and_unzip', return_value=True)
    @patch('definers.download_file', return_value=True)
    @patch('definers.add_to_path_windows')
    @patch('os.listdir', return_value=['rubberband-dir', 'fluidsynth-dir'])
    @patch.dict(os.environ, {'PATH': ''})
    @patch('builtins.print')
    def test_install_on_windows_first_time(
        self, mock_print, mock_listdir, mock_add_to_path, mock_download_file,
        mock_download_unzip, mock_makedirs, mock_exists, mock_get_os
    ):
        install_audio_effects()
        
        install_dir = os.path.join(os.path.expanduser("~"), "app_dependencies")
        
        mock_download_unzip.assert_any_call(
            "https://breakfastquay.com/files/releases/rubberband-3.3.0-gpl-executable-windows.zip",
            os.path.join(install_dir, "rubberband")
        )
        mock_download_unzip.assert_any_call(
            "https://github.com/FluidSynth/fluidsynth/releases/download/v2.3.5/fluidsynth-2.3.5-win64.zip",
            os.path.join(install_dir, "fluidsynth")
        )
        
        mock_add_to_path.assert_any_call(os.path.join(install_dir, "rubberband", "rubberband-dir"))
        mock_add_to_path.assert_any_call(os.path.join(install_dir, "fluidsynth", "bin"))

        mock_download_file.assert_called_once_with(
            "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/FluidR3_GM.sf2",
            os.path.join(install_dir, "soundfonts", "FluidR3_GM.sf2")
        )

    @patch('definers.get_os_name', return_value='windows')
    @patch('os.path.exists', return_value=True)
    @patch.dict(os.environ, {'PATH': 'C:\\rubberband;C:\\fluidsynth'})
    @patch('builtins.print')
    def test_install_on_windows_already_installed(self, mock_print, mock_exists, mock_get_os):
        install_audio_effects()
        mock_print.assert_any_call("\nInstalling Python packages with pip...")

    @patch('definers.get_os_name', return_value='darwin')
    @patch('builtins.print')
    def test_unsupported_os(self, mock_print, mock_get_os):
        install_audio_effects()
        mock_print.assert_any_call("Unsupported OS: darwin. Manual installation of system dependencies may be required.")

if __name__ == '__main__':
    unittest.main()
