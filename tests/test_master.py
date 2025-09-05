import unittest
from unittest.mock import patch, MagicMock, call
from definers import master

class TestMaster(unittest.TestCase):

    @patch('definers.Path')
    @patch('definers.tempfile.TemporaryDirectory')
    @patch('definers.google_drive_download')
    @patch('definers.mg.process')
    @patch('definers.pydub.AudioSegment')
    @patch('definers.export_audio', return_value="/path/to/mastered.mp3")
    @patch('definers.delete')
    @patch('definers.tmp', side_effect=["/tmp/result1.wav", "/tmp/result2.wav"])
    @patch('definers.catch')
    def test_successful_mastering_with_strength_gt_1(self, mock_catch, mock_tmp, mock_delete, mock_export, mock_pydub, mock_mg_process, mock_gdd, mock_tempdir, mock_path):
        
        mock_tempdir.return_value.__enter__.return_value = "/tmp/tempdir"
        
        mock_audio_segment = MagicMock()
        mock_audio_segment.__add__.return_value = mock_audio_segment
        mock_pydub.from_file.return_value = mock_audio_segment
        
        mock_path_instance = MagicMock()
        mock_path_instance.with_name.return_value = "/path/to/source_mastered"
        mock_path.return_value = mock_path_instance

        result = master("/path/to/source.wav", 2.5, "mp3")
        
        mock_gdd.assert_called_once()
        self.assertEqual(mock_mg_process.call_count, 2)
        
        mock_pydub.from_file.assert_called_once_with("/tmp/result2.wav")
        mock_audio_segment.__add__.assert_called_once_with(9.0)
        
        mock_export.assert_called_once_with(mock_audio_segment, "/path/to/source_mastered", "mp3")
        
        self.assertEqual(mock_delete.call_count, 1)
        self.assertEqual(result, "/path/to/mastered.mp3")
        mock_catch.assert_not_called()

    @patch('definers.Path')
    @patch('definers.tempfile.TemporaryDirectory')
    @patch('definers.google_drive_download')
    @patch('definers.mg.process')
    @patch('definers.pydub.AudioSegment')
    @patch('definers.export_audio')
    @patch('definers.delete')
    @patch('definers.tmp')
    @patch('definers.catch')
    def test_mastering_with_strength_lt_1(self, mock_catch, mock_tmp, mock_delete, mock_export, mock_pydub, mock_mg_process, mock_gdd, mock_tempdir, mock_path):
        mock_tempdir.return_value.__enter__.return_value = "/tmp/tempdir"
        mock_audio_segment = MagicMock()
        mock_audio_segment.__add__.return_value = mock_audio_segment
        mock_pydub.from_file.return_value = mock_audio_segment
        
        master("/path/to/source.wav", 0.8, "wav")
        
        mock_mg_process.assert_not_called()
        mock_pydub.from_file.assert_called_once_with("/path/to/source.wav")
        mock_audio_segment.__add__.assert_called_once_with(-1.2)
        
    @patch('definers.google_drive_download', side_effect=Exception("Download failed"))
    @patch('definers.catch')
    def test_gdd_failure(self, mock_catch, mock_gdd):
        result = master("source.wav", 1, "mp3")
        self.assertIsNone(result)
        mock_catch.assert_called_once()
        self.assertIsInstance(mock_catch.call_args[0][0], Exception)

    @patch('definers.tempfile.TemporaryDirectory')
    @patch('definers.google_drive_download')
    @patch('definers.mg.process', side_effect=Exception("Mastering failed"))
    @patch('definers.catch')
    @patch('definers.tmp')
    def test_mg_process_failure(self, mock_tmp, mock_catch, mock_mg_process, mock_gdd, mock_tempdir):
        mock_tempdir.return_value.__enter__.return_value = "/tmp/tempdir"
        result = master("source.wav", 1.5, "mp3")
        self.assertIsNone(result)
        mock_catch.assert_called_once()
        self.assertIsInstance(mock_catch.call_args[0][0], Exception)
        
if __name__ == '__main__':
    unittest.main()
