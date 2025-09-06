import unittest
from unittest.mock import patch
from definers import google_drive_download

class TestGoogleDriveDownload(unittest.TestCase):

    @patch('definers.download_file_from_google_drive')
    def test_google_drive_download_success(self, mock_download):
        file_id = "test_id"
        dest_path = "/tmp/test_file.zip"
        
        google_drive_download(file_id, dest_path)
        
        mock_download.assert_called_once_with(
            file_id=file_id, 
            dest_path=dest_path, 
            unzip=True, 
            showsize=False
        )

    @patch('definers.download_file_from_google_drive', side_effect=Exception("Download failed"))
    def test_google_drive_download_failure(self, mock_download):
        file_id = "fail_id"
        dest_path = "/tmp/fail_file.zip"
        
        with self.assertRaises(Exception) as context:
            google_drive_download(file_id, dest_path)
            
        self.assertTrue("Download failed" in str(context.exception))
        mock_download.assert_called_once_with(
            file_id=file_id, 
            dest_path=dest_path, 
            unzip=True, 
            showsize=False
        )

if __name__ == '__main__':
    unittest.main()
