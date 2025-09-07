import unittest
from unittest.mock import patch
from definers import google_drive_download

class TestGoogleDriveDownload(unittest.TestCase):
    @patch("googledrivedownloader.download_file_from_google_drive")
    def test_google_drive_download_success(
        self, mock_download
    ):
        file_id = "some_file_id"
        dest_path = "/fake/path/file.zip"
        google_drive_download(file_id, dest_path)
        mock_download.assert_called_once_with(
            file_id=file_id, dest_path=dest_path, unzip=True, showsize=False
        )

    @patch(
        "googledrivedownloader.download_file_from_google_drive",
        side_effect=Exception("Download failed"),
    )
    def test_google_drive_download_failure(
        self, mock_download
    ):
        file_id = "some_file_id"
        dest_path = "/fake/path/file.zip"
        try:
            google_drive_download(file_id, dest_path)
        except Exception:
            self.fail(
                "google_drive_download raised an exception instead of handling it."
            )
        mock_download.assert_called_once()

if __name__ == "__main__":
    unittest.main()

