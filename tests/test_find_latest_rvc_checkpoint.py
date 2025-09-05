import unittest
from unittest.mock import patch, MagicMock
from definers import find_latest_rvc_checkpoint

class TestFindLatestRvcCheckpoint(unittest.TestCase):

    @patch('os.path.isdir', return_value=False)
    @patch('definers.logger')
    def test_folder_not_found(self, mock_logger, mock_isdir):
        result = find_latest_rvc_checkpoint("/non/existent/path", "my_model")
        self.assertIsNone(result)
        mock_isdir.assert_called_once_with("/non/existent/path")
        mock_logger.error.assert_called_once_with("Error: Folder not found at /non/existent/path")

    @patch('os.path.isdir', return_value=True)
    @patch('os.listdir', return_value=[
        "my_model_e10_s5000.pth",
        "my_model_e20_s10000.pth",
        "my_model_e20_s15000.pth",
        "not_a_model_file.txt"
    ])
    @patch('definers.logger')
    def test_finds_latest_checkpoint_by_step(self, mock_logger, mock_listdir, mock_isdir):
        result = find_latest_rvc_checkpoint("/fake/path", "my_model")
        self.assertEqual(result, "my_model_e20_s15000.pth")
        mock_logger.info.assert_any_call("Latest checkpoint found: my_model_e20_s15000.pth")

    @patch('os.path.isdir', return_value=True)
    @patch('os.listdir', return_value=[
        "my_model_e30_s5000.pth",
        "my_model_e20_s15000.pth",
        "my_model_e10_s20000.pth"
    ])
    @patch('definers.logger')
    def test_finds_latest_checkpoint_by_epoch(self, mock_logger, mock_listdir, mock_isdir):
        result = find_latest_rvc_checkpoint("/fake/path", "my_model")
        self.assertEqual(result, "my_model_e30_s5000.pth")
        mock_logger.info.assert_any_call("Latest checkpoint found: my_model_e30_s5000.pth")

    @patch('os.path.isdir', return_value=True)
    @patch('os.listdir', return_value=["other_model_e1_s1.pth", "random_file.txt"])
    @patch('definers.logger')
    def test_no_matching_checkpoints(self, mock_logger, mock_listdir, mock_isdir):
        result = find_latest_rvc_checkpoint("/fake/path", "my_model")
        self.assertIsNone(result)
        mock_logger.warning.assert_called_once_with("No checkpoint found matching the pattern in '/fake/path'")

    @patch('os.path.isdir', return_value=True)
    @patch('os.listdir', return_value=[])
    @patch('definers.logger')
    def test_empty_directory(self, mock_logger, mock_listdir, mock_isdir):
        result = find_latest_rvc_checkpoint("/fake/path", "my_model")
        self.assertIsNone(result)
        mock_logger.warning.assert_called_once_with("No checkpoint found matching the pattern in '/fake/path'")

    @patch('os.path.isdir', return_value=True)
    @patch('os.listdir', side_effect=PermissionError("Access denied"))
    @patch('definers.logger')
    def test_os_listdir_raises_exception(self, mock_logger, mock_listdir, mock_isdir):
        result = find_latest_rvc_checkpoint("/fake/path", "my_model")
        self.assertIsNone(result)
        mock_logger.error.assert_called_once()
        self.assertIn("An error occurred while scanning the folder for checkpoints", mock_logger.error.call_args[0][0])

if __name__ == '__main__':
    unittest.main()
