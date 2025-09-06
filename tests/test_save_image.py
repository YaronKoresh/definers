import unittest
from unittest.mock import patch, MagicMock
import os
from definers import save_image

class TestSaveImage(unittest.TestCase):

    @patch('definers.random_string', return_value='random123')
    def test_save_image_returns_correct_path(self, mock_random_string):
        mock_img = MagicMock()
        
        result_path = save_image(mock_img, path=".")
        
        expected_path = os.path.join(".", "img_random123.png")
        self.assertEqual(result_path, expected_path)

    @patch('definers.random_string', return_value='another_random')
    def test_save_image_calls_save_method(self, mock_random_string):
        mock_img = MagicMock()
        
        save_image(mock_img, path="/tmp")
        
        expected_path = os.path.join("/tmp", "img_another_random.png")
        mock_img.save.assert_called_once_with(expected_path)

if __name__ == '__main__':
    unittest.main()
