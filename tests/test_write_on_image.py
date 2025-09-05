import unittest
from unittest.mock import patch, MagicMock, call
from PIL import Image, ImageDraw, ImageFont
from definers import write_on_image

class TestWriteOnImage(unittest.TestCase):

    @patch('definers.save_image')
    @patch('definers.google_drive_download')
    @patch('definers.read')
    @patch('PIL.Image.open')
    @patch('PIL.ImageDraw.Draw')
    @patch('PIL.ImageFont.truetype')
    def test_all_titles_font_download(self, mock_truetype, mock_draw, mock_open, mock_read, mock_download, mock_save):

        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_open.return_value = mock_img
        
        mock_draw_instance = MagicMock()
        mock_draw_instance.textlength.return_value = 200
        mock_draw.return_value = mock_draw_instance
        
        mock_font = MagicMock()
        mock_truetype.return_value = mock_font
        
        mock_read.return_value = ""
        
        mock_save.return_value = "output_path.png"

        image_path = "input.png"
        top = "Top Title"
        middle = "Middle Title"
        bottom = "Bottom Title"

        result = write_on_image(image_path, top_title=top, middle_title=middle, bottom_title=bottom)

        mock_open.assert_called_once_with(image_path)
        mock_read.assert_called_with(".")
        mock_download.assert_called_once_with("1C48KkYWQDYu7ypbNtSXAUJ6kuzoZ42sI", "./Alef-Bold.ttf")
        
        self.assertEqual(mock_draw_instance.text.call_count, 3)
        
        args_list = [c[0] for c in mock_draw_instance.text.call_args_list]
        self.assertTrue(any(top in arg for arg in args_list))
        self.assertTrue(any(middle in arg for arg in args_list))
        self.assertTrue(any(bottom in arg for arg in args_list))

        mock_save.assert_called_once_with(mock_img)
        self.assertEqual(result, "output_path.png")

    @patch('definers.save_image')
    @patch('definers.google_drive_download')
    @patch('definers.read')
    @patch('PIL.Image.open')
    @patch('PIL.ImageDraw.Draw')
    @patch('PIL.ImageFont.truetype')
    def test_some_titles_font_exists(self, mock_truetype, mock_draw, mock_open, mock_read, mock_download, mock_save):
        
        mock_img = MagicMock()
        mock_img.size = (1024, 768)
        mock_open.return_value = mock_img
        
        mock_draw_instance = MagicMock()
        mock_draw_instance.textlength.return_value = 300
        mock_draw.return_value = mock_draw_instance

        mock_read.return_value = ["Alef-Bold.ttf"]

        top = "Hello"
        bottom = "World"
        
        write_on_image("another_input.jpg", top_title=top, bottom_title=bottom)

        mock_download.assert_not_called()
        self.assertEqual(mock_draw_instance.text.call_count, 2)
        
        args_list = [c[0] for c in mock_draw_instance.text.call_args_list]
        self.assertTrue(any(top in arg for arg in args_list))
        self.assertTrue(any(bottom in arg for arg in args_list))


    @patch('definers.save_image')
    @patch('definers.google_drive_download')
    @patch('definers.read')
    @patch('PIL.Image.open')
    @patch('PIL.ImageDraw.Draw')
    @patch('PIL.ImageFont.truetype')
    def test_no_titles(self, mock_truetype, mock_draw, mock_open, mock_read, mock_download, mock_save):
        mock_img = MagicMock()
        mock_img.size = (500, 500)
        mock_open.return_value = mock_img
        
        mock_draw_instance = MagicMock()
        mock_draw.return_value = mock_draw_instance
        
        write_on_image("no_text.png")

        mock_draw_instance.text.assert_not_called()
        mock_save.assert_called_once_with(mock_img)

    @patch('definers.save_image')
    @patch('definers.google_drive_download')
    @patch('definers.read')
    @patch('PIL.Image.open')
    @patch('PIL.ImageDraw.Draw')
    @patch('PIL.ImageFont.truetype')
    def test_multiline_text(self, mock_truetype, mock_draw, mock_open, mock_read, mock_download, mock_save):
        mock_img = MagicMock()
        mock_img.size = (1200, 800)
        mock_open.return_value = mock_img
        
        mock_draw_instance = MagicMock()
        mock_draw_instance.textlength.return_value = 400
        mock_draw.return_value = mock_draw_instance
        
        mock_read.return_value = ["Alef-Bold.ttf"]
        
        middle_text = "This is a\nmultiline message."
        
        write_on_image("multiline.gif", middle_title=middle_text)

        mock_draw_instance.text.assert_called_once()
        self.assertIn(middle_text, mock_draw_instance.text.call_args[0])

if __name__ == '__main__':
    unittest.main()
