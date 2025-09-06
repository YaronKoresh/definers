import os
import tempfile
import unittest

from definers import read, save_temp_text


class TestSaveTempText(unittest.TestCase):

    def setUp(self):
        self.temp_files = []

    def tearDown(self):
        for f in self.temp_files:
            if os.path.exists(f):
                os.remove(f)

    def test_save_temp_text_with_content(self):
        content = "This is some temporary text."
        temp_path = save_temp_text(content)
        self.assertIsNotNone(temp_path)
        self.temp_files.append(temp_path)
        self.assertTrue(os.path.exists(temp_path))
        read_content = read(temp_path)
        self.assertEqual(read_content, content)

    def test_save_temp_text_empty_string(self):
        content = ""
        temp_path = save_temp_text(content)
        self.assertIsNotNone(temp_path)
        self.temp_files.append(temp_path)
        self.assertTrue(os.path.exists(temp_path))
        read_content = read(temp_path)
        self.assertEqual(read_content, content)

    def test_save_temp_text_none_input(self):
        temp_path = save_temp_text(None)
        self.assertIsNone(temp_path)

    def test_save_temp_text_falsey_string_input(self):
        content = ""
        temp_path = save_temp_text(content)
        self.assertIsNotNone(temp_path)
        self.temp_files.append(temp_path)

    def test_file_is_temporary(self):
        content = "some content"
        temp_path = save_temp_text(content)
        self.assertIsNotNone(temp_path)
        self.temp_files.append(temp_path)
        self.assertTrue(
            os.path.basename(temp_path).startswith(
                tempfile.gettempprefix()
            )
        )

    def test_file_has_correct_extension(self):
        content = "more content"
        temp_path = save_temp_text(content)
        self.assertIsNotNone(temp_path)
        self.temp_files.append(temp_path)
        self.assertTrue(temp_path.endswith(".data"))


if __name__ == "__main__":
    unittest.main()
