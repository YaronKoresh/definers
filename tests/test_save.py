import unittest
import os
import tempfile
import shutil
from definers import save, read

class TestSave(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.file_path = os.path.join(self.test_dir, "test_file.txt")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_to_new_file(self):
        content = "This is a test string."
        save(self.file_path, content)
        self.assertTrue(os.path.exists(self.file_path))
        read_content = read(self.file_path)
        self.assertEqual(read_content, content)

    def test_overwrite_existing_file(self):
        initial_content = "Initial content."
        with open(self.file_path, "w") as f:
            f.write(initial_content)
        
        new_content = "This content overwrites the old one."
        save(self.file_path, new_content)
        
        read_content = read(self.file_path)
        self.assertEqual(read_content, new_content)

    def test_save_empty_string(self):
        save(self.file_path, "")
        self.assertTrue(os.path.exists(self.file_path))
        read_content = read(self.file_path)
        self.assertEqual(read_content, "")

    def test_save_non_string_content(self):
        content = 12345
        save(self.file_path, content)
        read_content = read(self.file_path)
        self.assertEqual(read_content, str(content))

    def test_save_to_nested_directory(self):
        nested_dir = os.path.join(self.test_dir, "subdir1", "subdir2")
        nested_file_path = os.path.join(nested_dir, "nested_file.txt")
        content = "Content in a nested file."
        
        self.assertFalse(os.path.exists(nested_dir))
        save(nested_file_path, content)
        self.assertTrue(os.path.exists(nested_file_path))
        
        read_content = read(nested_file_path)
        self.assertEqual(read_content, content)
        
    def test_save_with_default_content(self):
        save(self.file_path)
        self.assertTrue(os.path.exists(self.file_path))
        read_content = read(self.file_path)
        self.assertEqual(read_content, "")

    def test_save_multiline_string(self):
        content = "Line 1\nLine 2\nLine 3"
        save(self.file_path, content)
        read_content = read(self.file_path)
        self.assertEqual(read_content, content)

if __name__ == '__main__':
    unittest.main()
