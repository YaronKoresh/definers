import unittest
import os
import tempfile
import shutil
from definers import directory

class TestDirectory(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_create_new_directory(self):
        new_dir_path = os.path.join(self.test_dir, "new_dir")
        self.assertFalse(os.path.exists(new_dir_path))
        directory(new_dir_path)
        self.assertTrue(os.path.isdir(new_dir_path))

    def test_directory_already_exists(self):
        existing_dir_path = os.path.join(self.test_dir, "existing_dir")
        os.makedirs(existing_dir_path)
        self.assertTrue(os.path.isdir(existing_dir_path))
        directory(existing_dir_path)
        self.assertTrue(os.path.isdir(existing_dir_path))

    def test_create_nested_directories(self):
        nested_dir_path = os.path.join(self.test_dir, "parent", "child")
        self.assertFalse(os.path.exists(nested_dir_path))
        directory(nested_dir_path)
        self.assertTrue(os.path.isdir(nested_dir_path))

    def test_path_is_a_file(self):
        file_path = os.path.join(self.test_dir, "a_file.txt")
        with open(file_path, "w") as f:
            f.write("hello")
        
        with self.assertRaises(FileExistsError):
            directory(file_path)

    def test_path_with_special_characters(self):
        special_dir_path = os.path.join(self.test_dir, "sp@ci&l-ch^rs")
        directory(special_dir_path)
        self.assertTrue(os.path.isdir(special_dir_path))

if __name__ == '__main__':
    unittest.main()
