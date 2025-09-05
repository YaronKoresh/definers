import unittest
import os
import tempfile
import shutil
from definers import read

class TestRead(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.file_path = os.path.join(self.test_dir, "test_file.txt")
        self.binary_file_path = os.path.join(self.test_dir, "test_binary.dat")
        self.dir_path = os.path.join(self.test_dir, "test_dir")
        os.makedirs(self.dir_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_read_text_file(self):
        content = "hello world\nthis is a test"
        with open(self.file_path, "w", encoding="utf8") as f:
            f.write(content)
        read_content = read(self.file_path)
        self.assertEqual(read_content, content)

    def test_read_binary_file(self):
        content = b'\x01\x02\x03\x04\x05'
        with open(self.binary_file_path, "wb") as f:
            f.write(content)
        read_content = read(self.binary_file_path)
        self.assertEqual(read_content, content)

    def test_read_directory(self):
        file1 = os.path.join(self.dir_path, "file1.txt")
        file2 = os.path.join(self.dir_path, "file2.log")
        with open(file1, "w") as f:
            f.write("1")
        with open(file2, "w") as f:
            f.write("2")
        dir_content = read(self.dir_path)
        self.assertIsInstance(dir_content, list)
        self.assertIn("file1.txt", dir_content)
        self.assertIn("file2.log", dir_content)
        self.assertEqual(len(dir_content), 2)

    def test_read_non_existent_file(self):
        non_existent_path = os.path.join(self.test_dir, "non_existent.txt")
        self.assertIsNone(read(non_existent_path))

    def test_read_empty_file(self):
        with open(self.file_path, "w") as f:
            pass
        self.assertEqual(read(self.file_path), "")

    def test_read_empty_directory(self):
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)
        self.assertEqual(read(empty_dir), [])

    def test_read_utf8_file(self):
        content = "こんにちは世界"
        with open(self.file_path, "w", encoding="utf8") as f:
            f.write(content)
        read_content = read(self.file_path)
        self.assertEqual(read_content, content)

if __name__ == '__main__':
    unittest.main()
