import os
import shutil
import tempfile
import unittest
from pathlib import Path

from definers import move


class TestMove(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.src_path = os.path.join(self.test_dir, "source")
        self.dest_path = os.path.join(self.test_dir, "destination")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_move_file(self):
        os.makedirs(self.src_path)
        src_file = os.path.join(self.src_path, "test_file.txt")
        with open(src_file, "w") as f:
            f.write("test content")

        dest_file = os.path.join(self.dest_path, "test_file.txt")

        self.assertFalse(os.path.exists(self.dest_path))
        os.makedirs(self.dest_path)

        move(src_file, dest_file)

        self.assertFalse(os.path.exists(src_file))
        self.assertTrue(os.path.exists(dest_file))
        with open(dest_file, "r") as f:
            self.assertEqual(f.read(), "test content")

    def test_move_directory(self):
        os.makedirs(self.src_path)
        with open(os.path.join(self.src_path, "file.txt"), "w") as f:
            f.write("hello")

        self.assertTrue(os.path.isdir(self.src_path))
        self.assertFalse(os.path.exists(self.dest_path))

        move(self.src_path, self.dest_path)

        self.assertFalse(os.path.exists(self.src_path))
        self.assertTrue(os.path.isdir(self.dest_path))
        self.assertTrue(
            os.path.exists(os.path.join(self.dest_path, "file.txt"))
        )

    def test_move_non_existent_source(self):
        with self.assertRaises(FileNotFoundError):
            move(
                os.path.join(self.test_dir, "non_existent"),
                self.dest_path,
            )

    def test_move_file_to_existing_destination_file(self):
        os.makedirs(self.src_path)
        os.makedirs(self.dest_path)
        src_file = os.path.join(self.src_path, "src.txt")
        dest_file = os.path.join(self.dest_path, "dest.txt")

        with open(src_file, "w") as f:
            f.write("source")
        with open(dest_file, "w") as f:
            f.write("destination")

        move(src_file, dest_file)

        self.assertFalse(os.path.exists(src_file))
        self.assertTrue(os.path.exists(dest_file))
        with open(dest_file, "r") as f:
            self.assertEqual(f.read(), "source")

    def test_move_directory_to_existing_non_empty_directory(self):
        os.makedirs(self.src_path)
        os.makedirs(self.dest_path)
        with open(os.path.join(self.src_path, "file1.txt"), "w") as f:
            f.write("1")
        with open(
            os.path.join(self.dest_path, "file2.txt"), "w"
        ) as f:
            f.write("2")

        dest_for_move = os.path.join(self.dest_path, "source_moved")
        move(self.src_path, dest_for_move)

        self.assertFalse(os.path.exists(self.src_path))
        self.assertTrue(os.path.isdir(dest_for_move))
        self.assertTrue(
            os.path.exists(os.path.join(dest_for_move, "file1.txt"))
        )


if __name__ == "__main__":
    unittest.main()
