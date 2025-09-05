import unittest
import os
import tempfile
import shutil
from pathlib import Path
from definers import delete, exist

class TestDelete(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_delete_file(self):
        file_path = os.path.join(self.test_dir, "test_file.txt")
        with open(file_path, "w") as f:
            f.write("delete me")
        self.assertTrue(exist(file_path))
        delete(file_path)
        self.assertFalse(exist(file_path))

    def test_delete_directory(self):
        dir_to_delete = os.path.join(self.test_dir, "subdir")
        os.makedirs(dir_to_delete)
        with open(os.path.join(dir_to_delete, "file.txt"), "w") as f:
            f.write("content")
        self.assertTrue(exist(dir_to_delete))
        delete(dir_to_delete)
        self.assertFalse(exist(dir_to_delete))

    def test_delete_non_existent_path(self):
        non_existent_path = os.path.join(self.test_dir, "does_not_exist")
        self.assertFalse(exist(non_existent_path))
        try:
            delete(non_existent_path)
        except Exception as e:
            self.fail(f"delete() raised an exception for a non-existent path: {e}")
        self.assertFalse(exist(non_existent_path))

    def test_delete_empty_directory(self):
        empty_dir = os.path.join(self.test_dir, "empty_dir")
        os.makedirs(empty_dir)
        self.assertTrue(exist(empty_dir))
        delete(empty_dir)
        self.assertFalse(exist(empty_dir))

    def test_delete_symbolic_link_to_file(self):
        target_file = os.path.join(self.test_dir, "target.txt")
        link_path = os.path.join(self.test_dir, "link.txt")
        with open(target_file, "w") as f:
            f.write("target")
        
        if hasattr(os, 'symlink'):
            os.symlink(target_file, link_path)
            self.assertTrue(os.path.islink(link_path))
            self.assertTrue(exist(target_file))
            
            delete(link_path)
            
            self.assertFalse(exist(link_path))
            self.assertTrue(exist(target_file))

    def test_delete_symbolic_link_to_directory(self):
        target_dir = os.path.join(self.test_dir, "target_dir")
        os.makedirs(target_dir)
        link_path = os.path.join(self.test_dir, "link_dir")
        
        if hasattr(os, 'symlink'):
            os.symlink(target_dir, link_path, target_is_directory=True)
            self.assertTrue(os.path.islink(link_path))
            self.assertTrue(exist(target_dir))
            
            delete(link_path)
            
            self.assertFalse(exist(link_path))
            self.assertTrue(exist(target_dir))

if __name__ == '__main__':
    unittest.main()
