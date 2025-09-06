import os
import shutil
import tempfile
import unittest
from pathlib import Path

from definers import is_package_path


class TestIsPackagePath(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_path_does_not_exist(self):
        non_existent_path = os.path.join(
            self.test_dir, "non_existent"
        )
        self.assertFalse(is_package_path(non_existent_path))

    def test_path_is_file(self):
        file_path = os.path.join(self.test_dir, "a_file.txt")
        Path(file_path).touch()
        self.assertFalse(is_package_path(file_path))

    def test_directory_not_a_package(self):
        empty_dir = os.path.join(self.test_dir, "empty_dir")
        os.makedirs(empty_dir)
        self.assertFalse(is_package_path(empty_dir))

    def test_package_with_init_py(self):
        package_dir = os.path.join(self.test_dir, "package_a")
        os.makedirs(package_dir)
        Path(os.path.join(package_dir, "__init__.py")).touch()
        self.assertTrue(is_package_path(package_dir))

    def test_package_with_matching_name_file(self):
        package_dir = os.path.join(self.test_dir, "package_b")
        os.makedirs(package_dir)
        Path(os.path.join(package_dir, "package_b")).touch()
        self.assertTrue(is_package_path(package_dir))

    def test_package_with_src_directory(self):
        package_dir = os.path.join(self.test_dir, "package_c")
        os.makedirs(package_dir)
        os.makedirs(os.path.join(package_dir, "src"))
        self.assertTrue(is_package_path(package_dir))

    def test_package_with_name_match(self):
        package_dir = os.path.join(self.test_dir, "my_package")
        os.makedirs(package_dir)
        Path(os.path.join(package_dir, "__init__.py")).touch()
        self.assertTrue(
            is_package_path(package_dir, package_name="my_package")
        )

    def test_package_with_name_mismatch(self):
        package_dir = os.path.join(self.test_dir, "my_package")
        os.makedirs(package_dir)
        Path(os.path.join(package_dir, "__init__.py")).touch()
        self.assertFalse(
            is_package_path(
                package_dir, package_name="another_package"
            )
        )

    def test_package_with_name_but_not_package(self):
        package_dir = os.path.join(self.test_dir, "not_a_package")
        os.makedirs(package_dir)
        self.assertFalse(
            is_package_path(package_dir, package_name="not_a_package")
        )


if __name__ == "__main__":
    unittest.main()
