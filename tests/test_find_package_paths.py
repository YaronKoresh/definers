import os
import sys
import unittest
from unittest.mock import patch
from definers import find_package_paths


class TestFindPackagePaths(unittest.TestCase):
    @patch("definers.site.getsitepackages")
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_package_not_found(
        self, mock_isdir, mock_exists, mock_getsitepackages
    ):
        mock_getsitepackages.return_value = []
        sys.path.append("/fake/sys/path")
        mock_exists.return_value = False
        mock_isdir.return_value = False
        paths = find_package_paths("nonexistent_package")
        self.assertEqual(paths, [])
        sys.path.remove("/fake/sys/path")

    @patch("definers.site.getsitepackages")
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_package_in_sys_path(
        self, mock_isdir, mock_exists, mock_getsitepackages
    ):
        mock_getsitepackages.return_value = []
        package_name = "my_package"
        fake_base = "/fake/sys/path"
        fake_path = os.path.join(fake_base, package_name)
        original_sys_path = sys.path[:]
        sys.path.append(fake_base)

        def exists_side_effect(path):
            return path == fake_path

        def isdir_side_effect(path):
            return path == fake_path

        mock_exists.side_effect = exists_side_effect
        mock_isdir.side_effect = isdir_side_effect
        paths = find_package_paths(package_name)
        self.assertIn(fake_path, paths)
        sys.path[:] = original_sys_path

    @patch("definers.site.getsitepackages")
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_package_in_site_packages(
        self, mock_isdir, mock_exists, mock_getsitepackages
    ):
        package_name = "my_package"
        site_base = "/fake/site-packages"
        site_packages_path = os.path.join(site_base, package_name)
        mock_getsitepackages.return_value = [site_base]
        mock_exists.return_value = True
        mock_isdir.return_value = True
        paths = find_package_paths(package_name)
        self.assertIn(site_packages_path, paths)

    @patch("definers.site.getsitepackages")
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_package_in_dist_packages(
        self, mock_isdir, mock_exists, mock_getsitepackages
    ):
        package_name = "my_package"
        dist_base = "/fake/dist-packages"
        dist_packages_path = os.path.join(dist_base, package_name)
        mock_getsitepackages.return_value = ["/fake/site-packages"]

        def exists_side_effect(path):
            return path == dist_packages_path

        def isdir_side_effect(path):
            return path == dist_packages_path

        mock_exists.side_effect = exists_side_effect
        mock_isdir.side_effect = isdir_side_effect
        paths = find_package_paths(package_name)
        self.assertIn(dist_packages_path, paths)


if __name__ == "__main__":
    unittest.main()
