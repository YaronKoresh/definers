import os
import site
import sys
import unittest
from unittest.mock import patch

from definers import find_package_paths


class TestFindPackagePaths(unittest.TestCase):

    @patch("site.getsitepackages")
    @patch("sys.path", new=[])
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_package_in_site_packages(
        self, mock_isdir, mock_exists, mock_getsitepackages
    ):
        mock_getsitepackages.return_value = [
            "/usr/lib/python3.8/site-packages"
        ]

        def exists_side_effect(path):
            return (
                path == "/usr/lib/python3.8/site-packages/my_package"
            )

        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True

        result = find_package_paths("my_package")
        self.assertCountEqual(
            result, ["/usr/lib/python3.8/site-packages/my_package"]
        )

    @patch("site.getsitepackages", return_value=[])
    @patch("sys.path", new=["/custom/path"])
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_package_in_sys_path(self, mock_isdir, mock_exists):

        def exists_side_effect(path):
            return path == "/custom/path/another_package"

        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True

        result = find_package_paths("another_package")
        self.assertCountEqual(
            result, ["/custom/path/another_package"]
        )

    @patch("site.getsitepackages")
    @patch("sys.path", new=[])
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_package_in_dist_packages(
        self, mock_isdir, mock_exists, mock_getsitepackages
    ):
        mock_getsitepackages.return_value = [
            "/usr/local/lib/python3.8/site-packages"
        ]

        def exists_side_effect(path):
            return (
                path
                == "/usr/local/lib/python3.8/dist-packages/dist_package"
            )

        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True

        result = find_package_paths("dist_package")
        self.assertCountEqual(
            result,
            ["/usr/local/lib/python3.8/dist-packages/dist_package"],
        )

    @patch("site.getsitepackages", return_value=[])
    @patch("sys.path", new=[])
    @patch("os.path.exists", return_value=False)
    @patch("os.path.isdir", return_value=False)
    def test_package_not_found(self, mock_isdir, mock_exists):
        result = find_package_paths("nonexistent_package")
        self.assertEqual(result, [])

    @patch("site.getsitepackages")
    @patch("sys.path")
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_package_in_multiple_locations(
        self,
        mock_isdir,
        mock_exists,
        mock_sys_path,
        mock_getsitepackages,
    ):
        mock_getsitepackages.return_value = ["/lib/site-packages"]
        mock_sys_path.__iter__.return_value = ["/usr/custom/lib"]

        def exists_side_effect(path):
            return path in [
                "/lib/site-packages/multi_location",
                "/usr/custom/lib/multi_location",
            ]

        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True

        result = find_package_paths("multi_location")
        self.assertCountEqual(
            result,
            [
                "/lib/site-packages/multi_location",
                "/usr/custom/lib/multi_location",
            ],
        )

    @patch("site.getsitepackages")
    @patch("sys.path")
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_duplicate_paths_are_unique(
        self,
        mock_isdir,
        mock_exists,
        mock_sys_path,
        mock_getsitepackages,
    ):
        mock_getsitepackages.return_value = ["/lib/site-packages"]
        mock_sys_path.__iter__.return_value = ["/lib/site-packages"]

        def exists_side_effect(path):
            return path == "/lib/site-packages/duplicate_package"

        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True

        result = find_package_paths("duplicate_package")
        self.assertCountEqual(
            result, ["/lib/site-packages/duplicate_package"]
        )
        self.assertEqual(len(result), 1)

    @patch("site.getsitepackages")
    @patch("sys.path", new=[])
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_package_name_with_hyphen(
        self, mock_isdir, mock_exists, mock_getsitepackages
    ):
        mock_getsitepackages.return_value = ["/lib/site-packages"]

        def exists_side_effect(path):
            return path == "/lib/site-packages/a_package_with_hyphen"

        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True

        result = find_package_paths("a-package-with-hyphen")
        self.assertCountEqual(
            result, ["/lib/site-packages/a_package_with_hyphen"]
        )


if __name__ == "__main__":
    unittest.main()
