import os
import site
import sys
import unittest
from unittest.mock import patch

from definers.system import add_path
from definers.system.services import (
    EnvironmentService,
    FileSystemService,
    InfrastructureServices,
    reset_infrastructure_services,
    set_infrastructure_services,
)


def _norm(p):
    return os.path.abspath(os.path.expanduser(p))


class TestAddPath(unittest.TestCase):
    def tearDown(self):
        reset_infrastructure_services()

    @patch("site.addsitedir")
    def test_add_new_path(self, mock_addsitedir):
        raw_path = "/new/test/path"
        expected_path = _norm(raw_path)
        permit_calls = []
        original_sys_path = sys.path[:]
        set_infrastructure_services(
            InfrastructureServices(
                environment=EnvironmentService(
                    get_os_name_fn=lambda: "unknown"
                ),
                filesystem=FileSystemService(
                    permit_fn=lambda path, **kwargs: (
                        permit_calls.append((path, kwargs)) or True
                    )
                ),
            )
        )
        try:
            sys.path = [p for p in original_sys_path if p != expected_path]
            add_path(raw_path)
            self.assertEqual(len(permit_calls), 1)
            self.assertEqual(permit_calls[0][0], expected_path)
            self.assertIn(expected_path, sys.path)
            mock_addsitedir.assert_called_once_with(expected_path)
        finally:
            sys.path = original_sys_path

    @patch("site.addsitedir")
    def test_add_existing_path(self, mock_addsitedir):
        raw_path = "/existing/test/path"
        expected_path = _norm(raw_path)
        permit_calls = []
        original_sys_path = sys.path[:]
        set_infrastructure_services(
            InfrastructureServices(
                environment=EnvironmentService(
                    get_os_name_fn=lambda: "unknown"
                ),
                filesystem=FileSystemService(
                    permit_fn=lambda path, **kwargs: (
                        permit_calls.append((path, kwargs)) or True
                    )
                ),
            )
        )
        try:
            if expected_path not in sys.path:
                sys.path.append(expected_path)
            initial_path_length = len(sys.path)
            add_path(raw_path)
            self.assertEqual(permit_calls, [])
            mock_addsitedir.assert_not_called()
            self.assertEqual(len(sys.path), initial_path_length)
        finally:
            sys.path = original_sys_path

    @patch("site.addsitedir")
    def test_add_empty_path(self, mock_addsitedir):
        test_path = ""
        permit_calls = []
        original_sys_path = sys.path[:]
        set_infrastructure_services(
            InfrastructureServices(
                environment=EnvironmentService(
                    get_os_name_fn=lambda: "unknown"
                ),
                filesystem=FileSystemService(
                    permit_fn=lambda path, **kwargs: (
                        permit_calls.append((path, kwargs)) or True
                    )
                ),
            )
        )
        try:
            sys.path = [p for p in original_sys_path if p != test_path]
            add_path(test_path)
            self.assertEqual(len(permit_calls), 1)
            self.assertEqual(permit_calls[0][0], test_path)
            self.assertIn(test_path, sys.path)
            mock_addsitedir.assert_called_once_with(test_path)
        finally:
            sys.path = original_sys_path


if __name__ == "__main__":
    unittest.main()
