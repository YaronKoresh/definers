import sys
import unittest
from unittest.mock import patch

import definers.system as system_module


class TestRun(unittest.TestCase):
    @patch.object(system_module, "get_infrastructure_services")
    def test_run_on_windows(self, mock_get_infrastructure_services):
        with patch.object(sys, "platform", "win32"):
            command = "dir"
            silent = True
            env = {"TEST_ENV": "123"}
            system_module.run(command, silent=silent, env=env)
            mock_get_infrastructure_services.return_value.processes.run.assert_called_once_with(
                command, silent=silent, env=env
            )

    @patch.object(system_module, "get_infrastructure_services")
    def test_run_on_windows_normalizes_missing_env(
        self, mock_get_infrastructure_services
    ):
        with patch.object(sys, "platform", "win32"):
            command = "dir"
            system_module.run(command)
            mock_get_infrastructure_services.return_value.processes.run.assert_called_once_with(
                command, silent=False, env={}
            )

    @patch.object(system_module, "get_infrastructure_services")
    def test_run_on_linux(self, mock_get_infrastructure_services):
        with patch.object(sys, "platform", "linux"):
            command = "ls"
            silent = False
            env = {"TEST_ENV": "abc"}
            system_module.run(command, silent=silent, env=env)
            mock_get_infrastructure_services.return_value.processes.run.assert_called_once_with(
                command, silent=silent, env=env
            )

    @patch.object(system_module, "get_infrastructure_services")
    def test_run_on_darwin(self, mock_get_infrastructure_services):
        with patch.object(sys, "platform", "darwin"):
            command = "ls -l"
            system_module.run(command)
            mock_get_infrastructure_services.return_value.processes.run.assert_called_once_with(
                command, silent=False, env={}
            )


if __name__ == "__main__":
    unittest.main()
