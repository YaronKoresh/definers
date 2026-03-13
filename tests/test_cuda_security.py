import unittest
from unittest.mock import call, patch

from definers import cuda_toolkit, free


class TestCudaRunCommands(unittest.TestCase):
    @patch("definers.run")
    def test_cuda_toolkit_runs_lists(self, mock_run):

        cuda_toolkit()

        self.assertTrue(mock_run.call_count >= 2)
        first, second = (
            mock_run.call_args_list[0][0][0],
            mock_run.call_args_list[1][0][0],
        )
        self.assertIsInstance(first, list)
        self.assertIsInstance(second, list)
        self.assertEqual(first[0], "apt-get")
        self.assertEqual(second[0], "apt-get")

    @patch("definers.run")
    def test_free_commands_are_lists(self, mock_run):

        free()
        for call_args in mock_run.call_args_list:
            arg = call_args[0][0]
            self.assertIsInstance(arg, list)


if __name__ == "__main__":
    unittest.main()
