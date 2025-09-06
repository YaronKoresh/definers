import unittest
from unittest.mock import call, patch

from definers import free


class TestFree(unittest.TestCase):

    @patch("definers.run")
    @patch("torch.cuda.empty_cache")
    @patch("os.path.exists", return_value=True)
    @patch(
        "os.path.expanduser",
        return_value="/home/user/miniconda3/bin/mamba",
    )
    def test_free_all_commands_run(
        self, mock_expanduser, mock_exists, mock_empty_cache, mock_run
    ):
        free()

        mock_empty_cache.assert_called_once()

        expected_run_calls = [
            call("rm -rf ~/.cache/huggingface/*", silent=True),
            call("rm -rf /data-nvme/zerogpu-offload/*", silent=True),
            call("rm -rf /opt/ml/checkpoints/*", silent=True),
            call("pip cache purge", silent=True),
            call(
                "/home/user/miniconda3/bin/mamba clean --all",
                silent=True,
            ),
        ]
        mock_run.assert_has_calls(expected_run_calls, any_order=True)

    @patch("definers.run")
    @patch(
        "torch.cuda.empty_cache",
        side_effect=Exception("Test Exception"),
    )
    @patch("os.path.exists", return_value=False)
    def test_free_handles_torch_exception(
        self, mock_exists, mock_empty_cache, mock_run
    ):
        with patch("definers.catch") as mock_catch:
            free()
            mock_empty_cache.assert_called_once()
            mock_catch.assert_called_once_with(
                mock_empty_cache.side_effect
            )
            self.assertIn(
                call("pip cache purge", silent=True),
                mock_run.call_args_list,
            )

    @patch("definers.run")
    @patch("torch.cuda.empty_cache")
    @patch("os.path.exists", return_value=False)
    def test_mamba_not_called_if_not_exists(
        self, mock_exists, mock_empty_cache, mock_run
    ):
        free()
        mamba_call = call(
            "/home/user/miniconda3/bin/mamba clean --all", silent=True
        )
        self.assertNotIn(mamba_call, mock_run.call_args_list)


if __name__ == "__main__":
    unittest.main()
