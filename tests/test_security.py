import os
import unittest
from unittest.mock import patch

import definers.application_data.tokenization as tokenization_module
from definers.constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH
from definers.media.web_transfer import download_and_unzip, download_file
from definers.ml import AutoTrainer
from definers.system import run, secure_path


class TestSecurity(unittest.TestCase):
    def test_run_rejects_injection(self):

        self.assertFalse(run("echo hello; rm -rf /"))
        self.assertFalse(run("echo $HOME"))

    def test_secure_path(self):

        with self.assertRaises(ValueError):
            secure_path("/etc/passwd")

        tmpfile = "temp_test_file.tmp"
        open(tmpfile, "w").close()
        try:
            result = secure_path(tmpfile)
            self.assertTrue(result.endswith(tmpfile))
        finally:
            os.remove(tmpfile)

        tmpfile2 = "temp_test_file2.tmp"
        open(tmpfile2, "w").close()
        try:
            result2 = secure_path(tmpfile2, os.path.abspath("some/nonexistent"))
            self.assertTrue(result2.endswith(tmpfile2))
        finally:
            os.remove(tmpfile2)

    def test_train_selected_rows_validation(self):
        trainer = AutoTrainer()

        with patch.object(
            tokenization_module, "init_tokenizer"
        ) as mock_init_tokenizer:
            with self.assertRaises(ValueError):
                trainer.train(select="x" * (MAX_INPUT_LENGTH + 1))
            with self.assertRaises(ValueError):
                trainer.train(
                    select="1 " + " " * (MAX_CONSECUTIVE_SPACES + 2) + "2"
                )
            with self.assertRaises(ValueError):
                trainer.train(data="http://" + "a" * (MAX_INPUT_LENGTH + 1))
            mock_init_tokenizer.assert_not_called()

    def test_download_validation(self):
        with self.assertRaises(ValueError):
            download_file("ftp://example.com/file", "dest")
        with self.assertRaises(ValueError):
            download_file("http://" + "a" * (MAX_INPUT_LENGTH + 1), "dest")
        with self.assertRaises(ValueError):
            download_and_unzip("file:///etc/passwd", "dest")


if __name__ == "__main__":
    unittest.main()
