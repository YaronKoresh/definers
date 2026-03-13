import os
import unittest

import gradio as gr

from definers import regex_utils, run
from definers.constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH
from definers.ml import train
from definers.system import secure_path
from definers.web import download_and_unzip, download_file


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

        with self.assertRaises(ValueError):
            train(
                model_path=None,
                remote_src=None,
                features=None,
                labels=None,
                selected_rows="x" * (MAX_INPUT_LENGTH + 1),
            )
        with self.assertRaises(ValueError):
            train(
                model_path=None,
                remote_src=None,
                features=None,
                labels=None,
                selected_rows="1 " + " " * (MAX_CONSECUTIVE_SPACES + 2) + "2",
            )
        with self.assertRaises(ValueError):
            train(
                model_path=None,
                remote_src="http://" + "a" * (MAX_INPUT_LENGTH + 1),
                features=None,
                labels=None,
            )

    def test_download_validation(self):
        with self.assertRaises(ValueError):
            download_file("ftp://example.com/file", "dest")
        with self.assertRaises(ValueError):
            download_file("http://" + "a" * (MAX_INPUT_LENGTH + 1), "dest")
        with self.assertRaises(ValueError):
            download_and_unzip("file:///etc/passwd", "dest")


if __name__ == "__main__":
    unittest.main()
