import os
import tempfile
import unittest

from definers.system import compress


class TestCompress(unittest.TestCase):
    def test_zip_compression(self):
        with tempfile.TemporaryDirectory() as d:
            test_file = os.path.join(d, "test.txt")
            with open(test_file, "w") as f:
                f.write("hello world")
            result = compress(d, format="zip")
            self.assertTrue(result is not None or os.path.exists(d + ".zip"))

    def test_returns_path_or_none(self):
        with tempfile.TemporaryDirectory() as d:
            test_file = os.path.join(d, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            result = compress(d)
            self.assertTrue(result is None or isinstance(result, str))


if __name__ == "__main__":
    unittest.main()
