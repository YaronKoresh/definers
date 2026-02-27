import tempfile
import unittest

from definers import file_to_sha3_512, string_to_sha3_512


class TestFileToSha3512(unittest.TestCase):
    def test_returns_hash_of_file_contents(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        result = file_to_sha3_512(path)
        expected = string_to_sha3_512("hello")
        self.assertEqual(result, expected)

    def test_nonexistent_file_returns_none(self):
        result = file_to_sha3_512("/nonexistent/path/abc123.txt")
        self.assertIsNone(result)

    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("consistent content")
            path = f.name
        r1 = file_to_sha3_512(path)
        r2 = file_to_sha3_512(path)
        self.assertEqual(r1, r2)

    def test_salt_changes_result(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("salted content")
            path = f.name
        r1 = file_to_sha3_512(path)
        r2 = file_to_sha3_512(path, salt_num=42)
        self.assertNotEqual(r1, r2)

    def test_returns_hex_string(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hex check")
            path = f.name
        result = file_to_sha3_512(path)
        self.assertIsInstance(result, str)
        int(result, 16)


if __name__ == "__main__":
    unittest.main()
