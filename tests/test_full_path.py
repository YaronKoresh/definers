import os
import unittest
from definers import full_path


class TestFullPath(unittest.TestCase):
    def test_returns_absolute_path(self):
        result = full_path("test")
        self.assertTrue(os.path.isabs(result))

    def test_joins_components(self):
        result = full_path("foo", "bar", "baz")
        self.assertTrue(result.endswith(os.path.join("foo", "bar", "baz")))

    def test_single_component(self):
        result = full_path("single")
        self.assertIn("single", result)


if __name__ == "__main__":
    unittest.main()
