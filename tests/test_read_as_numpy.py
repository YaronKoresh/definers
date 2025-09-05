import unittest
import os
import tempfile
import shutil
from unittest.mock import patch
from definers import read_as_numpy

class TestReadAsNumpy(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dummy_path = os.path.join(self.test_dir, "dummy_file.txt")
        with open(self.dummy_path, "w") as f:
            f.write("data")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('definers.load_as_numpy')
    def test_read_as_numpy_calls_load_as_numpy(self, mock_load_as_numpy):
        expected_result = "mocked_numpy_array"
        mock_load_as_numpy.return_value = expected_result
        
        result = read_as_numpy(self.dummy_path)
        
        mock_load_as_numpy.assert_called_once_with(self.dummy_path)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
