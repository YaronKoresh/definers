import unittest
from unittest.mock import patch, call
from definers import paths

class TestPaths(unittest.TestCase):

    @patch('definers.glob')
    @patch('os.path.abspath')
    @patch('os.path.expanduser')
    def test_single_pattern(self, mock_expanduser, mock_abspath, mock_glob):
        mock_abspath.side_effect = lambda p: p
        mock_expanduser.side_effect = lambda p: p
        mock_glob.return_value = ['/tmp/file1.txt', '/tmp/file2.log']
        
        result = paths('/tmp/*')
        
        mock_glob.assert_called_once_with('/tmp/*', recursive=True)
        self.assertCountEqual(result, ['/tmp/file1.txt', '/tmp/file2.log'])

    @patch('definers.glob')
    @patch('os.path.abspath')
    @patch('os.path.expanduser')
    def test_multiple_patterns(self, mock_expanduser, mock_abspath, mock_glob):
        mock_abspath.side_effect = lambda p: p
        mock_expanduser.side_effect = lambda p: p
        mock_glob.side_effect = [
            ['/home/user/doc1.txt'],
            ['/etc/conf1.conf', '/etc/conf2.conf']
        ]
        
        result = paths('/home/user/*.txt', '/etc/*.conf')
        
        self.assertEqual(mock_glob.call_count, 2)
        mock_glob.assert_has_calls([
            call('/home/user/*.txt', recursive=True),
            call('/etc/*.conf', recursive=True)
        ])
        self.assertCountEqual(result, ['/home/user/doc1.txt', '/etc/conf1.conf', '/etc/conf2.conf'])

    @patch('definers.glob')
    @patch('os.path.abspath')
    @patch('os.path.expanduser')
    def test_no_matches(self, mock_expanduser, mock_abspath, mock_glob):
        mock_abspath.side_effect = lambda p: p
        mock_expanduser.side_effect = lambda p: p
        mock_glob.return_value = []
        
        result = paths('/nonexistent/path/*')
        
        mock_glob.assert_called_once_with('/nonexistent/path/*', recursive=True)
        self.assertEqual(result, [])

    def test_no_patterns_provided(self):
        result = paths()
        self.assertEqual(result, [])

    @patch('definers.glob')
    @patch('os.path.abspath', side_effect=lambda p: p.replace('~', '/home/user'))
    @patch('os.path.expanduser', side_effect=lambda p: p.replace('~', '/home/user'))
    def test_home_directory_expansion(self, mock_expanduser, mock_abspath, mock_glob):
        mock_glob.return_value = ['/home/user/docs/report.docx']

        result = paths('~/docs/*.docx')
        
        mock_expanduser.assert_called_once_with('~/docs/*.docx')
        mock_abspath.assert_called_once_with('/home/user/docs/*.docx')
        mock_glob.assert_called_once_with('/home/user/docs/*.docx', recursive=True)
        self.assertCountEqual(result, ['/home/user/docs/report.docx'])

    @patch('definers.glob')
    @patch('os.path.abspath')
    @patch('os.path.expanduser')
    def test_duplicate_paths_are_removed(self, mock_expanduser, mock_abspath, mock_glob):
        mock_abspath.side_effect = lambda p: p
        mock_expanduser.side_effect = lambda p: p
        mock_glob.side_effect = [
            ['/data/file.csv'],
            ['/data/file.csv', '/data/another.csv']
        ]

        result = paths('/data/file.csv', '/data/*.csv')
        
        self.assertCountEqual(result, ['/data/file.csv', '/data/another.csv'])

    @patch('definers.glob')
    @patch('os.path.abspath')
    @patch('os.path.expanduser')
    def test_glob_exception(self, mock_expanduser, mock_abspath, mock_glob):
        mock_abspath.side_effect = lambda p: p
        mock_expanduser.side_effect = lambda p: p
        mock_glob.side_effect = Exception("Test exception")

        result = paths('/some/pattern/*')
        
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
