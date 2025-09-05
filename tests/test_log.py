import unittest
from unittest.mock import patch
from datetime import datetime
import re
from definers import log

class TestLog(unittest.TestCase):

    @patch('builtins.print')
    def test_log_status_true(self, mock_print):
        subject = "Success"
        data = "Operation completed"
        log(subject, data, status=True)
        
        output = "\n".join([call[0][0] for call in mock_print.call_args_list])
        self.assertIn("OK OK OK OK OK OK OK", output)
        self.assertIn(subject, output)
        self.assertIn(data, output)

    @patch('builtins.print')
    def test_log_status_false(self, mock_print):
        subject = "Failure"
        data = "Operation failed"
        log(subject, data, status=False)
        
        output = "\n".join([call[0][0] for call in mock_print.call_args_list])
        self.assertIn("x ERR x ERR x ERR x", output)
        self.assertIn(subject, output)
        self.assertIn(data, output)

    @patch('builtins.print')
    def test_log_status_none(self, mock_print):
        subject = "Information"
        data = "Some info here"
        log(subject, data, status=None)
        
        output = "\n".join([call[0][0] for call in mock_print.call_args_list])
        self.assertIn("===================", output)
        self.assertIn(subject, output)
        self.assertIn(data, output)

    @patch('builtins.print')
    def test_log_status_string(self, mock_print):
        subject = "Custom Status"
        data = "Custom data"
        status_str = "CUSTOM"
        log(subject, data, status=status_str)
        
        output = "\n".join([call[0][0] for call in mock_print.call_args_list])
        self.assertIn(status_str, output)
        self.assertIn(subject, output)
        self.assertIn(data, output)

    @patch('builtins.print')
    def test_log_status_else(self, mock_print):
        subject = "Default Log"
        data = "Default data"
        log(subject, data, status="") 
        
        output = "\n".join([call[0][0] for call in mock_print.call_args_list])
        
        self.assertNotIn("OK OK OK", output)
        self.assertNotIn("x ERR x", output)
        self.assertNotIn("===", output)
        self.assertIn(subject, output)
        self.assertIn(data, output)

    @patch('definers.datetime')
    def test_log_timestamp(self, mock_datetime):
        fake_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = fake_now
        
        with patch('builtins.print') as mock_print:
            log("Timestamp Test", "Data", status=True)
        
        output = "\n".join([call[0][0] for call in mock_print.call_args_list])
        self.assertIn(str(fake_now.time()), output)

if __name__ == '__main__':
    unittest.main()
