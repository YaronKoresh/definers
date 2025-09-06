import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from definers import tokenize_and_pad


class TestTokenizeAndPad(unittest.TestCase):

    def setUp(self):
        self.mock_tokenizer_instance = MagicMock()
        self.mock_tokenizer_instance.return_value = {
            "input_ids": [
                [101, 2054, 2049, 102],
                [101, 2088, 2049, 102],
            ]
        }

    @patch("definers.two_dim_numpy")
    def test_tokenize_with_dict_rows(self, mock_two_dim_numpy):
        mock_two_dim_numpy.side_effect = lambda x: np.array(x)
        rows = [
            {"feature1": "hello world", "feature2": 123},
            {"feature3": "another row", "feature4": [4, 5]},
        ]

        tokenize_and_pad(rows, tokenizer=self.mock_tokenizer_instance)

        self.mock_tokenizer_instance.assert_called_once()
        call_args = self.mock_tokenizer_instance.call_args[0][0]
        self.assertIn("hello world 123", call_args)
        self.assertIn("another row 4 5", call_args)

    @patch("definers.two_dim_numpy")
    def test_tokenize_with_string_rows(self, mock_two_dim_numpy):
        mock_two_dim_numpy.side_effect = lambda x: np.array(x)
        rows = ["first sentence", "second sentence"]

        tokenize_and_pad(rows, tokenizer=self.mock_tokenizer_instance)

        self.mock_tokenizer_instance.assert_called_once_with(
            ["first sentence", "second sentence"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    @patch("definers.two_dim_numpy")
    def test_tokenize_with_mixed_rows(self, mock_two_dim_numpy):
        mock_two_dim_numpy.side_effect = lambda x: np.array(x)
        rows = [
            {"feature1": "dict row", "feature2": 99},
            "a string row",
        ]

        tokenize_and_pad(rows, tokenizer=self.mock_tokenizer_instance)

        self.mock_tokenizer_instance.assert_called_once()
        call_args = self.mock_tokenizer_instance.call_args[0][0]
        self.assertEqual(call_args, ["dict row 99", "a string row"])

    def test_returns_original_for_unsupported_type(self):
        rows = [1, 2, 3]
        result = tokenize_and_pad(
            rows, tokenizer=self.mock_tokenizer_instance
        )
        self.assertEqual(result, [1, 2, 3])
        self.mock_tokenizer_instance.assert_not_called()

    @patch("definers.two_dim_numpy")
    @patch("definers.init_tokenizer")
    def test_initializes_tokenizer_if_not_provided(
        self, mock_init_tokenizer, mock_two_dim_numpy
    ):
        mock_init_tokenizer.return_value = (
            self.mock_tokenizer_instance
        )
        mock_two_dim_numpy.side_effect = lambda x: np.array(x)

        rows = ["some text"]
        tokenize_and_pad(rows)

        mock_init_tokenizer.assert_called_once()
        self.mock_tokenizer_instance.assert_called_once_with(
            ["some text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def test_handles_empty_input(self):
        rows = []
        result = tokenize_and_pad(
            rows, tokenizer=self.mock_tokenizer_instance
        )

        self.mock_tokenizer_instance.assert_called_once_with(
            [], padding=True, truncation=True, return_tensors="pt"
        )

    @patch("definers.two_dim_numpy")
    def test_handles_none_values_in_dicts(self, mock_two_dim_numpy):
        mock_two_dim_numpy.side_effect = lambda x: np.array(x)
        rows = [
            {"feature1": "value", "feature2": None},
            {"feature3": None, "feature4": "another value"},
        ]

        tokenize_and_pad(rows, tokenizer=self.mock_tokenizer_instance)

        self.mock_tokenizer_instance.assert_called_once()
        call_args = self.mock_tokenizer_instance.call_args[0][0]
        self.assertEqual(call_args, ["value", "another value"])


if __name__ == "__main__":
    unittest.main()
