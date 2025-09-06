import unittest
from unittest.mock import MagicMock, patch

from definers import init_tokenizer


class TestInitTokenizer(unittest.TestCase):

    @patch("definers.AutoTokenizer.from_pretrained")
    def test_initializes_with_default_model(
        self, mock_from_pretrained
    ):
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = init_tokenizer()

        mock_from_pretrained.assert_called_once_with(
            "google-bert/bert-base-multilingual-cased"
        )
        self.assertEqual(tokenizer, mock_tokenizer)

    @patch("definers.AutoTokenizer.from_pretrained")
    def test_initializes_with_custom_model(
        self, mock_from_pretrained
    ):
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer
        custom_model_name = "distilbert-base-uncased"

        tokenizer = init_tokenizer(mod=custom_model_name)

        mock_from_pretrained.assert_called_once_with(
            custom_model_name
        )
        self.assertEqual(tokenizer, mock_tokenizer)


if __name__ == "__main__":
    unittest.main()
