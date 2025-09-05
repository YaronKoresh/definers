import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from definers import train, check_parameter, simple_text

class TestTrain(unittest.TestCase):

    def setUp(self):
       self.mock_tokenizer = MagicMock()
       self.mock_tokenizer.return_value = {
            'input_ids': np.array([[1, 2, 3]]),
            'attention_mask': np.array([[1, 1, 1]])
        }

    @patch('definers.fetch_dataset')
    @patch('definers.files_to_dataset')
    @patch('definers.init_tokenizer')
    @patch('joblib.load')
    @patch('joblib.dump')
    def test_train_with_preloaded_model(self, mock_dump, mock_load, mock_init_tokenizer, mock_files_to_dataset, mock_fetch_dataset):
        mock_init_tokenizer.return_value = self.mock_tokenizer
        mock_fetch_dataset.return_value = [{'feature': 'data'}]
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        result = train(model_path='existing_model.joblib', remote_src='some_remote_src')
        
        mock_load.assert_called_with('existing_model.joblib')
        mock_model.fit.assert_called()
        mock_dump.assert_called()
        self.assertIsNotNone(result)

    @patch('definers.fetch_dataset')
    @patch('definers.init_tokenizer')
    @patch('definers.HybridModel')
    @patch('joblib.dump')
    def test_train_supervised_remote(self, mock_dump, mock_hybrid_model, mock_init_tokenizer, mock_fetch_dataset):
        mock_init_tokenizer.return_value = self.mock_tokenizer
        mock_fetch_dataset.return_value = [{'feature_1': 'text 1', 'label_1': 'target 1'}]
        mock_model_instance = MagicMock()
        mock_hybrid_model.return_value = mock_model_instance

        result = train(remote_src='remote_data', dataset_label_columns=['label_1'])
        
        mock_fetch_dataset.assert_called_with('remote_data', "parquet", None)
        mock_model_instance.fit.assert_called()
        mock_dump.assert_called()
        self.assertRegex(result, r'model_.*\.joblib')

    @patch('definers.files_to_dataset')
    @patch('definers.init_tokenizer')
    @patch('definers.HybridModel')
    @patch('joblib.dump')
    def test_train_unsupervised_local(self, mock_dump, mock_hybrid_model, mock_init_tokenizer, mock_files_to_dataset):
        mock_init_tokenizer.return_value = self.mock_tokenizer
        mock_files_to_dataset.return_value = [{'feature': 'local data'}]
        mock_model_instance = MagicMock()
        mock_hybrid_model.return_value = mock_model_instance

        result = train(features=['local_file.txt'])

        mock_files_to_dataset.assert_called_with(['local_file.txt'], None)
        mock_model_instance.fit.assert_called()
        mock_dump.assert_called()
        self.assertTrue(result.endswith('.joblib'))

    def test_train_no_input(self):
        result = train()
        self.assertIsNone(result)

    @patch('definers.fetch_dataset')
    @patch('definers.init_tokenizer')
    @patch('joblib.dump')
    def test_train_save_error(self, mock_dump, mock_init_tokenizer, mock_fetch_dataset):
        mock_init_tokenizer.return_value = self.mock_tokenizer
        mock_fetch_dataset.return_value = [{'feature': 'data'}]
        mock_dump.side_effect = Exception("Failed to save")

        result = train(remote_src='some_path')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
