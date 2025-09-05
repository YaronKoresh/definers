import unittest
from unittest.mock import patch, MagicMock, ANY
from definers import train

class TestTrain(unittest.TestCase):

    @patch('definers.joblib.dump')
    @patch('definers.fit', return_value=MagicMock())
    @patch('definers.feed', return_value=MagicMock())
    @patch('definers.numpy_to_cupy', side_effect=lambda x: x)
    @patch('definers.pad_sequences', side_effect=lambda x: x)
    @patch('definers.tokenize_and_pad', return_value='tokenized_padded')
    @patch('definers.split_columns', return_value=('X_batch', 'y_batch'))
    @patch('definers.to_loader', return_value=[['batch1_data']])
    @patch('definers.drop_columns', side_effect=lambda d, l: d)
    @patch('definers.fetch_dataset', return_value='remote_dataset')
    @patch('definers.init_tokenizer')
    @patch('definers.check_parameter', side_effect=lambda p: p is not None and p != '')
    @patch('definers.random_string', return_value='random_model_name')
    @patch('definers.log')
    def test_train_supervised_remote(self, mock_log, mock_random_string, mock_check_parameter, mock_init_tokenizer, mock_fetch, mock_drop, mock_to_loader, mock_split, mock_tokenize, mock_pad, mock_numpy_to_cupy, mock_feed, mock_fit, mock_dump):
        model_path = train(remote_src='some_remote_path', dataset_label_columns=['label'])
        
        mock_fetch.assert_called_with('some_remote_path', 'parquet', None)
        mock_split.assert_called_with(['batch1_data'], ['label'], is_batch=True)
        mock_tokenize.assert_called_with('y_batch', ANY)
        mock_fit.assert_called_once()
        mock_dump.assert_called_with(ANY, 'model_random_model_name.joblib')
        self.assertEqual(model_path, 'model_random_model_name.joblib')

    @patch('definers.joblib.dump')
    @patch('definers.fit', return_value=MagicMock())
    @patch('definers.feed', return_value=MagicMock())
    @patch('definers.numpy_to_cupy', side_effect=lambda x: x)
    @patch('definers.pad_sequences', side_effect=lambda x: x)
    @patch('definers.tokenize_and_pad', return_value='tokenized_padded')
    @patch('definers.to_loader', return_value=[['batch1_data']])
    @patch('definers.drop_columns', side_effect=lambda d, l: d)
    @patch('definers.files_to_dataset', return_value='local_dataset')
    @patch('definers.init_tokenizer')
    @patch('definers.check_parameter', side_effect=lambda p: p is not None and p != '')
    @patch('definers.random_string', return_value='random_model_name')
    @patch('definers.log')
    def test_train_unsupervised_local(self, mock_log, mock_random_string, mock_check_parameter, mock_init_tokenizer, mock_files_to_dataset, mock_drop, mock_to_loader, mock_tokenize, mock_pad, mock_numpy_to_cupy, mock_feed, mock_fit, mock_dump):
        model_path = train(features=['feature.file'])
        
        mock_files_to_dataset.assert_called_with(['feature.file'], None)
        mock_tokenize.assert_called_with(['batch1_data'], ANY)
        mock_fit.assert_called_once()
        self.assertEqual(model_path, 'model_random_model_name.joblib')

    @patch('definers.joblib.load', return_value=MagicMock())
    @patch('definers.joblib.dump')
    @patch('definers.fit')
    @patch('definers.feed')
    @patch('definers.fetch_dataset', return_value='remote_dataset')
    @patch('definers.to_loader', return_value=[])
    @patch('definers.init_tokenizer')
    @patch('definers.check_parameter', side_effect=lambda p: p is not None and p != '')
    @patch('definers.log')
    def test_train_with_preloaded_model(self, mock_log, mock_check_parameter, mock_init_tokenizer, mock_to_loader, mock_fetch, mock_feed, mock_fit, mock_dump, mock_load):
        train(model_path='existing_model.joblib', remote_src='some_remote_path')
        mock_load.assert_called_once_with('existing_model.joblib')

    @patch('definers.check_parameter', return_value=False)
    @patch('definers.log')
    def test_train_no_input(self, mock_log, mock_check_parameter):
        result = train()
        self.assertIsNone(result)

    @patch('definers.joblib.dump', side_effect=Exception("Save Error"))
    @patch('definers.fit', return_value=MagicMock())
    @patch('definers.feed', return_value=MagicMock())
    @patch('definers.fetch_dataset', return_value='remote_dataset')
    @patch('definers.to_loader', return_value=[['batch_data']])
    @patch('definers.tokenize_and_pad', return_value='tokenized')
    @patch('definers.pad_sequences', return_value='padded')
    @patch('definers.numpy_to_cupy', return_value='cupy_array')
    @patch('definers.init_tokenizer')
    @patch('definers.check_parameter', return_value=True)
    @patch('definers.log')
    def test_train_save_error(self, mock_log, mock_check, mock_init, mock_numpy_cupy, mock_pad, mock_tokenize, mock_to_loader, mock_fetch, mock_feed, mock_fit, mock_dump):
        result = train(remote_src='some_path')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
