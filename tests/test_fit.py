import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import definers.data.preparation as preparation
from definers.ml.training import fit


class TestFit(unittest.TestCase):
    def setUp(self):
        self.mock_model_supervised = MagicMock()
        self.mock_model_supervised.X_all = np.array([[1, 2], [3, 4]])
        self.mock_model_supervised.y_all = np.array([1, 2])
        self.mock_model_unsupervised = MagicMock()
        self.mock_model_unsupervised.X_all = np.array([[5, 6], [7, 8]])
        del self.mock_model_unsupervised.y_all
        self.array_adapter = SimpleNamespace(
            get_max_shapes=MagicMock(),
            reshape_numpy=MagicMock(side_effect=lambda x, lengths: x),
            numpy_to_cupy=MagicMock(side_effect=lambda x: x),
            cupy_to_numpy=MagicMock(side_effect=lambda x: x),
            catch=MagicMock(),
        )
        self.logger = MagicMock()
        self.error_handler = MagicMock()

    def test_fit_supervised(self):
        self.array_adapter.get_max_shapes.return_value = [(2, 2), (2,)]
        returned_model = fit(
            self.mock_model_supervised,
            array_adapter=self.array_adapter,
            logger=self.logger,
            error_handler=self.error_handler,
        )
        self.logger.assert_any_call(
            "Features", self.mock_model_supervised.X_all
        )
        self.logger.assert_any_call("Labels", self.mock_model_supervised.y_all)
        self.array_adapter.get_max_shapes.assert_called_once_with(
            self.mock_model_supervised.X_all, self.mock_model_supervised.y_all
        )
        self.mock_model_supervised.fit.assert_called_once_with(
            self.mock_model_supervised.X_all, self.mock_model_supervised.y_all
        )
        self.assertIs(returned_model, self.mock_model_supervised)

    def test_fit_unsupervised(self):
        self.array_adapter.get_max_shapes.return_value = [(2, 2)]
        returned_model = fit(
            self.mock_model_unsupervised,
            array_adapter=self.array_adapter,
            logger=self.logger,
            error_handler=self.error_handler,
        )
        self.logger.assert_any_call(
            "Features", self.mock_model_unsupervised.X_all
        )
        self.array_adapter.get_max_shapes.assert_called_once_with(
            self.mock_model_unsupervised.X_all
        )
        self.mock_model_unsupervised.fit.assert_called_once_with(
            self.mock_model_unsupervised.X_all
        )
        self.assertIs(returned_model, self.mock_model_unsupervised)

    def test_fit_supervised_exception(self):
        self.array_adapter.get_max_shapes.return_value = [(2, 2), (2,)]
        self.mock_model_supervised.fit.side_effect = [
            Exception("Supervised fit error"),
            None,
        ]
        fit(
            self.mock_model_supervised,
            array_adapter=self.array_adapter,
            logger=self.logger,
            error_handler=self.error_handler,
        )
        self.error_handler.assert_called_once()
        self.assertIsInstance(self.error_handler.call_args[0][0], Exception)

    def test_fit_unsupervised_exception(self):
        self.array_adapter.get_max_shapes.return_value = [(2, 2)]
        self.mock_model_unsupervised.fit.side_effect = [
            Exception("Unsupervised fit error"),
            None,
        ]
        fit(
            self.mock_model_unsupervised,
            array_adapter=self.array_adapter,
            logger=self.logger,
            error_handler=self.error_handler,
        )
        self.error_handler.assert_called_once()
        self.assertIsInstance(self.error_handler.call_args[0][0], Exception)


class TestPrepareDataCache(unittest.TestCase):
    def tearDown(self):
        preparation.clear_prepare_data_cache()

    def test_prepare_data_cache_reuses_callable_descriptor_and_clear_hook(self):
        loader_runtime = SimpleNamespace(
            load_source=MagicMock(return_value=[2, 1])
        )

        def rank_row(value):
            return value

        with patch.object(
            preparation, "_data_runtime", return_value=loader_runtime
        ):
            first = preparation.prepare_data(
                remote_src="demo",
                order_by=rank_row,
                batch_size=2,
            )
            second = preparation.prepare_data(
                remote_src="demo",
                order_by=rank_row,
                batch_size=2,
            )

        self.assertIs(first, second)
        loader_runtime.load_source.assert_called_once_with(
            "demo",
            None,
            None,
            None,
            None,
        )
        manifest = preparation.prepare_data_cache_manifest()
        self.assertEqual(len(manifest), 1)
        self.assertIn("rank_row", manifest[0]["order_by"])
        self.assertEqual(preparation.clear_prepare_data_cache(), 1)
        self.assertEqual(preparation.prepare_data_cache_manifest(), [])

    def test_prepare_data_cache_evicts_oldest_entry(self):
        loader_runtime = SimpleNamespace(
            load_source=MagicMock(return_value=[1])
        )

        with patch.object(
            preparation, "_data_runtime", return_value=loader_runtime
        ):
            for index in range(preparation._PREPARE_DATA_CACHE_MAX_SIZE + 1):
                preparation.prepare_data(remote_src=f"demo-{index}")

        manifest = preparation.prepare_data_cache_manifest()
        self.assertEqual(
            len(manifest),
            preparation._PREPARE_DATA_CACHE_MAX_SIZE,
        )
        self.assertNotIn(
            "demo-0",
            [entry["remote_src"] for entry in manifest],
        )


if __name__ == "__main__":
    unittest.main()
