import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from definers.application_ml import repository_sync, safe_deserialization
from definers.ml import AutoTrainer, init_custom_model


class _MaliciousPayload:
    def __reduce__(self):
        return (os.system, ("echo blocked",))


class TestDeserializationSecurity(unittest.TestCase):
    def _temp_path(self, suffix: str) -> str:
        file_descriptor, path = tempfile.mkstemp(suffix=suffix)
        os.close(file_descriptor)
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        return path

    def test_init_custom_model_loads_safe_pickle(self):
        model_path = self._temp_path(".pkl")
        expected_model = {"name": "safe-model", "version": 1}

        with open(model_path, "wb") as file_obj:
            pickle.dump(expected_model, file_obj)

        loaded_model = init_custom_model("pkl", model_path)

        self.assertEqual(loaded_model, expected_model)

    def test_init_custom_model_rejects_malicious_pickle(self):
        model_path = self._temp_path(".pkl")

        with open(model_path, "wb") as file_obj:
            pickle.dump(_MaliciousPayload(), file_obj)

        with patch("definers.ml.catch") as mock_catch:
            loaded_model = init_custom_model("pkl", model_path)

        self.assertIsNone(loaded_model)
        self.assertTrue(mock_catch.called)
        self.assertIn(
            "Unsafe serialized model rejected",
            str(mock_catch.call_args.args[0]),
        )

    def test_auto_trainer_load_supports_joblib_numpy_payloads(self):
        model_path = self._temp_path(".joblib")
        expected_model = {"weights": np.arange(4, dtype=np.float32)}
        trainer = AutoTrainer()

        joblib.dump(expected_model, model_path)

        loaded_model = trainer.load(model_path)

        self.assertEqual(trainer.model_path, model_path)
        np.testing.assert_array_equal(
            loaded_model["weights"], expected_model["weights"]
        )

    def test_auto_trainer_load_rejects_malicious_joblib(self):
        model_path = self._temp_path(".joblib")
        trainer = AutoTrainer()
        joblib.dump(_MaliciousPayload(), model_path)

        with self.assertRaisesRegex(
            ValueError, "Unsafe serialized model rejected"
        ):
            trainer.load(model_path)

    def test_repository_sync_rejects_malicious_joblib(self):
        model_path = self._temp_path(".joblib")
        joblib.dump(_MaliciousPayload(), model_path)

        with self.assertRaisesRegex(
            ValueError, "Unsafe serialized model rejected"
        ):
            repository_sync._load_model(model_path, "joblib")

    def test_repository_sync_rejects_html_joblib_download(self):
        model_path = self._temp_path(".joblib")
        Path(model_path).write_text(
            "<html>not a model</html>", encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "Downloaded HTML"):
            repository_sync._validate_downloaded_model_file(
                model_path,
                "joblib",
            )

    def test_repository_sync_passes_trusted_directories_to_serialized_loader(
        self,
    ):
        trusted_directories = (
            "C:\\Users\\User\\.cache\\huggingface\\hub\\model-a",
        )

        with patch.object(
            safe_deserialization,
            "load_serialized_model",
            return_value={"name": "safe-model"},
        ) as mock_load_serialized_model:
            loaded_model = repository_sync._load_model(
                "C:/Users/User/.cache/huggingface/hub/model-a/model.joblib",
                "joblib",
                trusted_directories=trusted_directories,
            )

        self.assertEqual(loaded_model, {"name": "safe-model"})
        mock_load_serialized_model.assert_called_once_with(
            "C:/Users/User/.cache/huggingface/hub/model-a/model.joblib",
            "joblib",
            trusted_directories=trusted_directories,
        )


if __name__ == "__main__":
    unittest.main()
