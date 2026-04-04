import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from definers.application_ml.trainer_plan import render_training_plan_markdown
from definers.ml import AutoTrainer


class TestAutoTrainer(unittest.TestCase):
    def test_fit_accepts_mapping_payload(self):
        trainer = AutoTrainer()
        trained_model = MagicMock()

        with (
            patch("definers.ml._feed", return_value=trained_model) as mock_feed,
            patch("definers.ml._fit", return_value=trained_model) as mock_fit,
            patch(
                "definers.ml._training_array_adapter", return_value=MagicMock()
            ),
        ):
            result = trainer.fit(
                data={"features": [[1, 2], [3, 4]], "labels": [1, 0]}
            )

        self.assertIs(result, trained_model)
        mock_feed.assert_called_once()
        feature_arg, label_arg = mock_feed.call_args.args[1:3]
        np.testing.assert_array_equal(feature_arg, np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(label_arg, np.array([1, 0]))
        mock_fit.assert_called_once()
        self.assertIs(mock_fit.call_args.args[0], trained_model)

    def test_train_uses_in_memory_data_and_saves_model(self):
        trainer = AutoTrainer(batch_size=4)
        trained_model = MagicMock()

        with (
            patch("definers.ml._feed", return_value=trained_model),
            patch("definers.ml._fit", return_value=trained_model),
            patch(
                "definers.ml._training_array_adapter", return_value=MagicMock()
            ),
            patch("joblib.dump") as mock_dump,
        ):
            model_path = trainer.train(
                data=[[1.0, 2.0], [3.0, 4.0]],
                target=["cat", "dog"],
                save_as="auto-model.joblib",
            )

        self.assertEqual(model_path, "auto-model.joblib")
        self.assertEqual(trainer.model_path, "auto-model.joblib")
        self.assertEqual(trainer.label_mapping, {"cat": 0, "dog": 1})
        mock_dump.assert_called_once_with(trained_model, "auto-model.joblib")

    def test_train_uses_embedded_file_pipeline_for_file_inputs(self):
        trainer = AutoTrainer(batch_size=8, source_type="json")
        trained_model = MagicMock()
        training_data = MagicMock(
            train=[[{"feature": "a", "label": "b"}]], val=None, test=None
        )

        with (
            patch("definers.ml._feed", return_value=trained_model) as mock_feed,
            patch("definers.ml._fit", return_value=trained_model),
            patch(
                "definers.ml._training_array_adapter", return_value=MagicMock()
            ),
            patch(
                "definers.application_data.preparation.prepare_data",
                return_value=training_data,
            ) as mock_prepare,
            patch(
                "definers.application_data.tokenization.init_tokenizer",
                return_value=object(),
            ),
            patch(
                "definers.application_data.tokenization.tokenize_and_pad",
                side_effect=lambda rows, tokenizer: rows,
            ),
            patch(
                "definers.application_data.preparation.pad_sequences",
                side_effect=lambda rows: rows,
            ),
            patch(
                "definers.application_data.arrays.numpy_to_cupy",
                side_effect=lambda value: value,
            ),
            patch("joblib.dump") as mock_dump,
        ):
            model_path = trainer.train(
                data=["features.csv"],
                target=["labels.csv"],
                save_as="pipeline.joblib",
            )

        self.assertEqual(model_path, "pipeline.joblib")
        mock_prepare.assert_called_once()
        self.assertEqual(
            mock_prepare.call_args.kwargs["features"], ["features.csv"]
        )
        self.assertEqual(
            mock_prepare.call_args.kwargs["labels"], ["labels.csv"]
        )
        self.assertEqual(mock_prepare.call_args.kwargs["batch_size"], 8)
        self.assertEqual(mock_prepare.call_args.kwargs["url_type"], "json")
        mock_feed.assert_called_once()
        mock_dump.assert_called_once_with(trained_model, "pipeline.joblib")

    def test_train_url_uses_beginner_friendly_online_entry_point(self):
        trainer = AutoTrainer()

        with patch.object(
            trainer, "train", return_value="remote.joblib"
        ) as mock_train:
            model_path = trainer.train_url(
                "https://example.com/data.parquet",
                target="label",
                save_as="remote.joblib",
            )

        self.assertEqual(model_path, "remote.joblib")
        mock_train.assert_called_once_with(
            save_as="remote.joblib",
            revision=None,
            source_type=None,
        )
        self.assertEqual(trainer.source, "https://example.com/data.parquet")
        self.assertEqual(trainer.target, "label")

    def test_predict_uses_loaded_model_for_in_memory_data(self):
        model = MagicMock()
        model.predict.return_value = np.array([1, 0])
        trainer = AutoTrainer(model=model)

        result = trainer.predict([[1, 2], [3, 4]])

        np.testing.assert_array_equal(result, np.array([1, 0]))
        model.predict.assert_called_once()

    def test_infer_uses_task_for_path_inputs(self):
        trainer = AutoTrainer(task="answer")

        with (
            patch("definers.ml.init_model_file") as mock_init,
            patch.dict(
                "definers.ml.MODELS",
                {
                    "answer": MagicMock(
                        predict=MagicMock(return_value=np.array([[1.0, 0.0]]))
                    )
                },
                clear=True,
            ),
            patch("definers.ml.read", return_value="hello"),
            patch("definers.ml.create_vectorizer", return_value=MagicMock()),
            patch(
                "definers.ml.extract_text_features",
                return_value=np.array([1.0, 0.0], dtype=np.float32),
            ),
            patch("definers.ml.numpy_to_cupy", side_effect=lambda value: value),
            patch(
                "definers.ml.one_dim_numpy",
                side_effect=lambda value: np.asarray(value).reshape(1, -1),
            ),
            patch(
                "definers.ml.cupy_to_numpy",
                side_effect=lambda value: np.asarray(value),
            ),
            patch("definers.ml.features_to_text", return_value="prediction"),
            patch("builtins.open", unittest.mock.mock_open()),
            patch("definers.ml.random_string", return_value="prediction"),
        ):
            result = trainer.infer("sample.txt")

        self.assertEqual(result, "prediction.txt")
        mock_init.assert_not_called()

    def test_training_plan_describes_remote_dataset_flow(self):
        trainer = AutoTrainer(
            batch_size=16,
            source_type="parquet",
            revision="main",
            validation_split=0.1,
            test_split=0.2,
        )

        plan = trainer.training_plan(
            data="owner/dataset",
            target="label",
            label_columns="label",
            drop="unused",
            select="1-20",
            resume_from="model.joblib",
        )

        self.assertEqual(plan.mode, "remote-dataset")
        self.assertEqual(plan.batch_size, 16)
        self.assertEqual(plan.source_type, "parquet")
        self.assertEqual(plan.label_columns, ("label",))
        self.assertEqual(plan.drop_columns, ("unused",))
        self.assertEqual(plan.selected_rows, "1-20")
        self.assertEqual(plan.resume_from, "model.joblib")
        self.assertIn(
            "Mode: remote-dataset", render_training_plan_markdown(plan)
        )


if __name__ == "__main__":
    unittest.main()
