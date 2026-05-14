import importlib
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from definers.ml.trainer_plan import render_training_plan_markdown


def _ml_module():
    return importlib.import_module("definers.ml")


class TestAutoTrainer(unittest.TestCase):
    def test_fit_accepts_mapping_payload(self):
        ml_module = _ml_module()
        trainer = ml_module.AutoTrainer()
        trained_model = MagicMock()

        with (
            patch.object(
                ml_module, "feed", return_value=trained_model
            ) as mock_feed,
            patch.object(
                ml_module, "fit", return_value=trained_model
            ) as mock_fit,
            patch.object(
                ml_module, "_training_array_adapter", return_value=MagicMock()
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
        ml_module = _ml_module()
        trainer = ml_module.AutoTrainer(batch_size=4)
        trained_model = MagicMock()

        with (
            patch.object(ml_module, "feed", return_value=trained_model),
            patch.object(ml_module, "fit", return_value=trained_model),
            patch.object(
                ml_module, "_training_array_adapter", return_value=MagicMock()
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
        ml_module = _ml_module()
        trainer = ml_module.AutoTrainer(batch_size=8, source_type="json")
        trained_model = MagicMock()
        training_data = MagicMock(
            train=[[{"feature": "a", "label": "b"}]], val=None, test=None
        )

        with (
            patch.object(
                ml_module, "feed", return_value=trained_model
            ) as mock_feed,
            patch.object(ml_module, "fit", return_value=trained_model),
            patch.object(
                ml_module, "_training_array_adapter", return_value=MagicMock()
            ),
            patch(
                "definers.data.preparation.prepare_data",
                return_value=training_data,
            ) as mock_prepare,
            patch(
                "definers.data.tokenization.init_tokenizer",
                return_value=object(),
            ),
            patch(
                "definers.data.tokenization.tokenize_and_pad",
                side_effect=lambda rows, tokenizer: rows,
            ),
            patch(
                "definers.data.preparation.pad_sequences",
                side_effect=lambda rows: rows,
            ),
            patch(
                "definers.data.arrays.numpy_to_cupy",
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
        trainer = _ml_module().AutoTrainer()

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
        trainer = _ml_module().AutoTrainer(model=model)

        result = trainer.predict([[1, 2], [3, 4]])

        np.testing.assert_array_equal(result, np.array([1, 0]))
        model.predict.assert_called_once()

    def test_infer_uses_task_for_path_inputs(self):
        ml_module = _ml_module()
        trainer = ml_module.AutoTrainer(task="answer")

        with (
            patch.object(ml_module, "init_model_file") as mock_init,
            patch.dict(
                ml_module.MODELS,
                {
                    "answer": MagicMock(
                        predict=MagicMock(return_value=np.array([[1.0, 0.0]]))
                    )
                },
                clear=True,
            ),
            patch.object(ml_module, "read", return_value="hello"),
            patch.object(
                ml_module, "create_vectorizer", return_value=MagicMock()
            ),
            patch.object(
                ml_module,
                "extract_text_features",
                return_value=np.array([1.0, 0.0], dtype=np.float32),
            ),
            patch.object(
                ml_module, "numpy_to_cupy", side_effect=lambda value: value
            ),
            patch.object(
                ml_module,
                "one_dim_numpy",
                side_effect=lambda value: np.asarray(value).reshape(1, -1),
            ),
            patch.object(
                ml_module,
                "cupy_to_numpy",
                side_effect=lambda value: np.asarray(value),
            ),
            patch.object(
                ml_module, "features_to_text", return_value="prediction"
            ),
            patch("builtins.open", unittest.mock.mock_open()),
            patch.object(ml_module, "random_string", return_value="prediction"),
        ):
            result = trainer.infer("sample.txt")

        self.assertEqual(result, "prediction.txt")
        mock_init.assert_not_called()

    def test_training_plan_describes_remote_dataset_flow(self):
        trainer = _ml_module().AutoTrainer(
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
            order_by="shuffle",
            select="1-20",
            stratify="label",
            resume_from="model.joblib",
        )

        self.assertEqual(plan.mode, "remote-dataset")
        self.assertEqual(plan.batch_size, 16)
        self.assertEqual(plan.source_type, "parquet")
        self.assertEqual(plan.label_columns, ("label",))
        self.assertEqual(plan.drop_columns, ("unused",))
        self.assertEqual(plan.order_by, "shuffle")
        self.assertEqual(plan.stratify, "label")
        self.assertEqual(plan.selected_rows, "1-20")
        self.assertEqual(plan.resume_from, "model.joblib")
        self.assertIn(
            "Mode: remote-dataset", render_training_plan_markdown(plan)
        )
        self.assertIn("Order By: shuffle", render_training_plan_markdown(plan))

    def test_file_dataset_detection_rejects_untrusted_absolute_path(self):
        trainer = _ml_module().AutoTrainer()
        outside_path = str(Path.cwd().parent / "outside.csv")

        self.assertFalse(trainer._is_file_dataset(outside_path))


class TestAutoTuneAndConvenience(unittest.TestCase):
    def test_suggest_training_defaults_picks_validation_for_large_dataset(self):
        from definers.ml.trainer_plan import suggest_training_defaults

        defaults = suggest_training_defaults(2000)

        self.assertEqual(defaults.batch_size, 64)
        self.assertEqual(defaults.validation_split, 0.1)
        self.assertTrue(defaults.early_stopping)
        self.assertGreaterEqual(defaults.patience, 1)
        self.assertTrue(defaults.notes)

    def test_suggest_training_defaults_respects_caller_values(self):
        from definers.ml.trainer_plan import suggest_training_defaults

        defaults = suggest_training_defaults(
            5000,
            batch_size=8,
            validation_split=0.25,
            early_stopping=False,
        )

        self.assertEqual(defaults.batch_size, 8)
        self.assertEqual(defaults.validation_split, 0.25)
        self.assertFalse(defaults.early_stopping)

    def test_training_plan_records_auto_tune_notes(self):
        ml_module = _ml_module()
        trainer = ml_module.AutoTrainer()
        features = np.random.rand(500, 3)
        labels = np.random.randint(0, 2, size=500)

        plan = trainer.training_plan(data=features, target=labels)

        self.assertTrue(plan.auto_tuned)
        self.assertGreater(plan.validation_split, 0.0)
        self.assertTrue(plan.early_stopping)
        markdown = render_training_plan_markdown(plan)
        self.assertIn("Auto-Tuned", markdown)
        self.assertIn("Early Stopping", markdown)

    def test_auto_train_convenience_invokes_trainer(self):
        ml_module = _ml_module()
        trained_model = MagicMock()

        with (
            patch.object(ml_module, "feed", return_value=trained_model),
            patch.object(ml_module, "fit", return_value=trained_model),
            patch.object(
                ml_module, "_training_array_adapter", return_value=MagicMock()
            ),
            patch("joblib.dump") as mock_dump,
        ):
            result = ml_module.auto_train(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                ["a", "b", "a", "b"],
                save_as="quick.joblib",
            )

        self.assertIsInstance(result, ml_module.AutoTrainingResult)
        self.assertEqual(result.artifact_path, "quick.joblib")
        self.assertEqual(result, "quick.joblib")
        mock_dump.assert_called_once()

    def test_auto_train_uses_guided_coach_for_single_file(self):
        ml_module = _ml_module()
        state_payload = json.dumps(
            {
                "ready": True,
                "recommendations": [],
                "checks": [],
                "resume_guidance": {},
            }
        )

        with (
            patch(
                "definers.ui.apps.train.coach_handlers.inspect_train_coach_request",
                return_value=(
                    state_payload,
                    "summary",
                    "inspection",
                    "validation",
                    "use",
                    "quick",
                    {},
                    {},
                    {},
                ),
            ) as mock_inspect,
            patch(
                "definers.ui.apps.train.coach_handlers.run_train_coach_workflow",
                return_value=(
                    "guided.joblib",
                    "plan",
                    "status",
                    "use-result",
                ),
            ) as mock_run,
        ):
            result = ml_module.auto_train("training.csv")

        self.assertIsInstance(result, ml_module.AutoTrainingResult)
        self.assertEqual(result.artifact_path, "guided.joblib")
        self.assertEqual(result.model_path, "guided.joblib")
        self.assertEqual(str(result), "guided.joblib")
        mock_inspect.assert_called_once()
        self.assertEqual(mock_inspect.call_args.args[0], "files")
        self.assertEqual(mock_inspect.call_args.args[1], "training.csv")
        mock_run.assert_called_once_with(state_payload)

    def test_auto_train_routes_path_inputs_to_guided_by_default(self):
        ml_module = _ml_module()
        state_payload = json.dumps(
            {
                "ready": True,
                "recommendations": [],
                "checks": [],
                "resume_guidance": {},
            }
        )

        with (
            patch(
                "definers.ui.apps.train.coach_handlers.inspect_train_coach_request",
                return_value=(
                    state_payload,
                    "summary",
                    "inspection",
                    "validation",
                    "use",
                    "quick",
                    {},
                    {},
                    {},
                ),
            ),
            patch(
                "definers.ui.apps.train.coach_handlers.run_train_coach_workflow",
                return_value=(
                    "guided.joblib",
                    "plan",
                    "status",
                    "use-result",
                ),
            ),
        ):
            result = ml_module.auto_train("training.csv")

        self.assertIsInstance(result, ml_module.AutoTrainingResult)
        self.assertEqual(result.artifact_path, "guided.joblib")

    def test_auto_train_reports_guided_block_without_training(self):
        ml_module = _ml_module()
        state_payload = json.dumps(
            {
                "ready": False,
                "unresolved_questions": ["Choose the label route."],
                "recommendations": [],
                "checks": [],
                "resume_guidance": {},
            }
        )

        with (
            patch(
                "definers.ui.apps.train.coach_handlers.inspect_train_coach_request",
                return_value=(
                    state_payload,
                    "summary",
                    "inspection",
                    "validation",
                    "use",
                    "quick",
                    {},
                    {},
                    {},
                ),
            ),
            patch(
                "definers.ui.apps.train.coach_handlers.preview_train_coach_plan",
                return_value="blocked plan",
            ),
            patch(
                "definers.ui.apps.train.coach_handlers.run_train_coach_workflow",
            ) as mock_run,
        ):
            result = ml_module.auto_train("training.csv")

        self.assertIsNone(result.artifact_path)
        self.assertEqual(result.plan_markdown, "blocked plan")
        self.assertIn("Choose the label route.", result.status_markdown)
        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
