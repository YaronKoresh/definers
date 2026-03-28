from types import SimpleNamespace

import pytest

from definers.presentation.apps.train_handlers import (
    build_training_plan_markdown,
    handle_prediction,
    handle_training,
    normalize_selected_rows,
)


def test_normalize_selected_rows_preserves_clean_value(monkeypatch):
    import definers.ml as ml_module

    monkeypatch.setattr(ml_module, "simple_text", lambda value: value.strip())

    assert normalize_selected_rows(" 1-3 5 ") == "1-3 5"


def test_build_training_plan_markdown_uses_auto_trainer_plan(monkeypatch):
    import definers.ml as ml_module

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def training_plan(self, **kwargs):
            return SimpleNamespace(mode="remote-dataset", source_summary="demo")

    monkeypatch.setattr(ml_module, "AutoTrainer", FakeTrainer)
    monkeypatch.setattr(ml_module, "simple_text", lambda value: value)
    monkeypatch.setattr(
        "definers.application_ml.trainer_plan.render_training_plan_markdown",
        lambda plan: f"plan:{plan.mode}:{plan.source_summary}",
    )

    result = build_training_plan_markdown(
        None,
        "label",
        None,
        "owner/dataset",
        "label",
        "main",
        "parquet",
        None,
        "1-5",
    )

    assert result == "plan:remote-dataset:demo"


def test_handle_training_returns_model_output_and_plan(monkeypatch):
    import definers.ml as ml_module

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def training_plan(self, **kwargs):
            return SimpleNamespace(mode="file-dataset", source_summary="features.csv")

        def train(self, **kwargs):
            assert kwargs["resume_from"] == "model.joblib"
            return "trained.joblib"

    monkeypatch.setattr(ml_module, "AutoTrainer", FakeTrainer)
    monkeypatch.setattr(ml_module, "simple_text", lambda value: value)
    monkeypatch.setattr(
        "definers.application_ml.trainer_plan.render_training_plan_markdown",
        lambda plan: f"plan:{plan.mode}:{plan.source_summary}",
    )

    model_output, plan = handle_training(
        ["features.csv"],
        ["labels.csv"],
        "model.joblib",
        None,
        "label",
        "main",
        "csv",
        None,
        "1-5",
    )

    assert model_output == "trained.joblib"
    assert plan == "plan:file-dataset:features.csv"


def test_handle_prediction_uses_model_path(monkeypatch):
    import definers.ml as ml_module

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, data):
            assert data == "predict.csv"
            return "predictions.csv"

    monkeypatch.setattr(ml_module, "AutoTrainer", FakeTrainer)

    assert handle_prediction("model.joblib", "predict.csv") == "predictions.csv"


def test_normalize_selected_rows_rejects_too_many_spaces(monkeypatch):
    import definers.ml as ml_module

    monkeypatch.setattr(ml_module, "simple_text", lambda value: value)

    with pytest.raises(Exception):
        normalize_selected_rows("1" + " " * 20 + "2")