import importlib
from types import SimpleNamespace

import pytest
from definers.ui.apps.train_handlers import (
    build_training_plan_markdown,
    handle_answer,
    handle_features_to_text,
    handle_inference,
    handle_init_model_files,
    handle_iterative_summary,
    handle_kmeans_suggestions,
    handle_language_lookup,
    handle_load_runtime_model,
    handle_map_reduce_summary,
    handle_ml_health_report,
    handle_prediction,
    handle_prompt_optimization,
    handle_quick_summary,
    handle_rvc_checkpoint_lookup,
    handle_text_feature_extraction,
    handle_training,
    handle_validate_ml_health,
    normalize_selected_rows,
)


def test_normalize_selected_rows_preserves_clean_value(monkeypatch):
    import definers.ml as ml_module

    monkeypatch.setattr(ml_module, "simple_text", lambda value: value.strip())

    assert normalize_selected_rows(" 1-3 5 ") == "1-3 5"


def test_build_training_plan_markdown_uses_auto_trainer_plan(monkeypatch):
    import definers.ml as ml_module

    trainer_plan_module = importlib.import_module("definers.ml.trainer_plan")

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def training_plan(self, **kwargs):
            return SimpleNamespace(mode="remote-dataset", source_summary="demo")

    monkeypatch.setattr(ml_module, "AutoTrainer", FakeTrainer)
    monkeypatch.setattr(ml_module, "simple_text", lambda value: value)
    monkeypatch.setattr(
        trainer_plan_module,
        "render_training_plan_markdown",
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
        16,
        0.1,
        0.2,
        "shuffle",
        "label",
    )

    assert result == "plan:remote-dataset:demo"


def test_handle_training_returns_model_output_and_plan(monkeypatch):
    import definers.ml as ml_module

    trainer_plan_module = importlib.import_module("definers.ml.trainer_plan")

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def training_plan(self, **kwargs):
            return SimpleNamespace(
                mode="file-dataset", source_summary="features.csv"
            )

        def train(self, **kwargs):
            assert kwargs["resume_from"] == "model.joblib"
            assert kwargs["save_as"] == "trained-output.joblib"
            assert kwargs["batch_size"] == 24
            assert kwargs["validation_split"] == 0.15
            assert kwargs["test_split"] == 0.2
            assert kwargs["order_by"] == "shuffle"
            assert kwargs["stratify"] == "label"
            return "trained.joblib"

    monkeypatch.setattr(ml_module, "AutoTrainer", FakeTrainer)
    monkeypatch.setattr(ml_module, "simple_text", lambda value: value)
    monkeypatch.setattr(
        trainer_plan_module,
        "render_training_plan_markdown",
        lambda plan: f"plan:{plan.mode}:{plan.source_summary}",
    )

    model_output, plan, status = handle_training(
        ["features.csv"],
        ["labels.csv"],
        "model.joblib",
        None,
        "label",
        "main",
        "csv",
        None,
        "1-5",
        "trained-output.joblib",
        24,
        0.15,
        0.2,
        "shuffle",
        "label",
    )

    assert model_output == "trained.joblib"
    assert plan == "plan:file-dataset:features.csv"
    assert "Artifact: trained.joblib" in status


def test_handle_prediction_uses_model_path(monkeypatch):
    import definers.ml as ml_module

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, data):
            assert data == "predict.csv"
            return "predictions.csv"

    monkeypatch.setattr(ml_module, "AutoTrainer", FakeTrainer)

    artifact, status, preview = handle_prediction(
        "model.joblib",
        "predict.csv",
        None,
    )

    assert artifact == "predictions.csv"
    assert "Artifact: predictions.csv" in status
    assert preview == "predictions.csv"


def test_handle_prediction_accepts_in_memory_payload(monkeypatch):
    import definers.ml as ml_module

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, data):
            assert data == [[1, 2], [3, 4]]
            return [0, 1]

    monkeypatch.setattr(ml_module, "AutoTrainer", FakeTrainer)

    artifact, status, preview = handle_prediction(
        "model.joblib",
        None,
        "[[1, 2], [3, 4]]",
    )

    assert artifact is None
    assert "Result returned inline" in status
    assert preview == "[\n  0,\n  1\n]"


def test_normalize_selected_rows_rejects_too_many_spaces(monkeypatch):
    import definers.ml as ml_module

    monkeypatch.setattr(ml_module, "simple_text", lambda value: value)

    with pytest.raises(Exception):
        normalize_selected_rows("1" + " " * 20 + "2")


def test_handle_inference_uses_task(monkeypatch):
    import definers.ml as ml_module

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def infer(self, data, task, model_type=None):
            assert data == "sample.txt"
            assert task == "answer"
            assert model_type is None
            return "inference.txt"

    monkeypatch.setattr(ml_module, "AutoTrainer", FakeTrainer)

    artifact, status, preview = handle_inference("answer", "sample.txt", "auto")

    assert artifact == "inference.txt"
    assert "Artifact: inference.txt" in status
    assert preview == "inference.txt"


def test_handle_answer_initializes_runtime(monkeypatch):
    import definers.ml as ml_module

    init_calls = []

    monkeypatch.setattr(
        ml_module,
        "init_pretrained_model",
        lambda task, turbo: init_calls.append((task, turbo)),
    )
    monkeypatch.setattr(
        ml_module,
        "answer",
        lambda history: f"messages:{len(history)}",
    )

    result = handle_answer("hello", "[]", None)

    assert result == "messages:1"
    assert init_calls == [("answer", True)]


def test_handle_text_feature_extraction_serializes_output(monkeypatch):
    import definers.ml as ml_module

    monkeypatch.setattr(
        ml_module,
        "extract_text_features",
        lambda text: [[1, 0, 1], [0, 1, 0]],
    )

    payload, summary = handle_text_feature_extraction("hello world")

    assert '"' not in payload
    assert "Rows: 2" in summary
    assert "Columns: 3" in summary


def test_handle_features_to_text_uses_vocabulary(monkeypatch):
    import definers.ml as ml_module

    monkeypatch.setattr(
        ml_module,
        "features_to_text",
        lambda features, vocabulary=None: f"{features}:{vocabulary}",
    )

    result = handle_features_to_text("[[1, 0, 1]]", "alpha, beta, gamma")

    assert result == "[[1, 0, 1]]:['alpha', 'beta', 'gamma']"


def test_summary_handlers_delegate_to_ml(monkeypatch):
    import definers.ml as ml_module

    init_calls = []
    monkeypatch.setattr(
        ml_module,
        "init_pretrained_model",
        lambda task, turbo: init_calls.append((task, turbo)),
    )
    monkeypatch.setattr(ml_module, "summarize", lambda text: f"quick:{text}")
    monkeypatch.setattr(
        ml_module,
        "map_reduce_summary",
        lambda text, max_words: f"map:{text}:{max_words}",
    )
    monkeypatch.setattr(
        ml_module,
        "summary",
        lambda text, max_words, min_loops: (
            f"iter:{text}:{max_words}:{min_loops}"
        ),
    )

    assert handle_quick_summary("hello") == "quick:hello"
    assert handle_map_reduce_summary("hello", 20) == "map:hello:20"
    assert handle_iterative_summary("hello", 15, 2) == "iter:hello:15:2"
    assert init_calls == [
        ("summary", True),
        ("summary", True),
        ("summary", True),
    ]


def test_handle_prompt_optimization_bootstraps_translate_and_summary(
    monkeypatch,
):
    import definers.ml as ml_module

    init_calls = []
    monkeypatch.setattr(
        ml_module,
        "init_pretrained_model",
        lambda task, turbo: init_calls.append((task, turbo)),
    )
    monkeypatch.setattr(
        ml_module,
        "preprocess_prompt",
        lambda prompt: f"prep:{prompt}",
    )
    monkeypatch.setattr(
        ml_module,
        "optimize_prompt_realism",
        lambda prompt: f"opt:{prompt}",
    )

    assert handle_prompt_optimization("sunrise") == (
        "prep:sunrise",
        "opt:sunrise",
    )
    assert init_calls == [("summary", True), ("translate", True)]


def test_health_handlers_render_runtime_status(monkeypatch):
    import definers.ml as ml_module

    snapshot = SimpleNamespace(training_ready=True, answer_runtime_ready=False)
    monkeypatch.setattr(ml_module, "get_ml_health_snapshot", lambda: snapshot)
    monkeypatch.setattr(ml_module, "ml_health_markdown", lambda: "health")
    monkeypatch.setattr(ml_module, "validate_ml_health", lambda: snapshot)

    report, status = handle_ml_health_report()

    assert report == "health"
    assert "Training Ready: True" in status
    assert "Status: ready" in handle_validate_ml_health()


def test_handle_kmeans_suggestions_formats_result(monkeypatch):
    import definers.ml as ml_module

    monkeypatch.setattr(
        ml_module,
        "kmeans_k_suggestions",
        lambda X, k_range, random_state=None: {
            "suggested_k_elbow": 3,
            "suggested_k_silhouette": 4,
            "final_suggestion": 4,
        },
    )

    summary, payload = handle_kmeans_suggestions("1 2\n3 4\n5 6", 2, 5, 7)

    assert "Final Suggestion: 4" in summary
    assert '"final_suggestion": 4' in payload


def test_lookup_handlers_delegate(monkeypatch):
    import definers.ml as ml_module

    monkeypatch.setattr(
        ml_module,
        "find_latest_rvc_checkpoint",
        lambda folder, name: f"{name}_e2_s40.pth",
    )
    monkeypatch.setattr(ml_module, "lang_code_to_name", lambda code: "English")

    assert (
        "Latest Checkpoint: voice_e2_s40.pth"
        in handle_rvc_checkpoint_lookup("./logs", "voice")
    )
    assert "en -> English" in handle_language_lookup("en")


def test_bootstrap_handlers_delegate(monkeypatch):
    import definers.ml as ml_module

    runtime_calls = []
    monkeypatch.setattr(
        ml_module,
        "init_model_file",
        lambda task, turbo, model_type=None: (
            f"{task}-{model_type or 'auto'}.joblib"
        ),
    )
    monkeypatch.setattr(
        ml_module,
        "init_pretrained_model",
        lambda task, turbo: runtime_calls.append((task, turbo)),
    )

    assert "Artifact: answer-joblib.joblib" in handle_init_model_files(
        "answer", True, "joblib"
    )
    assert "Model loaded into runtime" in handle_load_runtime_model(
        "answer", False
    )
    assert runtime_calls == [("answer", False)]
