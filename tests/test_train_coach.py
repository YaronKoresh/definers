import os
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

from definers.ui.apps.train import (
    coach as train_coach,
    coach_handlers,
    coach_manifest,
    coach_ui,
)


def _healthy_snapshot():
    return SimpleNamespace(
        training_ready=True,
        data_preparation_ready=True,
        answer_runtime_ready=True,
    )


def _write_temp_file(suffix: str, content: str) -> str:
    handle = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
        mode="w",
        encoding="utf-8",
        newline="",
    )
    handle.write(content)
    handle.close()
    return handle.name


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _ready_state(**overrides):
    values = {
        "requested_intent": "files",
        "effective_intent": "files",
        "confidence": "high",
        "hosted_runtime": "local",
        "source_mode": "local-tabular",
        "source_type": "parquet",
        "remote_src": None,
        "revision": None,
        "features": ("feature.csv",),
        "labels": (),
        "resume_from": None,
        "save_as": "demo.joblib",
        "column_names": ("text", "label"),
        "detected_file_families": ("tabular",),
        "label_candidates": ("label",),
        "selected_label_columns": ("label",),
        "suggested_batch_size": 16,
        "suggested_validation_split": 0.1,
        "suggested_test_split": 0.1,
        "order_by": "shuffle",
        "stratify": "label",
        "selected_rows": None,
        "row_count": 200,
        "unresolved_questions": (),
        "warnings": (),
        "notes": (),
        "recommendations": (),
        "resume_guidance": train_coach.TrainCoachResumeGuidance(
            strategy="none",
            confidence="high",
            detail="No previous model artifact was supplied.",
            use_resume_artifact=False,
            manifest_path=None,
            revalidation=(),
        ),
        "resolving_question": None,
        "checks": (),
        "ready": True,
    }
    values.update(overrides)
    return train_coach.TrainCoachState(**values)


def test_train_coach_entry_intents_are_stable():
    assert tuple(
        entry["id"] for entry in train_coach.train_coach_entry_intents()
    ) == ("files", "dataset", "resume")
    assert train_coach.train_coach_step_names() == (
        "Upload Or Connect",
        "Inspect And Confirm",
        "Review Plan",
        "Train",
        "Use Result",
    )


def test_build_train_coach_contract_mentions_safe_fallback():
    contract = train_coach.build_train_coach_contract_markdown().lower()

    assert "i have files" in contract
    assert "i have a dataset" in contract
    assert "continue yesterday's model" in contract
    assert "definers start train" in contract


def test_build_train_coach_state_infers_single_tabular_route():
    rows = ["text,label,fold"]
    for index in range(24):
        label = "greeting" if index % 2 == 0 else "farewell"
        rows.append(f"sample-{index},{label},train")
    csv_path = _write_temp_file(
        ".csv",
        "\n".join(rows) + "\n",
    )
    try:
        state = train_coach.build_train_coach_state(
            requested_intent="files",
            uploaded_files=[csv_path],
            remote_src=None,
            resume_artifact=None,
            revision=None,
            save_as="demo.joblib",
            health_snapshot=_healthy_snapshot(),
        )

        assert state.ready is True
        assert state.effective_intent == "files"
        assert state.source_mode == "local-tabular"
        assert tuple(_normalized_path(path) for path in state.features) == (
            _normalized_path(csv_path),
        )
        assert state.labels == ()
        assert state.label_candidates[0] == "label"
        assert state.selected_label_columns == ("label",)
        assert state.stratify == "label"
        recommendation_map = {
            recommendation.name: recommendation
            for recommendation in state.recommendations
        }
        assert recommendation_map["label_columns"].confidence == "high"
        assert recommendation_map["label_columns"].reason
        assert recommendation_map["batch_size"].applied is True
        assert recommendation_map["validation_split"].reason
        assert recommendation_map["test_split"].reason
    finally:
        os.unlink(csv_path)


def test_build_train_coach_state_blocks_ambiguous_multiple_tabular_files():
    first_path = _write_temp_file(
        ".csv",
        "text,score\nhello,1\nbye,2\n",
    )
    second_path = _write_temp_file(
        ".csv",
        "prompt,value\na,3\nb,4\n",
    )
    try:
        state = train_coach.build_train_coach_state(
            requested_intent="files",
            uploaded_files=[first_path, second_path],
            remote_src=None,
            resume_artifact=None,
            revision=None,
            save_as="demo.joblib",
            health_snapshot=_healthy_snapshot(),
        )

        assert state.ready is False
        assert state.source_mode == "local-tabular"
        assert state.resolving_question is not None
        assert state.resolving_question.question_id == "tabular-label-source"
        assert "contains the labels" in state.resolving_question.prompt
        assert state.unresolved_questions == (state.resolving_question.prompt,)
    finally:
        os.unlink(first_path)
        os.unlink(second_path)


def test_build_train_coach_state_uses_remote_dataset_preview(monkeypatch):
    monkeypatch.setattr(
        "definers.data.loaders.fetch_dataset",
        lambda src, source_type, revision, sample_rows=None: [
            {"text": "hello", "label": "greeting"},
            {"text": "bye", "label": "farewell"},
        ],
    )

    state = train_coach.build_train_coach_state(
        requested_intent="dataset",
        uploaded_files=None,
        remote_src="owner/dataset",
        resume_artifact=None,
        revision="main",
        save_as="remote.joblib",
        health_snapshot=_healthy_snapshot(),
    )

    assert state.ready is True
    assert state.effective_intent == "dataset"
    assert state.source_mode == "remote-dataset"
    assert state.label_candidates == ("label",)
    assert state.selected_label_columns == ("label",)
    assert state.row_count == 2


def test_build_train_coach_state_marks_large_remote_sampling_as_optional(
    monkeypatch,
):
    monkeypatch.setattr(
        "definers.data.loaders.fetch_dataset",
        lambda src, source_type, revision, sample_rows=None: {
            "text": [f"sample-{index}" for index in range(120000)],
            "label": [
                "a" if index % 2 == 0 else "b" for index in range(120000)
            ],
        },
    )

    state = train_coach.build_train_coach_state(
        requested_intent="dataset",
        uploaded_files=None,
        remote_src="owner/large-dataset",
        resume_artifact=None,
        revision="main",
        save_as="remote.joblib",
        health_snapshot=_healthy_snapshot(),
    )

    selected_rows_recommendation = next(
        recommendation
        for recommendation in state.recommendations
        if recommendation.name == "selected_rows"
    )
    assert selected_rows_recommendation.applied is False
    assert selected_rows_recommendation.value == "1-25000"
    assert selected_rows_recommendation.confidence == "medium"


def test_build_train_coach_state_asks_hosted_sampling_question(monkeypatch):
    monkeypatch.setattr(
        "definers.ui.apps.train.coach._detect_hosted_runtime",
        lambda: "zerogpu",
    )
    monkeypatch.setattr(
        "definers.data.loaders.fetch_dataset",
        lambda src, source_type, revision, sample_rows=None: {
            "text": [f"sample-{index}" for index in range(120000)],
            "label": [
                "a" if index % 2 == 0 else "b" for index in range(120000)
            ],
        },
    )

    state = train_coach.build_train_coach_state(
        requested_intent="dataset",
        uploaded_files=None,
        remote_src="owner/large-dataset",
        resume_artifact=None,
        revision="main",
        save_as="remote.joblib",
        health_snapshot=_healthy_snapshot(),
    )

    assert state.ready is False
    assert state.hosted_runtime == "zerogpu"
    assert state.resolving_question is not None
    assert state.resolving_question.question_id == "hosted-sampling"
    assert state.selected_rows is None
    assert any("ZeroGPU" in note for note in state.notes)


def test_build_train_coach_state_reads_resume_manifest_and_starts_fresh_when_incompatible(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))
    artifact_path = tmp_path / "artifact.joblib"
    artifact_path.write_text("model", encoding="utf-8")
    coach_manifest.write_train_session_manifest(
        normalized_request={"source_type": "parquet"},
        inspection_report={
            "source_mode": "local-image",
            "source_type": "parquet",
            "selected_label_columns": [],
        },
        recommendations=(),
        plan_markdown="plan",
        artifact_path=str(artifact_path),
        status_markdown="status",
        resume_strategy="safe-continue",
    )
    csv_path = _write_temp_file(
        ".csv",
        "text,label\nhello,greeting\nbye,farewell\n",
    )
    try:
        state = train_coach.build_train_coach_state(
            requested_intent="resume",
            uploaded_files=[csv_path],
            remote_src=None,
            resume_artifact=str(artifact_path),
            revision=None,
            save_as="fresh.joblib",
            health_snapshot=_healthy_snapshot(),
        )
    finally:
        os.unlink(csv_path)

    assert state.resume_guidance.strategy == "fresh-start"
    assert state.resume_guidance.use_resume_artifact is False
    assert state.ready is True
    assert any(
        recommendation.name == "resume_strategy"
        and recommendation.applied is False
        for recommendation in state.recommendations
    )


def test_build_train_coach_state_blocks_resume_without_new_data():
    model_path = _write_temp_file(".joblib", "placeholder")
    try:
        state = train_coach.build_train_coach_state(
            requested_intent="resume",
            uploaded_files=None,
            remote_src=None,
            resume_artifact=model_path,
            revision=None,
            save_as="resume.joblib",
            health_snapshot=_healthy_snapshot(),
        )

        assert state.ready is False
        assert state.source_mode == "resume-only"
        assert any(
            "fresh data" in check.detail.lower()
            for check in state.checks
            if not check.ok
        )
    finally:
        os.unlink(model_path)


def test_inspect_train_coach_request_enables_next_actions(monkeypatch):
    csv_path = _write_temp_file(
        ".csv",
        "text,label\nhello,greeting\nbye,farewell\n",
    )
    fake_gradio = ModuleType("gradio")
    fake_gradio.update = lambda **kwargs: kwargs
    monkeypatch.setitem(__import__("sys").modules, "gradio", fake_gradio)
    try:
        (
            state_payload,
            summary,
            inspection,
            validation,
            use_result,
            quick_decision,
            resolving_choice_update,
            preview_update,
            train_update,
        ) = coach_handlers.inspect_train_coach_request(
            "files",
            [csv_path],
            None,
            None,
            None,
            "demo.joblib",
        )

        assert state_payload
        assert "Guided Intake" in summary
        assert "Data Inspection" in inspection
        assert "Guided Validation" in validation
        assert "Use Result" in use_result
        assert "no clarification is needed" in quick_decision.lower()
        assert resolving_choice_update == {
            "choices": [],
            "visible": False,
            "label": "Quick Decision",
            "value": None,
        }
        assert preview_update == {"interactive": True}
        assert train_update == {"interactive": True}
    finally:
        os.unlink(csv_path)


def test_inspect_train_coach_request_exposes_quick_decision_for_ambiguous_tabular(
    monkeypatch,
):
    first_path = _write_temp_file(
        ".csv",
        "text,score\nhello,1\nbye,2\n",
    )
    second_path = _write_temp_file(
        ".csv",
        "prompt,value\na,3\nb,4\n",
    )
    fake_gradio = ModuleType("gradio")
    fake_gradio.update = lambda **kwargs: kwargs
    monkeypatch.setitem(__import__("sys").modules, "gradio", fake_gradio)
    try:
        result = coach_handlers.inspect_train_coach_request(
            "files",
            [first_path, second_path],
            None,
            None,
            None,
            "demo.joblib",
        )
    finally:
        os.unlink(first_path)
        os.unlink(second_path)

    assert "Which file contains the labels?" in result[5]
    assert result[6]["visible"] is True
    assert result[6]["label"] == "Which file contains the labels?"
    assert result[6]["value"] == "first-file-labels"
    assert [choice[1] for choice in result[6]["choices"]] == [
        "first-file-labels",
        "second-file-labels",
        "review-manually",
    ]
    assert any("Use" in choice[0] for choice in result[6]["choices"][:2])
    assert result[7] == {"interactive": False}
    assert result[8] == {"interactive": False}


def test_preview_train_coach_plan_delegates_to_train_handlers(monkeypatch):
    state = _ready_state()
    monkeypatch.setattr(
        "definers.ui.apps.train.handlers.build_training_plan_markdown",
        lambda *args, **kwargs: "plan-preview",
    )

    result = coach_handlers.preview_train_coach_plan(
        train_coach.train_coach_state_json(state)
    )

    assert result == "plan-preview"


def test_preview_train_coach_plan_ignores_incompatible_resume(monkeypatch):
    captured = {}
    state = _ready_state(
        requested_intent="resume",
        effective_intent="resume",
        resume_from="previous.joblib",
        resume_guidance=train_coach.TrainCoachResumeGuidance(
            strategy="fresh-start",
            confidence="high",
            detail="Start a fresh model.",
            use_resume_artifact=False,
            manifest_path="session.json",
            revalidation=("source mode",),
        ),
    )

    def _fake_build_training_plan_markdown(*args, **kwargs):
        captured["resume_model"] = args[2]
        captured["progress_profile"] = kwargs.get("progress_profile")
        return "plan-preview"

    monkeypatch.setattr(
        "definers.ui.apps.train.handlers.build_training_plan_markdown",
        _fake_build_training_plan_markdown,
    )

    result = coach_handlers.preview_train_coach_plan(
        train_coach.train_coach_state_json(state)
    )

    assert result == "plan-preview"
    assert captured == {
        "resume_model": None,
        "progress_profile": "guided",
    }


def test_run_train_coach_workflow_delegates_to_train_handlers(
    monkeypatch, tmp_path
):
    state = _ready_state(
        requested_intent="dataset",
        effective_intent="dataset",
        source_mode="remote-dataset",
        remote_src="owner/dataset",
        revision="main",
        features=(),
        detected_file_families=(),
        suggested_batch_size=32,
        row_count=1000,
        recommendations=(
            train_coach.TrainCoachRecommendation(
                name="label_columns",
                value=("label",),
                reason="Detected label column.",
                confidence="high",
                applied=True,
            ),
        ),
    )
    captured = {}
    artifact_path = tmp_path / "trained.joblib"
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    def _fake_handle_training(*args, **kwargs):
        captured["progress_profile"] = kwargs.get("progress_profile")
        return (
            str(artifact_path),
            "plan-preview",
            "## Training\n- Status: ready",
        )

    monkeypatch.setattr(
        "definers.ui.apps.train.handlers.handle_training",
        _fake_handle_training,
    )

    result = coach_handlers.run_train_coach_workflow(
        train_coach.train_coach_state_json(state)
    )

    loaded_manifest = coach_manifest.load_train_artifact_manifest(
        str(artifact_path)
    )

    assert captured == {"progress_profile": "guided"}
    assert result[0] == str(artifact_path)
    assert result[1] == "plan-preview"
    assert "## Training\n- Status: ready" in result[2]
    assert "Use Result" in result[3]
    assert loaded_manifest is not None
    assert loaded_manifest["artifact_path"] == str(artifact_path)


def test_write_train_session_manifest_persists_session_and_artifact_sidecar(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))
    artifact_path = tmp_path / "train" / "guided.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("model", encoding="utf-8")

    manifest = coach_manifest.write_train_session_manifest(
        normalized_request={"source_type": "parquet", "batch_size": 16},
        inspection_report={"source_mode": "remote-dataset"},
        recommendations=(
            {
                "name": "batch_size",
                "value": 16,
                "reason": "Safe first pass.",
                "confidence": "high",
                "applied": True,
            },
        ),
        plan_markdown="## Training Plan\n- Mode: remote-dataset",
        artifact_path=str(artifact_path),
        status_markdown="## Training\n- Status: ready",
        resume_strategy="safe-continue",
    )

    session_manifest_path = Path(manifest["session_manifest_path"])
    artifact_manifest_path = Path(manifest["artifact_manifest_path"])
    loaded_manifest = coach_manifest.load_train_artifact_manifest(
        str(artifact_path)
    )
    result_markdown = coach_manifest.render_train_result_markdown(manifest)

    assert session_manifest_path.exists()
    assert artifact_manifest_path.exists()
    assert loaded_manifest is not None
    assert loaded_manifest["artifact_path"] == str(artifact_path)
    assert "Run prediction" in result_markdown
    assert str(session_manifest_path) in result_markdown


def test_record_train_rollout_event_writes_scannable_metrics(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    payload = coach_manifest.record_train_rollout_event(
        event="inspection",
        inspection_report={
            "hosted_runtime": "zerogpu",
            "source_mode": "remote-dataset",
            "checks": [{"name": "Guided Route Resolved", "ok": False}],
            "resolving_question": {"question_id": "hosted-sampling"},
            "row_count": 120000,
        },
        recommendations=(
            {"name": "batch_size", "applied": True},
            {"name": "selected_rows", "applied": False},
        ),
        resume_strategy="safe-continue",
        guided_flow_completed=False,
        resolving_question_answered=True,
    )
    scanned_metrics = coach_manifest.scan_train_rollout_metrics()

    assert payload["rollout_metrics"]["hosted_runtime"] == "zerogpu"
    assert payload["rollout_metrics"]["accepted_recommendations"] == [
        "batch_size"
    ]
    assert payload["rollout_metrics"]["resolving_question_answered"] is True
    assert scanned_metrics[-1]["source_mode"] == "remote-dataset"
    assert (
        scanned_metrics[-1]["validation_failure_category"]
        == "Guided Route Resolved"
    )


class _FakeComponent:
    def __init__(self, registry, kind, *args, **kwargs):
        self.registry = registry
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        self.click_calls = []
        self.change_calls = []
        registry.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def click(self, fn=None, inputs=None, outputs=None, **kwargs):
        self.click_calls.append((fn, inputs, outputs, kwargs))
        return self

    def change(self, fn=None, inputs=None, outputs=None, **kwargs):
        self.change_calls.append((fn, inputs, outputs, kwargs))
        return self


def _build_fake_gradio_module(registry):
    fake_gradio = ModuleType("gradio")
    fake_gradio.update = lambda **kwargs: kwargs
    component_names = [
        "Button",
        "Column",
        "File",
        "HTML",
        "Markdown",
        "Radio",
        "Row",
        "Textbox",
    ]
    for name in component_names:
        setattr(
            fake_gradio,
            name,
            lambda *args, _name=name, **kwargs: _FakeComponent(
                registry,
                _name,
                *args,
                **kwargs,
            ),
        )
    return fake_gradio


def test_build_train_guided_mode_constructs_expected_actions(monkeypatch):
    registry = []
    fake_gradio = _build_fake_gradio_module(registry)
    monkeypatch.setitem(__import__("sys").modules, "gradio", fake_gradio)
    bind_calls = []

    coach_ui.build_train_guided_mode(
        bind_action=lambda button, handler, **kwargs: bind_calls.append(
            (
                button.args[0] if button.args else None,
                getattr(handler, "__name__", ""),
                kwargs.get("action_label"),
            )
        )
    )
    buttons = [
        component for component in registry if component.kind == "Button"
    ]
    button_labels = [
        component.args[0] for component in buttons if component.args
    ]
    assert "Inspect My Inputs" in button_labels
    assert "Review Guided Plan" in button_labels
    assert "Train With Guided Defaults" in button_labels
    assert bind_calls == [
        ("Inspect My Inputs", "inspect_train_coach_request", "Inspect Inputs"),
        (
            "Review Guided Plan",
            "preview_train_coach_plan",
            "Review Guided Plan",
        ),
        (
            "Train With Guided Defaults",
            "run_train_coach_workflow",
            "Train With Guided Defaults",
        ),
    ]
