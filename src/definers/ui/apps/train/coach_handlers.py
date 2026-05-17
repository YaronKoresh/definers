from __future__ import annotations

import json
from pathlib import Path

from .coach import (
    build_train_coach_state,
    parse_train_coach_state,
    render_train_coach_inspection_markdown,
    render_train_coach_summary_markdown,
    render_train_coach_validation_markdown,
    train_coach_state_json,
)


def _last_model_state_path() -> Path:
    state_dir = Path.home() / ".definers"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "last_trained_model.json"


def _write_last_model_path(model_path: str) -> None:
    try:
        with open(_last_model_state_path(), "w", encoding="utf-8") as f:
            json.dump({"last_model_path": str(model_path)}, f)
    except Exception:
        pass


def _read_last_model_path() -> str | None:
    try:
        state_file = _last_model_state_path()
        if not state_file.exists():
            return None
        with open(state_file, encoding="utf-8") as f:
            data = json.load(f)
        path = data.get("last_model_path")
        if path and Path(path).exists():
            return path
        return None
    except Exception:
        return None


def strip_saved_model(model_path: str | None) -> str | None:
    if not model_path:
        return None
    try:
        import joblib

        model = joblib.load(model_path)
        if hasattr(model, "strip"):
            model.strip()
        joblib.dump(model, model_path)
    except Exception:
        pass
    return model_path


def _interactive_update(interactive: bool):
    try:
        import gradio as gr

        return gr.update(interactive=interactive)
    except Exception:
        return {"interactive": interactive}


def _resolving_choice_update(question):
    choices = []
    visible = False
    label = "Quick Decision"
    value = None
    if question is not None:
        choices = list(zip(question.option_labels, question.option_values))
        visible = True
        label = question.prompt
        value = question.default_value
    try:
        import gradio as gr

        return gr.update(
            choices=choices,
            visible=visible,
            label=label,
            value=value,
        )
    except Exception:
        return {
            "choices": choices,
            "visible": visible,
            "label": label,
            "value": value,
        }


def _resolving_question_markdown(state) -> str:
    question = getattr(state, "resolving_question", None)
    if question is None:
        if getattr(state, "unresolved_questions", None):
            return "## Quick Decision\n- Status: clarification is pending."
        return "## Quick Decision\n- Status: no clarification is needed."
    lines = [
        "## Quick Decision",
        f"- Question: {question.prompt}",
    ]
    for option_label in question.option_labels:
        lines.append(f"- Option: {option_label}")
    lines.append("- Next Step: choose one option and inspect again.")
    return "\n".join(lines)


def _use_result_placeholder() -> str:
    return (
        "## Use Result\n"
        "- Status: train a model to unlock the saved session, manifest, and next actions."
    )


def _resume_artifact_for_training(state) -> str | None:
    if state.resume_guidance.use_resume_artifact:
        return state.resume_from
    return None


def reset_train_coach_state():
    return (
        "",
        "## Guided Intake\n- Status: waiting for inspection.",
        "## Data Inspection\n- Status: add files or a dataset, then run inspection.",
        "## Guided Validation\n- Status: inspection pending.",
        _use_result_placeholder(),
        "## Quick Decision\n- Status: no clarification is needed.",
        _resolving_choice_update(None),
        _interactive_update(False),
        _interactive_update(False),
    )


def inspect_train_coach_request(
    requested_intent,
    uploaded_files,
    remote_src,
    revision,
    resume_artifact,
    save_as,
    resolving_choice=None,
):
    from definers.system.download_activity import create_activity_reporter

    from .coach_manifest import record_train_rollout_event

    report = create_activity_reporter(3)
    state = build_train_coach_state(
        requested_intent=requested_intent,
        uploaded_files=uploaded_files,
        remote_src=remote_src,
        resume_artifact=resume_artifact,
        revision=revision,
        resolving_choice=resolving_choice,
        save_as=save_as,
        activity_reporter=report,
    )
    inspection_report = json.loads(train_coach_state_json(state))
    record_train_rollout_event(
        event="inspection",
        inspection_report=inspection_report,
        recommendations=tuple(inspection_report.get("recommendations", ())),
        resume_strategy=state.resume_guidance.strategy,
        guided_flow_completed=False,
        resolving_question_answered=bool(
            resolving_choice and state.resolving_question is None
        ),
    )
    return (
        json.dumps(inspection_report, ensure_ascii=False),
        render_train_coach_summary_markdown(state),
        render_train_coach_inspection_markdown(state),
        render_train_coach_validation_markdown(state),
        _use_result_placeholder(),
        _resolving_question_markdown(state),
        _resolving_choice_update(state.resolving_question),
        _interactive_update(state.ready),
        _interactive_update(state.ready),
    )


def preview_train_coach_plan(state_payload):
    from .handlers import build_training_plan_markdown

    state = parse_train_coach_state(state_payload)
    if state is None:
        return "## Training Plan\n- No guided state is available yet."
    if not state.ready:
        if state.resolving_question is not None:
            return (
                "## Training Plan\n"
                f"- Guided mode needs one quick decision before it can build the plan: {state.resolving_question.prompt}"
            )
        return (
            "## Training Plan\n"
            "- Guided mode is blocked. Inspect the validation panel or switch to `definers start train`."
        )
    return build_training_plan_markdown(
        list(state.features) or None,
        list(state.labels) or None,
        _resume_artifact_for_training(state),
        state.remote_src,
        list(state.selected_label_columns) or None,
        state.revision,
        state.source_type,
        None,
        state.selected_rows,
        state.suggested_batch_size,
        state.suggested_validation_split,
        state.suggested_test_split,
        state.order_by,
        state.stratify,
        progress_profile="guided",
    )


def run_train_coach_workflow(state_payload):
    from definers.system.download_activity import report_download_activity

    from .coach_manifest import (
        record_train_rollout_event,
        render_train_result_markdown,
        write_train_session_manifest,
    )
    from .handlers import _build_training_request, handle_training

    state = parse_train_coach_state(state_payload)
    if state is None:
        return (
            None,
            "## Training Plan\n- No guided state is available yet.",
            "## Training\n- Status: blocked\n- Inspect your inputs before training.",
            _use_result_placeholder(),
            None,
        )
    if not state.ready:
        return (
            None,
            "## Training Plan\n- Guided mode is blocked. Inspect the validation panel or finish the quick decision before training.",
            "## Training\n- Status: blocked\n- Guided validation did not pass.",
            _use_result_placeholder(),
            None,
        )
    resume_for_training = _resume_artifact_for_training(state)
    normalized_request = _build_training_request(
        list(state.features) or None,
        list(state.labels) or None,
        resume_for_training,
        state.remote_src,
        list(state.selected_label_columns) or None,
        state.revision,
        state.source_type,
        None,
        state.selected_rows,
        state.suggested_batch_size,
        state.suggested_validation_split,
        state.suggested_test_split,
        state.order_by,
        state.stratify,
        save_as=state.save_as,
    )
    model_output, plan_markdown, status_markdown = handle_training(
        list(state.features) or None,
        list(state.labels) or None,
        resume_for_training,
        state.remote_src,
        list(state.selected_label_columns) or None,
        state.revision,
        state.source_type,
        None,
        state.selected_rows,
        state.save_as,
        state.suggested_batch_size,
        state.suggested_validation_split,
        state.suggested_test_split,
        state.order_by,
        state.stratify,
        progress_profile="guided",
    )
    if not model_output:
        return (
            model_output,
            plan_markdown,
            status_markdown,
            _use_result_placeholder(),
            None,
        )
    report_download_activity(
        "Suggest next actions",
        detail="Writing the train session manifest and guided next steps.",
        phase="step",
        completed=6,
        total=6,
    )
    inspection_report = json.loads(train_coach_state_json(state))
    rollout_event = record_train_rollout_event(
        event="training",
        inspection_report=inspection_report,
        recommendations=tuple(inspection_report.get("recommendations", ())),
        resume_strategy=state.resume_guidance.strategy,
        guided_flow_completed=bool(model_output),
        artifact_path=model_output,
    )
    manifest = write_train_session_manifest(
        normalized_request=normalized_request,
        inspection_report=inspection_report,
        recommendations=tuple(inspection_report.get("recommendations", ())),
        plan_markdown=plan_markdown,
        artifact_path=model_output,
        status_markdown=status_markdown,
        resume_strategy=state.resume_guidance.strategy,
        rollout_metrics=rollout_event.get("rollout_metrics"),
    )
    try:
        from definers.system.output_paths import cleanup_managed_output_root
        from definers.system.runtime_budget import (
            should_cleanup_after_guided_training,
        )

        if should_cleanup_after_guided_training(state.hosted_runtime):
            report_download_activity(
                "Cleanup hosted session",
                detail="Pruning managed guided session outputs for the hosted runtime.",
                phase="step",
                completed=6,
                total=6,
            )
            cleanup_managed_output_root()
    except Exception:
        pass
    status_markdown = (
        status_markdown
        + f"\n- Session Manifest: {manifest['session_manifest_path']}"
    )
    if model_output:
        _write_last_model_path(model_output)
    return (
        model_output,
        plan_markdown,
        status_markdown,
        render_train_result_markdown(manifest),
        model_output,
    )


def run_train_coach_auto_workflow(
    requested_intent,
    uploaded_files,
    remote_src,
    revision,
    resume_artifact,
    save_as,
    resolving_choice=None,
):
    inspected = inspect_train_coach_request(
        requested_intent,
        uploaded_files,
        remote_src,
        revision,
        resume_artifact,
        save_as,
        resolving_choice=resolving_choice,
    )
    state_payload = inspected[0]
    state = parse_train_coach_state(state_payload)
    if state is not None and state.ready:
        train_output, training_plan, training_status, use_result, last_model = (
            run_train_coach_workflow(state_payload)
        )
    else:
        train_output = None
        last_model = None
        training_plan = preview_train_coach_plan(state_payload)
        training_status = (
            "## Training\n"
            "- Status: blocked\n"
            "- Guided validation needs one safe route before training starts."
        )
        use_result = inspected[4]
    return (
        inspected[0],
        inspected[1],
        inspected[2],
        inspected[3],
        use_result,
        inspected[5],
        inspected[6],
        inspected[7],
        inspected[8],
        train_output,
        training_plan,
        training_status,
        last_model,
    )


__all__ = [
    "inspect_train_coach_request",
    "preview_train_coach_plan",
    "reset_train_coach_state",
    "run_train_coach_auto_workflow",
    "run_train_coach_workflow",
    "strip_saved_model",
    "_read_last_model_path",
]
