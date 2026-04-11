from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from definers.system.output_paths import managed_output_dir
from definers.ui.job_state import (
    create_job_dir,
    manifest_path,
    read_manifest,
    write_manifest,
)

_TRAIN_SESSION_MANIFEST_NAME = "train_session.json"
_TRAIN_ROLLOUT_METRIC_NAME = "guided_rollout.json"


def train_artifact_manifest_name(artifact_path: str) -> str:
    artifact_stem = (
        Path(str(artifact_path or "")).stem.strip() or "trained_model"
    )
    return f"{artifact_stem}.train_manifest.json"


def load_train_artifact_manifest(
    artifact_path: str | None,
) -> dict[str, object] | None:
    normalized_artifact = str(artifact_path or "").strip()
    if not normalized_artifact:
        return None
    artifact_parent = str(Path(normalized_artifact).expanduser().parent)
    try:
        return read_manifest(
            artifact_parent,
            name=train_artifact_manifest_name(normalized_artifact),
        )
    except Exception:
        return None


def build_train_next_actions(
    *,
    artifact_path: str,
    session_manifest_path: str,
    resume_strategy: str,
) -> tuple[dict[str, str], ...]:
    normalized_artifact = str(artifact_path).strip()
    actions = [
        {
            "title": "Run prediction",
            "detail": "Open the Run tab in train and load this artifact into Predict.",
            "route": "train/run/predict",
        },
        {
            "title": "Continue guided training",
            "detail": (
                "Come back to Guided Mode with new files and reuse this artifact to continue training."
                if resume_strategy != "fresh-start"
                else "Start a fresh guided run with new files if this artifact is no longer compatible."
            ),
            "route": "train/guided",
        },
        {
            "title": "Review the saved session",
            "detail": f"Reopen the saved train session manifest at {session_manifest_path}.",
            "route": "train/session",
        },
        {
            "title": "Open the artifact folder",
            "detail": f"The trained artifact was saved at {normalized_artifact}.",
            "route": "train/outputs",
        },
    ]
    return tuple(actions)


def _validation_failure_category(
    inspection_report: Mapping[str, object],
) -> str | None:
    checks = inspection_report.get("checks", ())
    if isinstance(checks, (list, tuple)):
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            if bool(check.get("blocking", True)) and not bool(check.get("ok")):
                name = str(check.get("name", "")).strip()
                if name:
                    return name
    unresolved_questions = inspection_report.get("unresolved_questions", ())
    if isinstance(unresolved_questions, (list, tuple)) and unresolved_questions:
        return "guided-route"
    return None


def build_train_rollout_metrics(
    *,
    event: str,
    inspection_report: Mapping[str, object],
    recommendations: tuple[Mapping[str, object], ...],
    resume_strategy: str,
    guided_flow_completed: bool,
    artifact_path: str | None = None,
    resolving_question_answered: bool = False,
) -> dict[str, object]:
    accepted_recommendations = [
        str(entry.get("name", "")).strip()
        for entry in recommendations
        if isinstance(entry, Mapping) and bool(entry.get("applied"))
    ]
    hosted_runtime = (
        str(inspection_report.get("hosted_runtime", "local")).strip() or "local"
    )
    resolving_question = inspection_report.get("resolving_question")
    artifact_reused = str(resume_strategy).strip().lower() in {
        "safe-continue",
        "re-fit",
    }
    return {
        "event": str(event),
        "guided_flow_completed": bool(guided_flow_completed),
        "validation_failure_category": _validation_failure_category(
            inspection_report
        ),
        "accepted_recommendations": accepted_recommendations,
        "recommendation_acceptance_count": len(accepted_recommendations),
        "resolving_question_asked": resolving_question is not None,
        "resolving_question_answered": bool(resolving_question_answered),
        "resume_strategy": str(resume_strategy),
        "resume_success": bool(
            guided_flow_completed and artifact_reused and artifact_path
        ),
        "artifact_reused": artifact_reused,
        "artifact_path": str(artifact_path or "") or None,
        "hosted_runtime": hosted_runtime,
        "source_mode": str(inspection_report.get("source_mode", "")).strip(),
        "row_count": inspection_report.get("row_count"),
    }


def record_train_rollout_event(
    *,
    event: str,
    inspection_report: Mapping[str, object],
    recommendations: tuple[Mapping[str, object], ...],
    resume_strategy: str,
    guided_flow_completed: bool,
    artifact_path: str | None = None,
    resolving_question_answered: bool = False,
) -> dict[str, object]:
    job_dir = create_job_dir("train/rollout", stem=f"guided_{event}")
    rollout_metrics = build_train_rollout_metrics(
        event=event,
        inspection_report=inspection_report,
        recommendations=recommendations,
        resume_strategy=resume_strategy,
        guided_flow_completed=guided_flow_completed,
        artifact_path=artifact_path,
        resolving_question_answered=resolving_question_answered,
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_dir": job_dir,
        "rollout_metrics": rollout_metrics,
    }
    write_manifest(job_dir, payload, name=_TRAIN_ROLLOUT_METRIC_NAME)
    return payload


def scan_train_rollout_metrics(
    root_path: str | None = None,
) -> list[dict[str, object]]:
    resolved_root = (
        Path(str(root_path)).expanduser()
        if root_path is not None and str(root_path).strip()
        else Path(managed_output_dir("train/rollout"))
    )
    if not resolved_root.exists():
        return []
    metrics = []
    for manifest_file in sorted(
        resolved_root.rglob(_TRAIN_ROLLOUT_METRIC_NAME)
    ):
        try:
            payload = read_manifest(
                str(manifest_file.parent), name=manifest_file.name
            )
        except Exception:
            continue
        rollout_metrics = payload.get("rollout_metrics")
        if isinstance(rollout_metrics, dict):
            metrics.append(dict(rollout_metrics))
    return metrics


def write_train_session_manifest(
    *,
    normalized_request: Mapping[str, object],
    inspection_report: Mapping[str, object],
    recommendations: tuple[Mapping[str, object], ...],
    plan_markdown: str,
    artifact_path: str,
    status_markdown: str,
    resume_strategy: str,
    rollout_metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_artifact_path = str(Path(str(artifact_path)).expanduser())
    artifact_parent = str(Path(resolved_artifact_path).parent)
    job_dir = create_job_dir("train", stem=Path(resolved_artifact_path).stem)
    session_manifest_path = manifest_path(
        job_dir,
        name=_TRAIN_SESSION_MANIFEST_NAME,
    )
    artifact_manifest_path = manifest_path(
        artifact_parent,
        name=train_artifact_manifest_name(resolved_artifact_path),
    )
    next_actions = build_train_next_actions(
        artifact_path=resolved_artifact_path,
        session_manifest_path=session_manifest_path,
        resume_strategy=resume_strategy,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_dir": job_dir,
        "session_manifest_path": session_manifest_path,
        "artifact_manifest_path": artifact_manifest_path,
        "artifact_path": resolved_artifact_path,
        "normalized_request": dict(normalized_request),
        "inspection_report": dict(inspection_report),
        "recommendations": list(recommendations),
        "plan_markdown": str(plan_markdown),
        "status_markdown": str(status_markdown),
        "resume_strategy": str(resume_strategy),
        "rollout_metrics": dict(rollout_metrics or {}),
        "next_actions": list(next_actions),
    }
    write_manifest(job_dir, manifest, name=_TRAIN_SESSION_MANIFEST_NAME)
    write_manifest(
        artifact_parent,
        manifest,
        name=train_artifact_manifest_name(resolved_artifact_path),
    )
    return manifest


def render_train_result_markdown(manifest: Mapping[str, object]) -> str:
    lines = [
        "## Use Result",
        f"- Artifact: {manifest.get('artifact_path') or 'none'}",
        f"- Session Manifest: {manifest.get('session_manifest_path') or 'none'}",
        f"- Artifact Sidecar: {manifest.get('artifact_manifest_path') or 'none'}",
        f"- Resume Strategy: {manifest.get('resume_strategy') or 'none'}",
    ]
    next_actions = manifest.get("next_actions", ())
    for action in (
        next_actions if isinstance(next_actions, (list, tuple)) else ()
    ):
        if not isinstance(action, Mapping):
            continue
        title = str(action.get("title", "Next Action")).strip() or "Next Action"
        detail = str(action.get("detail", "")).strip()
        if detail:
            lines.append(f"- {title}: {detail}")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)


__all__ = (
    "build_train_next_actions",
    "build_train_rollout_metrics",
    "load_train_artifact_manifest",
    "record_train_rollout_event",
    "render_train_result_markdown",
    "scan_train_rollout_metrics",
    "train_artifact_manifest_name",
    "write_train_session_manifest",
)
