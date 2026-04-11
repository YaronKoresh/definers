from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

TRAIN_COACH_ENTRY_INTENTS = (
    {
        "id": "files",
        "title": "I have files",
        "description": "Upload local files and let Definers inspect the safest training route.",
    },
    {
        "id": "dataset",
        "title": "I have a dataset",
        "description": "Point Definers at a remote dataset and let it inspect columns and splits.",
    },
    {
        "id": "resume",
        "title": "Continue yesterday's model",
        "description": "Bring back a saved model and combine it with fresh data when guided intake can verify the route.",
    },
)

TRAIN_COACH_STEP_NAMES = (
    "Upload Or Connect",
    "Inspect And Confirm",
    "Review Plan",
    "Train",
    "Use Result",
)

_TABULAR_EXTENSIONS = frozenset({"csv", "json", "xlsx"})
_TEXT_EXTENSIONS = frozenset({"txt"})
_AUDIO_EXTENSIONS = frozenset({"wav", "mp3", "flac", "ogg", "m4a"})
_IMAGE_EXTENSIONS = frozenset(
    {"png", "jpg", "jpeg", "bmp", "gif", "webp", "tif", "tiff"}
)
_VIDEO_EXTENSIONS = frozenset({"mp4", "mkv", "mov", "avi", "webm"})
_MODEL_EXTENSIONS = frozenset(
    {"joblib", "pkl", "onnx", "pt", "pth", "safetensors"}
)
_LABEL_NAME_PATTERN = re.compile(
    r"(^|[^a-z0-9])(label|labels|target|targets|class|classes|annotation|annotations|y)([^a-z0-9]|$)"
)
_DROP_NAME_PATTERN = re.compile(
    r"(^|[^a-z0-9])(id|ids|uuid|guid|index|row_id|filename|file_name|filepath|path|url|uri)([^a-z0-9]|$)"
)
_SOURCE_TYPES = frozenset(
    {"parquet", "json", "csv", "arrow", "webdataset", "txt"}
)


@dataclass(frozen=True, slots=True)
class TrainCoachCheck:
    name: str
    ok: bool
    detail: str
    blocking: bool = True


@dataclass(frozen=True, slots=True)
class TrainCoachRecommendation:
    name: str
    value: object
    reason: str
    confidence: str
    applied: bool = False


@dataclass(frozen=True, slots=True)
class TrainCoachResumeGuidance:
    strategy: str
    confidence: str
    detail: str
    use_resume_artifact: bool
    manifest_path: str | None
    revalidation: tuple[str, ...]
    recovered_label_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainCoachResolvingQuestion:
    question_id: str
    prompt: str
    option_values: tuple[str, ...]
    option_labels: tuple[str, ...]
    default_value: str | None = None


@dataclass(frozen=True, slots=True)
class TrainCoachState:
    requested_intent: str
    effective_intent: str
    confidence: str
    hosted_runtime: str
    source_mode: str
    source_type: str
    remote_src: str | None
    revision: str | None
    features: tuple[str, ...]
    labels: tuple[str, ...]
    resume_from: str | None
    save_as: str | None
    column_names: tuple[str, ...]
    detected_file_families: tuple[str, ...]
    label_candidates: tuple[str, ...]
    selected_label_columns: tuple[str, ...]
    suggested_batch_size: int
    suggested_validation_split: float
    suggested_test_split: float
    order_by: str | None
    stratify: str | None
    selected_rows: str | None
    row_count: int | None
    unresolved_questions: tuple[str, ...]
    warnings: tuple[str, ...]
    notes: tuple[str, ...]
    recommendations: tuple[TrainCoachRecommendation, ...]
    resume_guidance: TrainCoachResumeGuidance
    resolving_question: TrainCoachResolvingQuestion | None
    checks: tuple[TrainCoachCheck, ...]
    ready: bool


def train_coach_entry_intents():
    return TRAIN_COACH_ENTRY_INTENTS


def train_coach_step_names():
    return TRAIN_COACH_STEP_NAMES


def build_train_coach_contract_markdown() -> str:
    lines = [
        "## Beginner Train Coach",
        "Start with one clear intent and let Definers inspect the safest guided route before you see the full plan.",
    ]
    for entry in TRAIN_COACH_ENTRY_INTENTS:
        lines.append(f"- **{entry['title']}:** {entry['description']}")
    lines.extend(
        [
            "- Guided mode blocks unsafe or ambiguous routes before training starts.",
            "- If guided intake cannot resolve your files safely, switch to `definers start train` for the advanced workbench.",
        ]
    )
    return "\n".join(lines)


def train_coach_state_json(state: TrainCoachState) -> str:
    return json.dumps(asdict(state), ensure_ascii=False)


def parse_train_coach_state(payload) -> TrainCoachState | None:
    normalized_payload = str(payload or "").strip()
    if not normalized_payload:
        return None
    data = json.loads(normalized_payload)
    recommendations = tuple(
        TrainCoachRecommendation(
            name=str(entry.get("name", "")).strip(),
            value=entry.get("value"),
            reason=str(entry.get("reason", "")).strip(),
            confidence=str(entry.get("confidence", "low")).strip() or "low",
            applied=bool(entry.get("applied")),
        )
        for entry in data.get("recommendations", ())
    )
    resume_guidance_payload = data.get("resume_guidance", {})
    if not isinstance(resume_guidance_payload, dict):
        resume_guidance_payload = {}
    checks = tuple(
        TrainCoachCheck(
            name=str(entry.get("name", "")).strip(),
            ok=bool(entry.get("ok")),
            detail=str(entry.get("detail", "")).strip(),
            blocking=bool(entry.get("blocking", True)),
        )
        for entry in data.get("checks", ())
    )
    return TrainCoachState(
        requested_intent=str(data.get("requested_intent", "files")),
        effective_intent=str(data.get("effective_intent", "files")),
        confidence=str(data.get("confidence", "low")),
        hosted_runtime=str(data.get("hosted_runtime", "local")),
        source_mode=str(data.get("source_mode", "empty")),
        source_type=str(data.get("source_type", "parquet")),
        remote_src=_normalize_optional_text(data.get("remote_src")),
        revision=_normalize_optional_text(data.get("revision")),
        features=tuple(str(value) for value in data.get("features", ())),
        labels=tuple(str(value) for value in data.get("labels", ())),
        resume_from=_normalize_optional_text(data.get("resume_from")),
        save_as=_normalize_optional_text(data.get("save_as")),
        column_names=tuple(
            str(value) for value in data.get("column_names", ())
        ),
        detected_file_families=tuple(
            str(value) for value in data.get("detected_file_families", ())
        ),
        label_candidates=tuple(
            str(value) for value in data.get("label_candidates", ())
        ),
        selected_label_columns=tuple(
            str(value) for value in data.get("selected_label_columns", ())
        ),
        suggested_batch_size=int(data.get("suggested_batch_size", 32)),
        suggested_validation_split=float(
            data.get("suggested_validation_split", 0.0)
        ),
        suggested_test_split=float(data.get("suggested_test_split", 0.0)),
        order_by=_normalize_optional_text(data.get("order_by")),
        stratify=_normalize_optional_text(data.get("stratify")),
        selected_rows=_normalize_optional_text(data.get("selected_rows")),
        row_count=_coerce_optional_int(data.get("row_count")),
        unresolved_questions=tuple(
            str(value) for value in data.get("unresolved_questions", ())
        ),
        warnings=tuple(str(value) for value in data.get("warnings", ())),
        notes=tuple(str(value) for value in data.get("notes", ())),
        recommendations=recommendations,
        resume_guidance=TrainCoachResumeGuidance(
            strategy=str(
                resume_guidance_payload.get("strategy", "none")
            ).strip()
            or "none",
            confidence=str(
                resume_guidance_payload.get("confidence", "high")
            ).strip()
            or "high",
            detail=str(resume_guidance_payload.get("detail", "")).strip(),
            use_resume_artifact=bool(
                resume_guidance_payload.get("use_resume_artifact")
            ),
            manifest_path=_normalize_optional_text(
                resume_guidance_payload.get("manifest_path")
            ),
            revalidation=tuple(
                str(value)
                for value in resume_guidance_payload.get("revalidation", ())
            ),
            recovered_label_columns=tuple(
                str(value)
                for value in resume_guidance_payload.get(
                    "recovered_label_columns", ()
                )
            ),
        ),
        resolving_question=_parse_resolving_question(
            data.get("resolving_question")
        ),
        checks=checks,
        ready=bool(data.get("ready")),
    )


def build_train_coach_state(
    *,
    requested_intent,
    uploaded_files=None,
    local_collection_path=None,
    remote_src=None,
    resume_artifact=None,
    revision=None,
    resolving_choice=None,
    save_as=None,
    health_snapshot=None,
    activity_reporter=None,
) -> TrainCoachState:
    normalized_intent = _normalize_requested_intent(requested_intent)
    normalized_remote = _normalize_optional_text(remote_src)
    normalized_revision = _normalize_optional_text(revision)
    normalized_save_as = _normalize_optional_text(save_as)
    normalized_choice = _normalize_optional_text(resolving_choice)
    hosted_runtime = _detect_hosted_runtime()
    normalized_files = _normalize_uploaded_files(uploaded_files)
    normalized_collection_path = _safe_local_path(local_collection_path)
    if normalized_collection_path is not None:
        normalized_files = tuple(
            dict.fromkeys(
                (
                    *normalized_files,
                    *_expand_local_input_path(normalized_collection_path),
                )
            )
        )
    source_files, inferred_resume, file_warnings = _extract_resume_candidates(
        normalized_files
    )
    resolved_resume = _safe_local_path(resume_artifact) or inferred_resume
    source_families = _detected_file_families(source_files)
    source_type = _infer_source_type(normalized_remote)
    if source_files and normalized_remote is not None:
        if normalized_choice == "local-files":
            normalized_remote = None
        elif normalized_choice == "remote-dataset":
            source_files = ()
            source_families = ()
    requested_source_count = int(bool(source_files)) + int(
        bool(normalized_remote)
    )
    effective_intent = _resolve_effective_intent(
        normalized_intent,
        source_files,
        normalized_remote,
        resolved_resume,
    )
    _report_guided_activity(
        activity_reporter,
        1,
        "Check files",
        "Validating files, datasets, and resume artifacts.",
    )
    confidence = _resolve_confidence(
        normalized_intent,
        effective_intent,
        requested_source_count,
        source_families,
    )
    row_count = None
    column_names: tuple[str, ...] = ()
    label_candidates: tuple[str, ...] = ()
    selected_label_columns: tuple[str, ...] = ()
    drop_candidates: tuple[str, ...] = ()
    feature_files: tuple[str, ...] = ()
    label_files: tuple[str, ...] = ()
    selected_rows = None
    resolving_question = None
    unresolved_questions: list[str] = []
    warnings = list(file_warnings)
    notes: list[str] = []
    source_mode = "empty"

    if normalized_collection_path is not None:
        notes.append(
            f"Guided mode expanded the local collection path {normalized_collection_path}."
        )
    if hosted_runtime != "local":
        notes.append(
            f"Guided mode is running inside {_hosted_runtime_label(hosted_runtime)} and will keep preview and retention inside hosted-safe budgets."
        )

    if requested_source_count > 1:
        resolving_question = _build_source_resolving_question()
        unresolved_questions.append(resolving_question.prompt)
        source_mode = "mixed-source"
    elif normalized_remote is not None:
        _report_guided_activity(
            activity_reporter,
            2,
            "Understand data",
            "Inspecting the remote dataset for columns, labels, and dataset size.",
        )
        source_mode = "remote-dataset"
        remote_inspection = _inspect_remote_dataset(
            normalized_remote,
            source_type,
            normalized_revision,
            hosted_runtime=hosted_runtime,
            resolving_choice=normalized_choice,
        )
        column_names = remote_inspection["column_names"]
        label_candidates = remote_inspection["label_candidates"]
        selected_label_columns = remote_inspection["selected_label_columns"]
        drop_candidates = remote_inspection["drop_candidates"]
        row_count = remote_inspection["row_count"]
        selected_rows = remote_inspection["selected_rows"]
        resolving_question = remote_inspection["resolving_question"]
        warnings.extend(remote_inspection["warnings"])
        unresolved_questions.extend(remote_inspection["unresolved_questions"])
        notes.extend(remote_inspection["notes"])
    elif source_files:
        _report_guided_activity(
            activity_reporter,
            2,
            "Understand data",
            "Inspecting local files for schema, labels, and safe training routes.",
        )
        source_mode = _resolve_local_source_mode(source_families)
        local_inspection = _inspect_local_files(
            source_files,
            source_families,
            resolving_choice=normalized_choice,
            hosted_runtime=hosted_runtime,
        )
        column_names = local_inspection["column_names"]
        label_candidates = local_inspection["label_candidates"]
        selected_label_columns = local_inspection["selected_label_columns"]
        drop_candidates = local_inspection["drop_candidates"]
        feature_files = local_inspection["feature_files"]
        label_files = local_inspection["label_files"]
        row_count = local_inspection["row_count"]
        selected_rows = local_inspection["selected_rows"]
        resolving_question = local_inspection["resolving_question"]
        source_mode = str(local_inspection.get("source_mode") or source_mode)
        source_families = _detected_file_families(feature_files or source_files)
        warnings.extend(local_inspection["warnings"])
        unresolved_questions.extend(local_inspection["unresolved_questions"])
        notes.extend(local_inspection["notes"])
    elif resolved_resume is not None:
        source_mode = "resume-only"
        unresolved_questions.append(
            "Guided resume needs fresh data as well as the previous model artifact."
        )
    else:
        unresolved_questions.append(
            "Add files, connect a dataset, or choose a previous model artifact to begin."
        )

    if not selected_label_columns and label_candidates:
        selected_label_columns = (label_candidates[0],)
    resume_guidance = _build_resume_guidance(
        resume_from=resolved_resume,
        source_mode=source_mode,
        source_type=source_type,
        column_names=column_names,
        selected_label_columns=selected_label_columns,
    )
    if not selected_label_columns and resume_guidance.recovered_label_columns:
        selected_label_columns = resume_guidance.recovered_label_columns
        notes.append(
            "Guided mode restored label columns from the previous train session manifest."
        )
    budget_adjustment = _apply_hosted_runtime_budget(
        hosted_runtime=hosted_runtime,
        source_mode=source_mode,
        feature_files=feature_files or source_files,
        label_files=label_files,
        row_count=row_count,
        selected_rows=selected_rows,
        resolving_choice=normalized_choice,
        resolving_question=resolving_question,
    )
    feature_files = budget_adjustment["feature_files"]
    label_files = budget_adjustment["label_files"]
    selected_rows = budget_adjustment["selected_rows"]
    if resolving_question is None:
        resolving_question = budget_adjustment["resolving_question"]
    warnings.extend(budget_adjustment["warnings"])
    unresolved_questions.extend(budget_adjustment["unresolved_questions"])
    notes.extend(budget_adjustment["notes"])
    suggested_validation_split, suggested_test_split = _suggest_splits(
        row_count
    )
    suggested_batch_size = _suggest_batch_size(row_count, source_families)
    stratify = (
        selected_label_columns[0]
        if row_count is not None and row_count >= 20 and selected_label_columns
        else None
    )
    recommendations = _build_recommendations(
        selected_label_columns=selected_label_columns,
        label_candidates=label_candidates,
        drop_candidates=drop_candidates,
        source_families=source_families,
        source_mode=source_mode,
        row_count=row_count,
        selected_rows=selected_rows,
        suggested_batch_size=suggested_batch_size,
        suggested_validation_split=suggested_validation_split,
        suggested_test_split=suggested_test_split,
        stratify=stratify,
        hosted_runtime=hosted_runtime,
        resume_guidance=resume_guidance,
    )
    _report_guided_activity(
        activity_reporter,
        3,
        "Write recommendations",
        "Explaining the recommended labels, splits, resume route, and next safe defaults.",
    )
    checks = _build_checks(
        health_snapshot=health_snapshot,
        source_files=feature_files or source_files,
        remote_src=normalized_remote,
        resume_from=resolved_resume,
        resume_guidance=resume_guidance,
        resolving_question=resolving_question,
        hosted_runtime=hosted_runtime,
        unresolved_questions=tuple(unresolved_questions),
        source_mode=source_mode,
    )
    ready = all(check.ok for check in checks if check.blocking)
    if not ready:
        if resolving_question is not None:
            notes.append(
                "Guided mode is waiting for one quick decision before it can continue."
            )
        else:
            notes.append(
                "Use `definers start train` if you need manual control over columns, file routing, or mixed inputs."
            )
    elif not selected_label_columns and source_mode in {
        "remote-dataset",
        "local-tabular",
    }:
        notes.append(
            "No clear label column was inferred. Guided mode will keep the plan unlabeled unless you switch to the advanced workbench."
        )
    if (
        resolved_resume is not None
        and feature_files
        and resume_guidance.use_resume_artifact
    ):
        notes.append(
            "Guided mode will reuse the previous model artifact and add the new data to the training route."
        )
    elif (
        resolved_resume is not None and not resume_guidance.use_resume_artifact
    ):
        notes.append(
            "Guided mode will keep the previous model artifact for comparison only and start a fresh model route."
        )
    return TrainCoachState(
        requested_intent=normalized_intent,
        effective_intent=effective_intent,
        confidence=confidence,
        hosted_runtime=hosted_runtime,
        source_mode=source_mode,
        source_type=source_type,
        remote_src=normalized_remote,
        revision=normalized_revision,
        features=feature_files or source_files,
        labels=label_files,
        resume_from=resolved_resume,
        save_as=normalized_save_as,
        column_names=column_names,
        detected_file_families=source_families,
        label_candidates=label_candidates,
        selected_label_columns=selected_label_columns,
        suggested_batch_size=suggested_batch_size,
        suggested_validation_split=suggested_validation_split,
        suggested_test_split=suggested_test_split,
        order_by="shuffle" if row_count is not None and row_count > 1 else None,
        stratify=stratify,
        selected_rows=selected_rows,
        row_count=row_count,
        unresolved_questions=tuple(dict.fromkeys(unresolved_questions)),
        warnings=tuple(dict.fromkeys(warnings)),
        notes=tuple(dict.fromkeys(notes)),
        recommendations=recommendations,
        resume_guidance=resume_guidance,
        resolving_question=resolving_question,
        checks=checks,
        ready=ready,
    )


def render_train_coach_summary_markdown(state: TrainCoachState) -> str:
    detail = (
        "Guided route is ready for plan preview and training."
        if state.ready
        else "Guided route needs manual review before training can continue."
    )
    lines = [
        "## Guided Intake",
        detail,
        f"- Requested Intent: {state.requested_intent}",
        f"- Guided Route: {state.effective_intent}",
        f"- Confidence: {state.confidence}",
        f"- Runtime: {_hosted_runtime_label(state.hosted_runtime)}",
        f"- Source Mode: {state.source_mode}",
        f"- Source Type: {state.source_type}",
        f"- Rows Detected: {state.row_count if state.row_count is not None else 'unknown'}",
    ]
    if state.selected_label_columns:
        lines.append(
            "- Selected Label Columns: "
            + ", ".join(state.selected_label_columns)
        )
    else:
        lines.append("- Selected Label Columns: none")
    if state.resume_from is not None:
        lines.append(f"- Resume Artifact: {state.resume_from}")
        lines.append(
            f"- Resume Strategy: {_humanize_resume_strategy(state.resume_guidance.strategy)}"
        )
    if state.resolving_question is not None:
        lines.append(f"- Quick Decision: {state.resolving_question.prompt}")
    if not state.ready and state.resolving_question is None:
        lines.append("- Advanced Fallback: definers start train")
    return "\n".join(lines)


def render_train_coach_inspection_markdown(state: TrainCoachState) -> str:
    lines = [
        "## Data Inspection",
        f"- Runtime: {_hosted_runtime_label(state.hosted_runtime)}",
        f"- Detected File Families: {', '.join(state.detected_file_families) or 'none'}",
        f"- Feature Files: {len(state.features)}",
        f"- Label Files: {len(state.labels)}",
        f"- Row Count: {state.row_count if state.row_count is not None else 'unknown'}",
        "- Column Names: " + (", ".join(state.column_names) or "none"),
        "- Label Candidates: " + (", ".join(state.label_candidates) or "none"),
        f"- Suggested Batch Size: {state.suggested_batch_size}",
        f"- Suggested Validation Split: {state.suggested_validation_split}",
        f"- Suggested Test Split: {state.suggested_test_split}",
    ]
    for recommendation in state.recommendations:
        lines.append(
            "- Recommendation: "
            + f"{recommendation.name} -> {_recommendation_value_text(recommendation.value)} "
            + f"({recommendation.confidence}, {'applied' if recommendation.applied else 'optional'})"
        )
        lines.append(f"- Reason: {recommendation.reason}")
    for warning in state.warnings:
        lines.append(f"- Warning: {warning}")
    for note in state.notes:
        lines.append(f"- Note: {note}")
    return "\n".join(lines)


def render_train_coach_validation_markdown(state: TrainCoachState) -> str:
    lines = [
        "## Guided Validation",
        "- Status: ready" if state.ready else "- Status: blocked",
    ]
    for check in state.checks:
        prefix = (
            "Ready" if check.ok else ("Blocked" if check.blocking else "Warn")
        )
        lines.append(f"- {prefix}: {check.name} - {check.detail}")
    lines.append(f"- Runtime: {_hosted_runtime_label(state.hosted_runtime)}")
    if state.resume_from is not None:
        lines.append(
            f"- Resume Strategy: {_humanize_resume_strategy(state.resume_guidance.strategy)}"
        )
        lines.append(f"- Resume Confidence: {state.resume_guidance.confidence}")
        if state.resume_guidance.manifest_path is not None:
            lines.append(
                f"- Previous Session Manifest: {state.resume_guidance.manifest_path}"
            )
        for item in state.resume_guidance.revalidation:
            lines.append(f"- Revalidate: {item}")
    if state.resolving_question is not None:
        lines.append(f"- Quick Decision: {state.resolving_question.prompt}")
    for question in state.unresolved_questions:
        lines.append(f"- Needs Review: {question}")
    if not state.ready:
        if state.resolving_question is not None:
            lines.append(
                "- Next Step: answer the quick decision and inspect again."
            )
        else:
            lines.append(
                "- Next Step: open the advanced workbench with `definers start train`."
            )
    return "\n".join(lines)


def _normalize_requested_intent(value) -> str:
    normalized_value = str(value or "files").strip().lower()
    if normalized_value not in {
        entry["id"] for entry in TRAIN_COACH_ENTRY_INTENTS
    }:
        return "files"
    return normalized_value


def _normalize_optional_text(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_resolving_question(payload) -> TrainCoachResolvingQuestion | None:
    if not isinstance(payload, dict):
        return None
    option_values = tuple(
        str(value)
        for value in payload.get("option_values", ())
        if str(value).strip()
    )
    option_labels = tuple(
        str(value)
        for value in payload.get("option_labels", ())
        if str(value).strip()
    )
    if not option_values or len(option_values) != len(option_labels):
        return None
    return TrainCoachResolvingQuestion(
        question_id=str(payload.get("question_id", "")).strip()
        or "guided-decision",
        prompt=str(payload.get("prompt", "")).strip()
        or "Resolve the guided route.",
        option_values=option_values,
        option_labels=option_labels,
        default_value=_normalize_optional_text(payload.get("default_value")),
    )


def _detect_hosted_runtime() -> str:
    try:
        from definers.system.runtime_budget import detect_hosted_runtime

        return detect_hosted_runtime()
    except Exception:
        return "local"


def _hosted_runtime_label(runtime: str) -> str:
    try:
        from definers.system.runtime_budget import hosted_runtime_label

        return hosted_runtime_label(runtime)
    except Exception:
        return str(runtime or "local").strip() or "local"


def _coerce_optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _safe_local_path(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "name") and not isinstance(value, (str, bytes)):
        value = getattr(value, "name")
    try:
        from definers.data.loader_runtime import LoaderRuntimeSupport

        return LoaderRuntimeSupport._safe_path(str(value).strip())
    except Exception:
        return None


def _expand_local_input_path(path: str | None) -> tuple[str, ...]:
    resolved_path = _safe_local_path(path)
    if resolved_path is None:
        return ()
    path_object = Path(resolved_path)
    if not path_object.exists():
        return ()
    if path_object.is_file():
        return (resolved_path,)
    if not path_object.is_dir():
        return ()
    expanded_paths = []
    for child_path in sorted(path_object.rglob("*")):
        if not child_path.is_file():
            continue
        safe_child_path = _safe_local_path(str(child_path))
        if safe_child_path is None:
            continue
        expanded_paths.append(safe_child_path)
    return tuple(expanded_paths)


def _normalize_uploaded_files(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        value = [value]
    resolved_paths = []
    seen_paths = set()
    for item in value:
        for safe_path in _expand_local_input_path(item):
            if safe_path in seen_paths:
                continue
            seen_paths.add(safe_path)
            resolved_paths.append(safe_path)
    return tuple(resolved_paths)


def _path_extension(path: str) -> str | None:
    suffix = Path(path).suffix.strip().lower().lstrip(".")
    return suffix or None


def _file_family(path: str) -> str:
    extension = _path_extension(path)
    if extension in _TABULAR_EXTENSIONS:
        return "tabular"
    if extension in _TEXT_EXTENSIONS:
        return "text"
    if extension in _AUDIO_EXTENSIONS:
        return "audio"
    if extension in _IMAGE_EXTENSIONS:
        return "image"
    if extension in _VIDEO_EXTENSIONS:
        return "video"
    if extension in _MODEL_EXTENSIONS:
        return "model"
    return "other"


def _detected_file_families(paths: tuple[str, ...]) -> tuple[str, ...]:
    families = []
    for path in paths:
        family = _file_family(path)
        if family not in families:
            families.append(family)
    return tuple(families)


def _extract_resume_candidates(paths: tuple[str, ...]):
    source_files = []
    model_artifacts = []
    warnings = []
    for path in paths:
        if _file_family(path) == "model":
            model_artifacts.append(path)
            continue
        source_files.append(path)
    if len(model_artifacts) > 1:
        warnings.append(
            "Multiple model artifacts were uploaded. Guided mode will ignore them unless you pick one in the resume field."
        )
        return tuple(source_files), None, tuple(warnings)
    return (
        tuple(source_files),
        (model_artifacts[0] if model_artifacts else None),
        tuple(warnings),
    )


def _resolve_effective_intent(
    requested_intent: str,
    source_files: tuple[str, ...],
    remote_src: str | None,
    resume_from: str | None,
) -> str:
    if resume_from is not None and (source_files or remote_src):
        return "resume"
    if remote_src is not None:
        return "dataset"
    if source_files:
        return "files"
    if resume_from is not None:
        return "resume"
    return requested_intent


def _resolve_confidence(
    requested_intent: str,
    effective_intent: str,
    requested_source_count: int,
    source_families: tuple[str, ...],
) -> str:
    if requested_source_count > 1 or "other" in source_families:
        return "low"
    if requested_intent == effective_intent:
        return "high"
    if requested_source_count == 1:
        return "medium"
    return "low"


def _infer_source_type(remote_src: str | None) -> str:
    if remote_src is None:
        return "parquet"
    parsed_url = urlparse(remote_src)
    extension = (
        Path(parsed_url.path or remote_src).suffix.strip().lower().lstrip(".")
    )
    if extension in _SOURCE_TYPES:
        return extension
    return "parquet"


def _resolve_local_source_mode(source_families: tuple[str, ...]) -> str:
    if not source_families:
        return "empty"
    if len(source_families) > 1:
        return "mixed-local"
    family = source_families[0]
    return f"local-{family}"


def _build_resolving_question(
    question_id: str,
    prompt: str,
    options: tuple[tuple[str, str], ...],
    *,
    default_value: str | None = None,
) -> TrainCoachResolvingQuestion:
    option_values = tuple(str(value) for value, _ in options)
    option_labels = tuple(str(label) for _, label in options)
    return TrainCoachResolvingQuestion(
        question_id=question_id,
        prompt=prompt,
        option_values=option_values,
        option_labels=option_labels,
        default_value=default_value
        or (option_values[0] if option_values else None),
    )


def _build_source_resolving_question() -> TrainCoachResolvingQuestion:
    return _build_resolving_question(
        "source-choice",
        "Which source should guided mode inspect first?",
        (
            ("local-files", "Inspect the local files"),
            ("remote-dataset", "Inspect the remote dataset"),
        ),
    )


def _build_tabular_label_resolving_question(
    paths: tuple[str, ...],
) -> TrainCoachResolvingQuestion:
    first_name = Path(paths[0]).name if paths else "the first file"
    second_name = Path(paths[1]).name if len(paths) > 1 else "the second file"
    return _build_resolving_question(
        "tabular-label-source",
        "Which file contains the labels?",
        (
            ("first-file-labels", f"Use {first_name} as the label file"),
            ("second-file-labels", f"Use {second_name} as the label file"),
            ("review-manually", "Review this in advanced mode"),
        ),
    )


def _build_media_label_resolving_question(
    family: str,
) -> TrainCoachResolvingQuestion:
    return _build_resolving_question(
        f"{family}-label-route",
        f"No confident labels were found for these {family} files. What should guided mode do?",
        (
            (
                "continue-unlabeled",
                "Continue with the files only for a first pass",
            ),
            ("review-manually", "Review this outside guided mode"),
        ),
    )


def _build_dominant_family_resolving_question(
    dominant_family: str,
) -> TrainCoachResolvingQuestion:
    return _build_resolving_question(
        "dominant-family-route",
        f"The upload mixes file families. Keep only the detected {dominant_family} files for a first guided pass?",
        (
            ("keep-dominant-family", f"Keep only the {dominant_family} files"),
            ("review-manually", "Review this in advanced mode"),
        ),
    )


def _build_hosted_sampling_resolving_question(
    hosted_runtime: str,
    limit: int,
    unit_label: str,
) -> TrainCoachResolvingQuestion:
    return _build_resolving_question(
        "hosted-sampling",
        f"{_hosted_runtime_label(hosted_runtime)} is better with a smaller first pass. Use the first {limit} {unit_label}?",
        (
            ("sample-first-pass", f"Use the first {limit} {unit_label}"),
            ("review-outside-host", "Review this outside the hosted runtime"),
        ),
    )


def _group_paths_by_family(
    paths: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    grouped_paths: dict[str, list[str]] = {}
    for path in paths:
        family = _file_family(path)
        grouped_paths.setdefault(family, []).append(path)
    return {family: tuple(values) for family, values in grouped_paths.items()}


def _dominant_local_family(paths: tuple[str, ...]) -> str | None:
    grouped_paths = _group_paths_by_family(paths)
    dominant_family = None
    dominant_count = 0
    for family, values in grouped_paths.items():
        if len(values) <= dominant_count:
            continue
        dominant_family = family
        dominant_count = len(values)
    return dominant_family


def _parent_folder_labels(paths: tuple[str, ...]) -> tuple[str, ...] | None:
    if not paths:
        return None
    parent_names = tuple(Path(path).parent.name.strip() for path in paths)
    unique_parent_names = {
        name for name in parent_names if name and name not in {".", ".."}
    }
    if len(unique_parent_names) < 2:
        return None
    return parent_names


def _normalize_label_text(value) -> str:
    text = str(value if value is not None else "unknown").strip()
    return text.replace("\r", " ").replace("\n", " ") or "unknown"


def _materialize_label_sidecars(
    labels: tuple[object, ...],
    *,
    stem: str,
) -> tuple[tuple[str, ...], str]:
    from definers.system.output_paths import managed_output_session_dir

    session_dir = Path(
        managed_output_session_dir("train/label_sidecars", stem=stem)
    )
    label_paths = []
    for index, label_value in enumerate(labels, start=1):
        label_path = session_dir / f"label_{index:05d}.csv"
        with label_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(["label"])
            writer.writerow([_normalize_label_text(label_value)])
        label_paths.append(str(label_path))
    return tuple(label_paths), str(session_dir)


def _read_text_lines(path: str) -> tuple[str, ...]:
    with open(path, encoding="utf-8", errors="ignore") as file_obj:
        return tuple(
            line.strip()
            for line in file_obj.read().splitlines()
            if line.strip()
        )


def _read_tabular_dataframe(path: str):
    import pandas

    extension = _path_extension(path)
    if extension == "csv":
        dataframe = pandas.read_csv(path)
        if dataframe.empty:
            dataframe = pandas.read_csv(path, header=None)
        return dataframe
    if extension == "xlsx":
        from definers import optional_dependencies

        optional_dependencies.ensure_module_runtime("openpyxl")
        return pandas.read_excel(path)
    return pandas.read_json(path)


def _resolve_media_labels_from_text_sidecars(
    media_files: tuple[str, ...],
    sidecar_files: tuple[str, ...],
):
    warnings = []
    notes = []
    if len(sidecar_files) == 1:
        try:
            lines = _read_text_lines(sidecar_files[0])
        except Exception as error:
            return (), tuple([f"Could not read the text sidecar: {error}"]), ()
        if len(lines) == len(media_files):
            notes.append(
                "Guided mode aligned one text sidecar line per media file."
            )
            return lines, tuple(warnings), tuple(notes)
        warnings.append(
            "The text sidecar does not contain one line per media file."
        )
        return (), tuple(warnings), tuple(notes)
    if len(sidecar_files) == len(media_files):
        labels = []
        for sidecar_path in sorted(sidecar_files):
            try:
                labels.append(" ".join(_read_text_lines(sidecar_path)))
            except Exception as error:
                return (
                    (),
                    tuple([f"Could not read a text sidecar: {error}"]),
                    (),
                )
        notes.append(
            "Guided mode aligned one text sidecar file per media file."
        )
        return tuple(labels), tuple(warnings), tuple(notes)
    warnings.append(
        "Guided mode could not align the text sidecars with the media files."
    )
    return (), tuple(warnings), tuple(notes)


def _resolve_media_labels_from_tabular_sidecar(
    media_files: tuple[str, ...],
    sidecar_files: tuple[str, ...],
):
    warnings = []
    notes = []
    if len(sidecar_files) != 1:
        warnings.append(
            "Guided mode expected one tabular sidecar file for the media collection."
        )
        return (), (), tuple(warnings), tuple(notes)
    try:
        dataframe = _read_tabular_dataframe(sidecar_files[0])
    except Exception as error:
        warnings.append(f"Could not inspect the tabular sidecar: {error}")
        return (), (), tuple(warnings), tuple(notes)
    column_names = tuple(str(column).strip() for column in dataframe.columns)
    label_candidates = _dataframe_label_candidates(dataframe)
    label_column = label_candidates[0] if label_candidates else None
    if label_column is None:
        non_path_columns = [
            column_name
            for column_name in column_names
            if not _DROP_NAME_PATTERN.search(column_name.lower())
        ]
        if len(non_path_columns) == 1:
            label_column = non_path_columns[0]
        elif len(column_names) == 1:
            label_column = column_names[0]
    if label_column is None:
        warnings.append(
            "Guided mode could not infer a label column from the tabular sidecar."
        )
        return (), column_names, tuple(warnings), tuple(notes)
    key_column = next(
        (
            column_name
            for column_name in column_names
            if re.search(
                r"(^|[^a-z0-9])(path|file|filename|file_name|audio|image|video|media)([^a-z0-9]|$)",
                column_name.lower(),
            )
        ),
        None,
    )
    labels = []
    if key_column is not None:
        label_by_key = {}
        for _, row in dataframe.iterrows():
            row_key = _normalize_optional_text(row.get(key_column))
            row_label = _normalize_optional_text(row.get(label_column))
            if row_key is None or row_label is None:
                continue
            label_by_key.setdefault(Path(row_key).name, row_label)
            label_by_key.setdefault(Path(row_key).stem, row_label)
            label_by_key.setdefault(row_key, row_label)
        for media_path in media_files:
            media_name = Path(media_path).name
            media_stem = Path(media_path).stem
            resolved_label = (
                label_by_key.get(media_name)
                or label_by_key.get(media_stem)
                or label_by_key.get(media_path)
            )
            if resolved_label is None:
                labels = []
                break
            labels.append(resolved_label)
        if labels:
            notes.append(
                f"Guided mode aligned the tabular sidecar using the '{key_column}' column."
            )
            return tuple(labels), column_names, tuple(warnings), tuple(notes)
    if len(dataframe) == len(media_files):
        notes.append("Guided mode aligned the tabular sidecar by row order.")
        return (
            tuple(
                _normalize_label_text(value)
                for value in dataframe[label_column].tolist()
            ),
            column_names,
            tuple(warnings),
            tuple(notes),
        )
    warnings.append(
        "Guided mode could not match the tabular sidecar rows to the media files."
    )
    return (), column_names, tuple(warnings), tuple(notes)


def _inspect_media_files(
    source_files: tuple[str, ...],
    family: str,
    *,
    resolving_choice: str | None,
):
    warnings = []
    unresolved_questions = []
    notes = [
        f"Guided mode detected a local {family} collection and sampled the file structure before training."
    ]
    label_files: tuple[str, ...] = ()
    resolving_question = None
    parent_labels = _parent_folder_labels(source_files)
    if parent_labels is not None:
        label_files, label_dir = _materialize_label_sidecars(
            parent_labels,
            stem=f"{family}_folder_labels",
        )
        notes.append(
            f"Guided mode inferred labels from parent folders and materialized them under {label_dir}."
        )
    elif resolving_choice == "continue-unlabeled":
        notes.append(
            "Guided mode will keep the media files without explicit labels for this first pass."
        )
    elif resolving_choice == "review-manually":
        unresolved_questions.append(
            "Guided mode stopped before training because the media collection needs manual label routing."
        )
    else:
        resolving_question = _build_media_label_resolving_question(family)
        unresolved_questions.append(resolving_question.prompt)
    return {
        "feature_files": source_files,
        "label_files": label_files,
        "column_names": (),
        "label_candidates": (),
        "selected_label_columns": (),
        "drop_candidates": (),
        "row_count": len(source_files),
        "selected_rows": None,
        "resolving_question": resolving_question,
        "source_mode": f"local-{family}",
        "warnings": tuple(warnings),
        "unresolved_questions": tuple(unresolved_questions),
        "notes": tuple(notes),
    }


def _inspect_media_sidecar_files(
    media_files: tuple[str, ...],
    sidecar_files: tuple[str, ...],
    *,
    family: str,
    sidecar_family: str,
    resolving_choice: str | None,
):
    warnings = []
    unresolved_questions = []
    notes = []
    label_values: tuple[object, ...] = ()
    column_names: tuple[str, ...] = ()
    if sidecar_family == "text":
        label_values, sidecar_warnings, sidecar_notes = (
            _resolve_media_labels_from_text_sidecars(
                media_files,
                sidecar_files,
            )
        )
        warnings.extend(sidecar_warnings)
        notes.extend(sidecar_notes)
    else:
        label_values, column_names, sidecar_warnings, sidecar_notes = (
            _resolve_media_labels_from_tabular_sidecar(
                media_files,
                sidecar_files,
            )
        )
        warnings.extend(sidecar_warnings)
        notes.extend(sidecar_notes)
    if label_values:
        label_files, label_dir = _materialize_label_sidecars(
            tuple(label_values),
            stem=f"{family}_sidecar_labels",
        )
        notes.append(
            f"Guided mode normalized the sidecar labels into managed CSV files under {label_dir}."
        )
        return {
            "feature_files": media_files,
            "label_files": label_files,
            "column_names": column_names,
            "label_candidates": (),
            "selected_label_columns": (),
            "drop_candidates": (),
            "row_count": len(media_files),
            "selected_rows": None,
            "resolving_question": None,
            "source_mode": f"local-{family}-sidecar",
            "warnings": tuple(warnings),
            "unresolved_questions": tuple(unresolved_questions),
            "notes": tuple(notes),
        }
    fallback_inspection = _inspect_media_files(
        media_files,
        family,
        resolving_choice=resolving_choice,
    )
    notes.extend(fallback_inspection["notes"])
    warnings.extend(fallback_inspection["warnings"])
    unresolved_questions.extend(fallback_inspection["unresolved_questions"])
    return {
        **fallback_inspection,
        "column_names": column_names,
        "source_mode": f"local-{family}-sidecar",
        "warnings": tuple(dict.fromkeys(warnings)),
        "unresolved_questions": tuple(dict.fromkeys(unresolved_questions)),
        "notes": tuple(dict.fromkeys(notes)),
    }


def _inspect_remote_dataset(
    remote_src: str,
    source_type: str,
    revision: str | None,
    *,
    hosted_runtime: str,
    resolving_choice: str | None,
):
    warnings = []
    unresolved_questions = []
    notes = []
    row_count = None
    column_names = ()
    label_candidates = ()
    selected_label_columns = ()
    drop_candidates = ()
    selected_rows = None
    resolving_question = None
    try:
        from definers.data.loaders import fetch_dataset

        preview_row_limit = _hosted_preview_row_limit(hosted_runtime)

        dataset = fetch_dataset(
            remote_src,
            source_type,
            revision,
            sample_rows=preview_row_limit,
        )
    except Exception as error:
        unresolved_questions.append(
            f"Could not inspect the remote dataset: {error}"
        )
        return {
            "row_count": None,
            "column_names": (),
            "label_candidates": (),
            "selected_label_columns": (),
            "drop_candidates": (),
            "selected_rows": None,
            "resolving_question": None,
            "warnings": tuple(warnings),
            "unresolved_questions": tuple(unresolved_questions),
            "notes": tuple(notes),
        }
    if dataset is None:
        unresolved_questions.append(
            "The remote dataset could not be loaded for inspection."
        )
        return {
            "row_count": None,
            "column_names": (),
            "label_candidates": (),
            "selected_label_columns": (),
            "drop_candidates": (),
            "selected_rows": None,
            "resolving_question": None,
            "warnings": tuple(warnings),
            "unresolved_questions": tuple(unresolved_questions),
            "notes": tuple(notes),
        }
    row_count = _dataset_row_count(dataset)
    if (
        hosted_runtime != "local"
        and preview_row_limit is not None
        and row_count is not None
        and row_count >= preview_row_limit
    ):
        notes.append(
            f"Guided mode sampled the first {preview_row_limit} remote rows for {_hosted_runtime_label(hosted_runtime)} preview inspection."
        )
    column_names = _dataset_column_names(dataset)
    label_candidates = _dataset_label_candidates(dataset, column_names)
    drop_candidates = _dataset_drop_candidates(dataset, column_names)
    if label_candidates:
        selected_label_columns = (label_candidates[0],)
        notes.append(
            f"Guided mode selected '{label_candidates[0]}' as the first label candidate."
        )
    elif column_names:
        warnings.append(
            "No low-cardinality label column was inferred from the remote dataset sample."
        )
    return {
        "row_count": row_count,
        "column_names": column_names,
        "label_candidates": label_candidates,
        "selected_label_columns": selected_label_columns,
        "drop_candidates": drop_candidates,
        "selected_rows": selected_rows,
        "resolving_question": resolving_question,
        "warnings": tuple(warnings),
        "unresolved_questions": tuple(unresolved_questions),
        "notes": tuple(notes),
    }


def _inspect_local_files(
    source_files: tuple[str, ...],
    source_families: tuple[str, ...],
    *,
    resolving_choice: str | None,
    hosted_runtime: str,
):
    warnings = []
    unresolved_questions = []
    notes = []
    feature_files = source_files
    label_files: tuple[str, ...] = ()
    column_names: tuple[str, ...] = ()
    label_candidates: tuple[str, ...] = ()
    selected_label_columns: tuple[str, ...] = ()
    drop_candidates: tuple[str, ...] = ()
    row_count = len(source_files) if source_files else None
    selected_rows = None
    resolving_question = None
    if len(source_families) > 1:
        grouped_paths = _group_paths_by_family(source_files)
        media_families = [
            family
            for family in source_families
            if family in {"audio", "image", "video"}
        ]
        sidecar_families = [
            family
            for family in source_families
            if family in {"tabular", "text"}
        ]
        if (
            len(media_families) == 1
            and len(sidecar_families) == 1
            and len(source_families) == 2
        ):
            return _inspect_media_sidecar_files(
                grouped_paths.get(media_families[0], ()),
                grouped_paths.get(sidecar_families[0], ()),
                family=media_families[0],
                sidecar_family=sidecar_families[0],
                resolving_choice=resolving_choice,
            )
        dominant_family = _dominant_local_family(source_files)
        if (
            resolving_choice == "keep-dominant-family"
            and dominant_family is not None
        ):
            dominant_paths = grouped_paths.get(dominant_family, ())
            dominant_result = _inspect_local_files(
                dominant_paths,
                (dominant_family,),
                resolving_choice=None,
                hosted_runtime=hosted_runtime,
            )
            notes = list(dominant_result["notes"])
            notes.append(
                f"Guided mode ignored the non-{dominant_family} files for this first pass."
            )
            return {
                **dominant_result,
                "notes": tuple(dict.fromkeys(notes)),
            }
        resolving_question = _build_dominant_family_resolving_question(
            dominant_family or "supported"
        )
        unresolved_questions.append(resolving_question.prompt)
        return {
            "feature_files": feature_files,
            "label_files": label_files,
            "column_names": column_names,
            "label_candidates": label_candidates,
            "selected_label_columns": selected_label_columns,
            "drop_candidates": drop_candidates,
            "row_count": row_count,
            "selected_rows": selected_rows,
            "resolving_question": resolving_question,
            "source_mode": "mixed-local",
            "warnings": tuple(warnings),
            "unresolved_questions": tuple(unresolved_questions),
            "notes": tuple(notes),
        }
    if not source_families:
        return {
            "feature_files": feature_files,
            "label_files": label_files,
            "column_names": column_names,
            "label_candidates": label_candidates,
            "selected_label_columns": selected_label_columns,
            "drop_candidates": drop_candidates,
            "row_count": row_count,
            "selected_rows": selected_rows,
            "resolving_question": resolving_question,
            "source_mode": "empty",
            "warnings": tuple(warnings),
            "unresolved_questions": tuple(unresolved_questions),
            "notes": tuple(notes),
        }
    family = source_families[0]
    if family == "tabular":
        inferred = _split_tabular_files(
            source_files,
            resolving_choice=resolving_choice,
        )
        feature_files = inferred["feature_files"]
        label_files = inferred["label_files"]
        resolving_question = inferred["resolving_question"]
        warnings.extend(inferred["warnings"])
        unresolved_questions.extend(inferred["unresolved_questions"])
        if len(feature_files) == 1:
            preview = _inspect_tabular_file(feature_files[0])
            column_names = preview["column_names"]
            label_candidates = preview["label_candidates"]
            selected_label_columns = preview["selected_label_columns"]
            drop_candidates = preview["drop_candidates"]
            row_count = preview["row_count"]
            warnings.extend(preview["warnings"])
        elif feature_files:
            row_count = len(feature_files)
            notes.append(
                "Multiple tabular feature files were detected. Guided mode will keep them as local feature inputs."
            )
    elif family in {"audio", "image", "video"}:
        return _inspect_media_files(
            source_files,
            family,
            resolving_choice=resolving_choice,
        )
    elif family == "text":
        notes.append(
            f"Guided mode detected a local {family} training route and will keep the files as feature inputs."
        )
        row_count = len(feature_files)
    else:
        unresolved_questions.append(
            "Guided mode does not recognize these files as a supported beginner training route."
        )
    return {
        "feature_files": feature_files,
        "label_files": label_files,
        "column_names": column_names,
        "label_candidates": label_candidates,
        "selected_label_columns": selected_label_columns,
        "drop_candidates": drop_candidates,
        "row_count": row_count,
        "selected_rows": selected_rows,
        "resolving_question": resolving_question,
        "source_mode": f"local-{family}",
        "warnings": tuple(warnings),
        "unresolved_questions": tuple(unresolved_questions),
        "notes": tuple(notes),
    }


def _split_tabular_files(
    paths: tuple[str, ...],
    *,
    resolving_choice: str | None,
):
    warnings = []
    unresolved_questions = []
    resolving_question = None
    label_files = [path for path in paths if _looks_like_label_file(path)]
    if label_files:
        feature_files = tuple(path for path in paths if path not in label_files)
        if not feature_files:
            unresolved_questions.append(
                "Only label-like files were detected. Add feature files or switch to the advanced workbench."
            )
        return {
            "feature_files": feature_files,
            "label_files": tuple(label_files),
            "resolving_question": None,
            "warnings": tuple(warnings),
            "unresolved_questions": tuple(unresolved_questions),
        }
    if len(paths) == 2:
        first_preview = _inspect_tabular_file(paths[0])
        second_preview = _inspect_tabular_file(paths[1])
        first_width = len(first_preview["column_names"])
        second_width = len(second_preview["column_names"])
        if first_width == 1 and second_width > 1:
            return {
                "feature_files": (paths[1],),
                "label_files": (paths[0],),
                "resolving_question": None,
                "warnings": tuple(warnings),
                "unresolved_questions": tuple(unresolved_questions),
            }
        if second_width == 1 and first_width > 1:
            return {
                "feature_files": (paths[0],),
                "label_files": (paths[1],),
                "resolving_question": None,
                "warnings": tuple(warnings),
                "unresolved_questions": tuple(unresolved_questions),
            }
        if resolving_choice == "first-file-labels":
            return {
                "feature_files": (paths[1],),
                "label_files": (paths[0],),
                "resolving_question": None,
                "warnings": tuple(warnings),
                "unresolved_questions": tuple(unresolved_questions),
            }
        if resolving_choice == "second-file-labels":
            return {
                "feature_files": (paths[0],),
                "label_files": (paths[1],),
                "resolving_question": None,
                "warnings": tuple(warnings),
                "unresolved_questions": tuple(unresolved_questions),
            }
    if len(paths) > 1:
        if len(paths) == 2 and resolving_choice != "review-manually":
            resolving_question = _build_tabular_label_resolving_question(paths)
            unresolved_questions.append(resolving_question.prompt)
        else:
            unresolved_questions.append(
                "Multiple tabular files were uploaded but guided mode could not tell which one contains labels."
            )
    return {
        "feature_files": paths,
        "label_files": (),
        "resolving_question": resolving_question,
        "warnings": tuple(warnings),
        "unresolved_questions": tuple(unresolved_questions),
    }


def _looks_like_label_file(path: str) -> bool:
    normalized_name = Path(path).stem.strip().lower()
    return bool(_LABEL_NAME_PATTERN.search(normalized_name))


def _inspect_tabular_file(path: str):
    import pandas

    warnings = []
    row_count = None
    label_candidates: tuple[str, ...] = ()
    selected_label_columns: tuple[str, ...] = ()
    drop_candidates: tuple[str, ...] = ()
    dataframe = None
    extension = _path_extension(path)
    try:
        if extension == "csv":
            dataframe = pandas.read_csv(path)
            if dataframe.empty:
                dataframe = pandas.read_csv(path, header=None)
            row_count = _csv_row_count(path)
        elif extension == "xlsx":
            from definers import optional_dependencies

            optional_dependencies.ensure_module_runtime("openpyxl")
            dataframe = pandas.read_excel(path)
            row_count = len(dataframe)
        else:
            dataframe = pandas.read_json(path)
            row_count = len(dataframe)
    except Exception as error:
        warnings.append(f"Could not inspect tabular file '{path}': {error}")
        return {
            "column_names": (),
            "label_candidates": (),
            "selected_label_columns": (),
            "drop_candidates": (),
            "row_count": row_count,
            "warnings": tuple(warnings),
        }
    column_names = tuple(str(column) for column in dataframe.columns)
    label_candidates = _dataframe_label_candidates(dataframe)
    drop_candidates = _dataframe_drop_candidates(dataframe)
    if label_candidates:
        selected_label_columns = (label_candidates[0],)
    return {
        "column_names": column_names,
        "label_candidates": label_candidates,
        "selected_label_columns": selected_label_columns,
        "drop_candidates": drop_candidates,
        "row_count": row_count if row_count is not None else len(dataframe),
        "warnings": tuple(warnings),
    }


def _dataframe_label_candidates(dataframe) -> tuple[str, ...]:
    candidates = []
    for column in dataframe.columns:
        series = dataframe[column]
        if series.empty:
            continue
        normalized_name = str(column).strip()
        sampled_values = [
            value for value in series.head(128).tolist() if value == value
        ]
        unique_values = []
        for value in sampled_values:
            text = str(value).strip()
            if text and text not in unique_values:
                unique_values.append(text)
        if _LABEL_NAME_PATTERN.search(normalized_name.lower()):
            candidates.append(normalized_name)
            continue
        if len(unique_values) <= 1 or len(unique_values) > 20:
            continue
        if len(unique_values) >= max(2, min(8, len(sampled_values))):
            continue
        if series.dtype.kind in {"b", "i", "u", "O"}:
            candidates.append(normalized_name)
    return tuple(candidates)


def _dataframe_drop_candidates(dataframe) -> tuple[str, ...]:
    candidates = []
    for column in dataframe.columns:
        normalized_name = str(column).strip()
        series = dataframe[column]
        sampled_values = [
            value for value in series.head(128).tolist() if value == value
        ]
        unique_values = []
        for value in sampled_values:
            text = str(value).strip()
            if text and text not in unique_values:
                unique_values.append(text)
        if len(unique_values) <= 1 and sampled_values:
            candidates.append(normalized_name)
            continue
        if _DROP_NAME_PATTERN.search(normalized_name.lower()) and len(
            unique_values
        ) >= min(len(sampled_values), 4):
            candidates.append(normalized_name)
    return tuple(dict.fromkeys(candidates))


def _dataset_row_count(dataset) -> int | None:
    if isinstance(dataset, dict):
        for value in dataset.values():
            if isinstance(value, (str, bytes)):
                continue
            try:
                return len(value)
            except Exception:
                continue
        return len(dataset) if dataset else None
    try:
        return len(dataset)
    except Exception:
        return None


def _dataset_column_names(dataset) -> tuple[str, ...]:
    try:
        column_names = tuple(getattr(dataset, "column_names", ()) or ())
    except Exception:
        column_names = ()
    if column_names:
        return tuple(str(value) for value in column_names)
    if isinstance(dataset, list) and dataset and isinstance(dataset[0], dict):
        return tuple(str(key) for key in dataset[0])
    if isinstance(dataset, dict):
        return tuple(str(key) for key in dataset)
    return ()


def _dataset_column_values(
    dataset, column_name: str, sample_size: int = 128
) -> list[object]:
    try:
        values = dataset[column_name][:sample_size]
        if hasattr(values, "tolist"):
            return list(values.tolist())
        return list(values)
    except Exception:
        if isinstance(dataset, list):
            return [
                row.get(column_name)
                for row in dataset[:sample_size]
                if isinstance(row, dict)
            ]
        if isinstance(dataset, dict):
            values = dataset.get(column_name, ())
            if isinstance(values, list):
                return list(values[:sample_size])
        return []


def _dataset_label_candidates(
    dataset, column_names: tuple[str, ...]
) -> tuple[str, ...]:
    candidates = []
    for column_name in column_names:
        sampled_values = [
            value
            for value in _dataset_column_values(dataset, column_name)
            if value == value
        ]
        unique_values = []
        for value in sampled_values:
            text = str(value).strip()
            if text and text not in unique_values:
                unique_values.append(text)
        if _LABEL_NAME_PATTERN.search(column_name.lower()):
            candidates.append(column_name)
            continue
        if len(unique_values) <= 1 or len(unique_values) > 20:
            continue
        if len(unique_values) >= max(2, min(8, len(sampled_values))):
            continue
        if all(
            isinstance(value, (str, int, bool)) and not isinstance(value, bytes)
            for value in sampled_values
        ):
            candidates.append(column_name)
    return tuple(candidates)


def _dataset_drop_candidates(
    dataset, column_names: tuple[str, ...]
) -> tuple[str, ...]:
    candidates = []
    for column_name in column_names:
        sampled_values = [
            value
            for value in _dataset_column_values(dataset, column_name)
            if value == value
        ]
        unique_values = []
        for value in sampled_values:
            text = str(value).strip()
            if text and text not in unique_values:
                unique_values.append(text)
        if len(unique_values) <= 1 and sampled_values:
            candidates.append(column_name)
            continue
        if _DROP_NAME_PATTERN.search(column_name.lower()) and len(
            unique_values
        ) >= min(len(sampled_values), 4):
            candidates.append(column_name)
    return tuple(dict.fromkeys(candidates))


def _csv_row_count(path: str) -> int | None:
    try:
        with open(path, encoding="utf-8", errors="ignore") as file_obj:
            total_lines = sum(1 for _ in file_obj)
    except Exception:
        return None
    return max(total_lines - 1, 0)


def _suggest_splits(row_count: int | None) -> tuple[float, float]:
    if row_count is None or row_count < 20:
        return 0.0, 0.0
    if row_count < 200:
        return 0.1, 0.0
    if row_count < 2000:
        return 0.1, 0.1
    if row_count > 100000:
        return 0.05, 0.05
    return 0.1, 0.1


def _suggest_batch_size(
    row_count: int | None,
    source_families: tuple[str, ...],
) -> int:
    if source_families and source_families[0] in {"audio", "image", "video"}:
        return 8
    if row_count is None or row_count < 64:
        return 8
    if row_count < 1000:
        return 16
    if row_count < 10000:
        return 32
    return 64


def _suggest_selected_rows(
    row_count: int | None,
    source_mode: str,
) -> str | None:
    if row_count is None or row_count < 50000:
        return None
    if source_mode not in {"remote-dataset", "local-tabular"}:
        return None
    if row_count >= 250000:
        return "1-50000"
    if row_count >= 100000:
        return "1-25000"
    return "1-10000"


def _hosted_preview_row_limit(hosted_runtime: str) -> int | None:
    try:
        from definers.system.runtime_budget import hosted_preview_row_limit

        return hosted_preview_row_limit(hosted_runtime)
    except Exception:
        return None


def _hosted_guided_row_limit(hosted_runtime: str) -> int | None:
    try:
        from definers.system.runtime_budget import hosted_guided_row_limit

        return hosted_guided_row_limit(hosted_runtime)
    except Exception:
        return None


def _hosted_guided_media_file_limit(hosted_runtime: str) -> int | None:
    try:
        from definers.system.runtime_budget import (
            hosted_guided_media_file_limit,
        )

        return hosted_guided_media_file_limit(hosted_runtime)
    except Exception:
        return None


def _apply_hosted_runtime_budget(
    *,
    hosted_runtime: str,
    source_mode: str,
    feature_files: tuple[str, ...],
    label_files: tuple[str, ...],
    row_count: int | None,
    selected_rows: str | None,
    resolving_choice: str | None,
    resolving_question: TrainCoachResolvingQuestion | None,
):
    warnings = []
    unresolved_questions = []
    notes = []
    resolved_feature_files = feature_files
    resolved_label_files = label_files
    resolved_selected_rows = selected_rows
    budget_question = None
    if hosted_runtime == "local" or resolving_question is not None:
        return {
            "feature_files": resolved_feature_files,
            "label_files": resolved_label_files,
            "selected_rows": resolved_selected_rows,
            "resolving_question": budget_question,
            "warnings": tuple(warnings),
            "unresolved_questions": tuple(unresolved_questions),
            "notes": tuple(notes),
        }
    row_limit = _hosted_guided_row_limit(hosted_runtime)
    media_limit = _hosted_guided_media_file_limit(hosted_runtime)
    if (
        source_mode in {"remote-dataset", "local-tabular"}
        and row_limit is not None
        and row_count is not None
        and row_count > row_limit
    ):
        if resolving_choice == "sample-first-pass":
            resolved_selected_rows = f"1-{row_limit}"
            notes.append(
                f"Guided mode constrained the first pass to rows 1-{row_limit} for {_hosted_runtime_label(hosted_runtime)}."
            )
        elif resolving_choice == "review-outside-host":
            unresolved_questions.append(
                f"This dataset is larger than the guided hosted budget for {_hosted_runtime_label(hosted_runtime)}."
            )
        else:
            budget_question = _build_hosted_sampling_resolving_question(
                hosted_runtime,
                row_limit,
                "rows",
            )
            unresolved_questions.append(budget_question.prompt)
    if (
        any(family in source_mode for family in ("audio", "image", "video"))
        and media_limit is not None
        and len(resolved_feature_files) > media_limit
    ):
        if resolving_choice == "sample-first-pass":
            resolved_feature_files = resolved_feature_files[:media_limit]
            if resolved_label_files:
                resolved_label_files = resolved_label_files[:media_limit]
            notes.append(
                f"Guided mode kept the first {media_limit} media files for {_hosted_runtime_label(hosted_runtime)}."
            )
        elif resolving_choice == "review-outside-host":
            unresolved_questions.append(
                f"This media collection is larger than the guided hosted budget for {_hosted_runtime_label(hosted_runtime)}."
            )
        elif budget_question is None:
            budget_question = _build_hosted_sampling_resolving_question(
                hosted_runtime,
                media_limit,
                "files",
            )
            unresolved_questions.append(budget_question.prompt)
    return {
        "feature_files": resolved_feature_files,
        "label_files": resolved_label_files,
        "selected_rows": resolved_selected_rows,
        "resolving_question": budget_question,
        "warnings": tuple(warnings),
        "unresolved_questions": tuple(unresolved_questions),
        "notes": tuple(notes),
    }


def _build_recommendations(
    *,
    selected_label_columns: tuple[str, ...],
    label_candidates: tuple[str, ...],
    drop_candidates: tuple[str, ...],
    source_families: tuple[str, ...],
    source_mode: str,
    row_count: int | None,
    selected_rows: str | None,
    suggested_batch_size: int,
    suggested_validation_split: float,
    suggested_test_split: float,
    stratify: str | None,
    hosted_runtime: str,
    resume_guidance: TrainCoachResumeGuidance,
) -> tuple[TrainCoachRecommendation, ...]:
    recommendations = []
    if selected_label_columns:
        recommendations.append(
            TrainCoachRecommendation(
                name="label_columns",
                value=selected_label_columns,
                reason=(
                    "Restored from the previous train session manifest."
                    if resume_guidance.recovered_label_columns
                    and selected_label_columns
                    == resume_guidance.recovered_label_columns
                    else "Column name and sampled values suggest a stable target column."
                    if any(
                        _LABEL_NAME_PATTERN.search(column.lower())
                        for column in selected_label_columns
                    )
                    else "Low-cardinality sampled values suggest a target column."
                ),
                confidence=(
                    "high"
                    if any(
                        _LABEL_NAME_PATTERN.search(column.lower())
                        for column in selected_label_columns
                    )
                    or resume_guidance.recovered_label_columns
                    else "medium"
                ),
                applied=True,
            )
        )
    resolved_drop_candidates = tuple(
        candidate
        for candidate in drop_candidates
        if candidate not in selected_label_columns
    )
    if resolved_drop_candidates:
        recommendations.append(
            TrainCoachRecommendation(
                name="drop_columns",
                value=resolved_drop_candidates,
                reason="These columns look constant or identifier-like, so they are better treated as metadata than model features.",
                confidence=(
                    "high"
                    if all(
                        _DROP_NAME_PATTERN.search(column.lower())
                        for column in resolved_drop_candidates
                    )
                    else "medium"
                ),
                applied=False,
            )
        )
    recommendations.append(
        TrainCoachRecommendation(
            name="batch_size",
            value=suggested_batch_size,
            reason=(
                "Media-style inputs keep batches smaller to stay inside beginner-safe runtime budgets."
                if source_families
                and source_families[0] in {"audio", "image", "video"}
                else "Dataset size suggests this batch size as a safe first pass."
            ),
            confidence="high" if row_count is not None else "medium",
            applied=True,
        )
    )
    recommendations.append(
        TrainCoachRecommendation(
            name="validation_split",
            value=suggested_validation_split,
            reason=(
                "Small datasets stay fully available for fitting until there is enough data for a stable validation slice."
                if suggested_validation_split == 0.0
                else "Dataset size is large enough to hold out validation data without starving training."
            ),
            confidence="high" if row_count is not None else "medium",
            applied=True,
        )
    )
    recommendations.append(
        TrainCoachRecommendation(
            name="test_split",
            value=suggested_test_split,
            reason=(
                "A separate test split is skipped until the dataset is large enough to support it safely."
                if suggested_test_split == 0.0
                else "Dataset size is large enough to keep an unbiased test sample."
            ),
            confidence="high" if row_count is not None else "medium",
            applied=True,
        )
    )
    if stratify is not None:
        recommendations.append(
            TrainCoachRecommendation(
                name="stratify",
                value=stratify,
                reason="The inferred label column has enough rows to keep class balance stable across splits.",
                confidence="high",
                applied=True,
            )
        )
    recommended_selected_rows = selected_rows or _suggest_selected_rows(
        row_count,
        source_mode,
    )
    if recommended_selected_rows is not None:
        recommendations.append(
            TrainCoachRecommendation(
                name="selected_rows",
                value=recommended_selected_rows,
                reason=(
                    f"{_hosted_runtime_label(hosted_runtime)} is using a smaller first pass to stay inside hosted runtime budgets."
                    if selected_rows is not None and hosted_runtime != "local"
                    else "A sampled first pass can shorten beginner iteration time when the dataset is very large."
                ),
                confidence=(
                    "high"
                    if selected_rows is not None and hosted_runtime != "local"
                    else "medium"
                    if source_mode == "remote-dataset"
                    else "low"
                ),
                applied=selected_rows is not None,
            )
        )
    if resume_guidance.strategy != "none":
        recommendations.append(
            TrainCoachRecommendation(
                name="resume_strategy",
                value=_humanize_resume_strategy(resume_guidance.strategy),
                reason=resume_guidance.detail,
                confidence=resume_guidance.confidence,
                applied=resume_guidance.use_resume_artifact,
            )
        )
    if not selected_label_columns and label_candidates:
        recommendations.append(
            TrainCoachRecommendation(
                name="label_candidates",
                value=label_candidates,
                reason="Guided mode found plausible target columns but could not apply one with enough confidence.",
                confidence="low",
                applied=False,
            )
        )
    return tuple(recommendations)


def _report_guided_activity(
    activity_reporter, completed: int, title: str, detail: str
):
    if activity_reporter is None:
        return None
    try:
        return activity_reporter(completed, title, detail=detail)
    except Exception:
        return None


def _resume_source_family(source_mode: str) -> str:
    normalized_mode = str(source_mode or "").strip().lower()
    if normalized_mode == "remote-dataset" or normalized_mode.endswith(
        "tabular"
    ):
        return "tabular"
    for family in ("text", "audio", "image", "video"):
        if family in normalized_mode:
            return family
    if normalized_mode in {"mixed-source", "mixed-local"}:
        return "mixed"
    return normalized_mode or "unknown"


def _source_modes_compatible(current_mode: str, previous_mode: str) -> bool:
    current_family = _resume_source_family(current_mode)
    previous_family = _resume_source_family(previous_mode)
    if current_family in {"unknown", "resume-only", "empty"}:
        return True
    if previous_family in {"unknown", "resume-only", "empty"}:
        return True
    return current_family == previous_family


def _build_resume_guidance(
    *,
    resume_from: str | None,
    source_mode: str,
    source_type: str,
    column_names: tuple[str, ...],
    selected_label_columns: tuple[str, ...],
) -> TrainCoachResumeGuidance:
    if resume_from is None:
        return TrainCoachResumeGuidance(
            strategy="none",
            confidence="high",
            detail="No previous model artifact was supplied.",
            use_resume_artifact=False,
            manifest_path=None,
            revalidation=(),
        )
    from .coach_manifest import load_train_artifact_manifest

    manifest = load_train_artifact_manifest(resume_from)
    if manifest is None:
        return TrainCoachResumeGuidance(
            strategy="re-fit",
            confidence="low",
            detail="Definers found a previous model artifact but no saved train session manifest, so it must revalidate the new route before reusing the artifact.",
            use_resume_artifact=True,
            manifest_path=None,
            revalidation=("label columns", "source type", "splits"),
        )
    inspection_report = manifest.get("inspection_report", {})
    if not isinstance(inspection_report, dict):
        inspection_report = {}
    normalized_request = manifest.get("normalized_request", {})
    if not isinstance(normalized_request, dict):
        normalized_request = {}
    previous_source_mode = str(inspection_report.get("source_mode", "")).strip()
    previous_source_type = str(
        inspection_report.get("source_type")
        or normalized_request.get("source_type")
        or ""
    ).strip()
    previous_label_columns = tuple(
        str(value)
        for value in inspection_report.get("selected_label_columns", ())
        if str(value).strip()
    )
    recovered_label_columns = tuple(
        column for column in previous_label_columns if column in column_names
    )
    manifest_path = _normalize_optional_text(
        manifest.get("session_manifest_path")
    )
    if not _source_modes_compatible(source_mode, previous_source_mode):
        return TrainCoachResumeGuidance(
            strategy="fresh-start",
            confidence="high",
            detail=f"The previous session used {previous_source_mode or 'a different'} inputs while the new route uses {source_mode}. Guided mode will start a fresh model instead of forcing an unsafe resume.",
            use_resume_artifact=False,
            manifest_path=manifest_path,
            revalidation=("source mode",),
        )
    if (
        previous_label_columns
        and selected_label_columns
        and not set(previous_label_columns).intersection(selected_label_columns)
    ):
        return TrainCoachResumeGuidance(
            strategy="fresh-start",
            confidence="high",
            detail="The previous model used different label columns than the new inspection. Guided mode will start a fresh model instead of reusing the artifact unsafely.",
            use_resume_artifact=False,
            manifest_path=manifest_path,
            revalidation=("label columns",),
        )
    if (
        previous_source_type
        and source_type
        and previous_source_type != source_type
    ):
        return TrainCoachResumeGuidance(
            strategy="re-fit",
            confidence="medium",
            detail="The previous session used a different dataset source type, so Definers will reuse the artifact only after revalidating the route.",
            use_resume_artifact=True,
            manifest_path=manifest_path,
            revalidation=("source type", "splits"),
            recovered_label_columns=recovered_label_columns,
        )
    if previous_label_columns and not (
        selected_label_columns or recovered_label_columns
    ):
        return TrainCoachResumeGuidance(
            strategy="re-fit",
            confidence="medium",
            detail="The previous session knew the target columns, but the new data has not confirmed them yet. Guided mode will reuse the artifact only after revalidating labels.",
            use_resume_artifact=True,
            manifest_path=manifest_path,
            revalidation=("label columns",),
            recovered_label_columns=recovered_label_columns,
        )
    return TrainCoachResumeGuidance(
        strategy="safe-continue",
        confidence="high",
        detail="The previous train session matches the current guided route, so Definers can continue safely.",
        use_resume_artifact=True,
        manifest_path=manifest_path,
        revalidation=("runtime readiness",),
        recovered_label_columns=recovered_label_columns,
    )


def _humanize_resume_strategy(strategy: str) -> str:
    return {
        "none": "No resume artifact",
        "safe-continue": "Continue Safely",
        "re-fit": "Re-Fit With Checks",
        "fresh-start": "Start Fresh",
    }.get(str(strategy or "none").strip().lower(), "Resume Review")


def _recommendation_value_text(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "none"
    return str(value)


def _build_checks(
    *,
    health_snapshot=None,
    source_files: tuple[str, ...],
    remote_src: str | None,
    resume_from: str | None,
    resume_guidance: TrainCoachResumeGuidance,
    resolving_question: TrainCoachResolvingQuestion | None,
    hosted_runtime: str,
    unresolved_questions: tuple[str, ...],
    source_mode: str,
) -> tuple[TrainCoachCheck, ...]:
    snapshot = health_snapshot
    if snapshot is None:
        from definers.ml import get_ml_health_snapshot

        snapshot = get_ml_health_snapshot()
    checks = [
        TrainCoachCheck(
            name="Runtime Ready",
            ok=bool(
                getattr(snapshot, "training_ready", False)
                and getattr(snapshot, "data_preparation_ready", False)
            ),
            detail=(
                "Training and data preparation capabilities are available."
                if getattr(snapshot, "training_ready", False)
                and getattr(snapshot, "data_preparation_ready", False)
                else "Install the missing ML runtime dependencies before guided training."
            ),
        ),
        TrainCoachCheck(
            name="Source Selected",
            ok=bool(source_files or remote_src),
            detail=(
                "A trainable source was detected."
                if source_files or remote_src
                else "Guided mode needs files or a remote dataset before training can continue."
            ),
        ),
        TrainCoachCheck(
            name="Guided Route Resolved",
            ok=(not unresolved_questions and resolving_question is None),
            detail=(
                "Guided mode resolved a safe route."
                if not unresolved_questions and resolving_question is None
                else resolving_question.prompt
                if resolving_question is not None
                else unresolved_questions[0]
            ),
        ),
        TrainCoachCheck(
            name="Resume Artifact",
            ok=(
                resume_from is None
                or str(resume_from).lower().endswith(".joblib")
            ),
            detail=(
                "No resume artifact was supplied."
                if resume_from is None
                else (
                    "A compatible .joblib artifact is ready to resume."
                    if str(resume_from).lower().endswith(".joblib")
                    else "Guided resume currently supports .joblib artifacts only."
                )
            ),
        ),
    ]
    if resume_from is not None:
        checks.append(
            TrainCoachCheck(
                name="Resume Strategy",
                ok=resume_guidance.use_resume_artifact,
                detail=resume_guidance.detail,
                blocking=False,
            )
        )
    if hosted_runtime != "local":
        checks.append(
            TrainCoachCheck(
                name="Hosted Runtime",
                ok=True,
                detail=f"Guided mode is honoring the {_hosted_runtime_label(hosted_runtime)} runtime budget and retention policy.",
                blocking=False,
            )
        )
    if source_mode == "resume-only":
        checks.append(
            TrainCoachCheck(
                name="Fresh Data Present",
                ok=False,
                detail="Add files or connect a dataset before continuing yesterday's model.",
            )
        )
    if source_mode == "mixed-local" and resolving_question is None:
        checks.append(
            TrainCoachCheck(
                name="Mixed Local Inputs",
                ok=False,
                detail="Guided mode does not resolve mixed local file families yet.",
            )
        )
    return tuple(checks)
