from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingDefaults:
    batch_size: int
    validation_split: float
    test_split: float
    early_stopping: bool
    patience: int
    cv_folds: int
    notes: tuple[str, ...]


def suggest_training_defaults(
    n_samples: int | None,
    *,
    batch_size: int | None = None,
    validation_split: float | None = None,
    test_split: float | None = None,
    early_stopping: bool | None = None,
    cv_folds: int | None = None,
) -> TrainingDefaults:

    notes: list[str] = []
    samples = (
        int(n_samples) if isinstance(n_samples, int) and n_samples > 0 else 0
    )

    if batch_size is None:
        if samples == 0:
            chosen_batch = 32
        elif samples < 64:
            chosen_batch = max(1, samples)
        elif samples < 1024:
            chosen_batch = 32
        elif samples < 16384:
            chosen_batch = 64
        else:
            chosen_batch = 128
        if samples:
            notes.append(
                f"batch_size auto-tuned to {chosen_batch} for {samples} samples"
            )
    else:
        chosen_batch = int(batch_size)

    if validation_split is None or validation_split == 0.0:
        if samples >= 200:
            chosen_val = 0.1
            notes.append("validation_split auto-set to 0.1")
        elif samples >= 40:
            chosen_val = 0.2
            notes.append("validation_split auto-set to 0.2 (small dataset)")
        else:
            chosen_val = 0.0
    else:
        chosen_val = float(validation_split)

    if test_split is None:
        chosen_test = 0.0
    else:
        chosen_test = float(test_split)

    if early_stopping is None:
        chosen_early = chosen_val > 0.0
        if chosen_early:
            notes.append("early_stopping auto-enabled (validation set present)")
    else:
        chosen_early = bool(early_stopping)

    chosen_patience = 3 if chosen_early else 0

    if cv_folds is None:
        chosen_cv = 0
    else:
        chosen_cv = max(0, int(cv_folds))
        if chosen_cv == 1:
            chosen_cv = 0

    return TrainingDefaults(
        batch_size=chosen_batch,
        validation_split=chosen_val,
        test_split=chosen_test,
        early_stopping=chosen_early,
        patience=chosen_patience,
        cv_folds=chosen_cv,
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    mode: str
    source_summary: str
    target_summary: str
    batch_size: int
    source_type: str
    revision: str | None
    validation_split: float
    test_split: float
    label_columns: tuple[str, ...]
    drop_columns: tuple[str, ...]
    order_by: str | None
    stratify: str | None
    selected_rows: str | None
    resume_from: str | None
    early_stopping: bool = False
    patience: int = 0
    cv_folds: int = 0
    auto_tuned: tuple[str, ...] = ()


def summarize_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        preview_items = [str(item) for item in value[:3]]
        preview = ", ".join(preview_items)
        if len(value) > 3:
            preview += f", ... (+{len(value) - 3} more)"
        return preview or "empty"
    return str(value)


def detect_training_mode(*, is_remote_dataset, is_file_dataset) -> str:
    if is_remote_dataset:
        return "remote-dataset"
    if is_file_dataset:
        return "file-dataset"
    return "in-memory"


def build_training_plan(
    *,
    source,
    target,
    batch_size: int,
    source_type: str,
    revision: str | None,
    validation_split: float,
    test_split: float,
    label_columns,
    drop_columns,
    order_by: str | None,
    stratify: str | None,
    selected_rows: str | None,
    resume_from: str | None,
    is_remote_dataset: bool,
    is_file_dataset: bool,
    early_stopping: bool = False,
    patience: int = 0,
    cv_folds: int = 0,
    auto_tuned: tuple[str, ...] = (),
) -> TrainingPlan:
    return TrainingPlan(
        mode=detect_training_mode(
            is_remote_dataset=is_remote_dataset,
            is_file_dataset=is_file_dataset,
        ),
        source_summary=summarize_value(source),
        target_summary=summarize_value(target),
        batch_size=batch_size,
        source_type=source_type,
        revision=revision,
        validation_split=validation_split,
        test_split=test_split,
        label_columns=tuple(label_columns or ()),
        drop_columns=tuple(drop_columns or ()),
        order_by=order_by,
        stratify=stratify,
        selected_rows=selected_rows,
        resume_from=resume_from,
        early_stopping=bool(early_stopping),
        patience=int(patience or 0),
        cv_folds=int(cv_folds or 0),
        auto_tuned=tuple(auto_tuned or ()),
    )


def render_training_plan_markdown(plan: TrainingPlan) -> str:
    lines = [
        "## Training Plan",
        f"- Mode: {plan.mode}",
        f"- Source: {plan.source_summary}",
        f"- Target: {plan.target_summary}",
        f"- Batch Size: {plan.batch_size}",
        f"- Source Type: {plan.source_type}",
        f"- Revision: {plan.revision or 'default'}",
        f"- Validation Split: {plan.validation_split}",
        f"- Test Split: {plan.test_split}",
        f"- Label Columns: {', '.join(plan.label_columns) or 'none'}",
        f"- Drop Columns: {', '.join(plan.drop_columns) or 'none'}",
        f"- Order By: {plan.order_by or 'none'}",
        f"- Stratify: {plan.stratify or 'none'}",
        f"- Selected Rows: {plan.selected_rows or 'all'}",
        f"- Resume From: {plan.resume_from or 'none'}",
        f"- Early Stopping: {'on (patience=' + str(plan.patience) + ')' if plan.early_stopping else 'off'}",
        f"- Cross-Validation Folds: {plan.cv_folds or 'none'}",
    ]
    if plan.auto_tuned:
        lines.append("- Auto-Tuned:")
        for note in plan.auto_tuned:
            lines.append(f"  - {note}")
    return "\n".join(lines)
