from __future__ import annotations

from dataclasses import dataclass


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


class TrainerPlanService:
    @staticmethod
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

    @staticmethod
    def detect_training_mode(*, is_remote_dataset, is_file_dataset) -> str:
        if is_remote_dataset:
            return "remote-dataset"
        if is_file_dataset:
            return "file-dataset"
        return "in-memory"

    @classmethod
    def build_training_plan(
        cls,
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
    ) -> TrainingPlan:
        return TrainingPlan(
            mode=cls.detect_training_mode(
                is_remote_dataset=is_remote_dataset,
                is_file_dataset=is_file_dataset,
            ),
            source_summary=cls.summarize_value(source),
            target_summary=cls.summarize_value(target),
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
        )

    @staticmethod
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
        ]
        return "\n".join(lines)


build_training_plan = TrainerPlanService.build_training_plan
render_training_plan_markdown = TrainerPlanService.render_training_plan_markdown
