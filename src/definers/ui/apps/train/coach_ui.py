from __future__ import annotations

from .coach import (
    build_train_coach_contract_markdown,
    train_coach_entry_intents,
    train_coach_step_names,
)


def _render_train_coach_steps_html() -> str:
    items = "".join(
        f"<li><strong>{step}</strong></li>" for step in train_coach_step_names()
    )
    return f'<section class="train-guided-steps"><ol>{items}</ol></section>'


def build_train_guided_mode(*, container, bind_action):
    from PySide6.QtWidgets import QPushButton

    from definers.ui.qt_widgets import (
        FilePicker,
        HtmlLabel,
        LabeledCombo,
        LabeledLineEdit,
        LabeledTextbox,
        MarkdownLabel,
        Row,
        Section,
    )

    from .coach_handlers import (
        _read_last_model_path,
        inspect_train_coach_request,
        preview_train_coach_plan,
        reset_train_coach_state,
        run_train_coach_auto_workflow,
        run_train_coach_workflow,
        strip_saved_model,
    )

    container.addWidget(HtmlLabel(_render_train_coach_steps_html()))
    container.addWidget(MarkdownLabel(build_train_coach_contract_markdown()))

    state_payload = LabeledTextbox("Guided State", value="", lines=2)
    state_payload.setVisible(False)
    container.addWidget(state_payload)

    intake_row = Row()
    left = Section()
    entry_intent_choices = tuple(
        (str(entry["title"]), str(entry["id"]))
        for entry in train_coach_entry_intents()
    )
    entry_intent = LabeledCombo(
        "Start Here",
        choices=tuple(label for label, _id in entry_intent_choices),
        value=next(
            (label for label, _id in entry_intent_choices if _id == "files"),
            entry_intent_choices[0][0] if entry_intent_choices else "",
        ),
    )
    uploaded_files = FilePicker("Files")
    remote_src = LabeledLineEdit(
        "Remote Dataset", placeholder="owner/dataset or https://..."
    )
    revision = LabeledLineEdit("Revision", placeholder="main")
    resume_artifact = FilePicker("Previous Model Artifact")
    last_model_path = _read_last_model_path()
    if last_model_path:
        resume_artifact.set_value(last_model_path)
    save_as = LabeledLineEdit(
        "Save Artifact As", placeholder="guided-model.joblib"
    )
    inspect_button = QPushButton("Inspect My Inputs")
    auto_train_button = QPushButton("Train Automatically")
    preview_button = QPushButton("Review Guided Plan")
    preview_button.setEnabled(False)
    train_button = QPushButton("Train With Guided Defaults")
    train_button.setEnabled(False)
    for widget in (
        entry_intent,
        uploaded_files,
        remote_src,
        revision,
        resume_artifact,
        save_as,
        inspect_button,
        auto_train_button,
        preview_button,
        train_button,
    ):
        left.addWidget(widget)

    right = Section()
    intake_summary = MarkdownLabel(
        "## Guided Intake\n- Status: waiting for inspection."
    )
    inspection_markdown = MarkdownLabel(
        "## Data Inspection\n- Status: add files or a dataset,"
        " then run inspection."
    )
    validation_markdown = MarkdownLabel(
        "## Guided Validation\n- Status: inspection pending."
    )
    right.addWidget(intake_summary)
    right.addWidget(inspection_markdown)
    right.addWidget(validation_markdown)
    intake_row.addWidget(left)
    intake_row.addWidget(right)
    container.addWidget(intake_row)

    plan_row = Row()

    left = Section()
    training_plan = MarkdownLabel(
        "## Training Plan\n- Guided inspection will unlock the plan preview."
    )
    training_status = MarkdownLabel("## Training\n- Status: idle")
    left.addWidget(training_plan)
    left.addWidget(training_status)

    right = Section()

    train_output = FilePicker("Training Output")
    strip_button = QPushButton("Strip Training Data")
    use_result_markdown = MarkdownLabel(
        "## Use Result\n- Status: train a model to unlock the saved session,"
        " manifest, and next actions."
    )
    right.addWidget(training_status)
    right.addWidget(train_output)
    right.addWidget(strip_button)
    right.addWidget(use_result_markdown)
    plan_row.addWidget(left)
    plan_row.addWidget(right)
    container.addWidget(plan_row)

    question_row = Row()
    left = Section()
    resolving_question_markdown = MarkdownLabel(
        "## Quick Decision\n- Status: no clarification is needed."
    )
    left.addWidget(resolving_question_markdown)
    right = Section()
    resolving_choice = LabeledCombo("Quick Decision", choices=(), value=None)
    resolving_choice.setVisible(False)
    right.addWidget(resolving_choice)
    question_row.addWidget(left)
    question_row.addWidget(right)
    container.addWidget(question_row)

    inspect_inputs = [
        entry_intent,
        uploaded_files,
        remote_src,
        revision,
        resume_artifact,
        save_as,
        resolving_choice,
    ]
    reset_outputs = [
        state_payload,
        intake_summary,
        inspection_markdown,
        validation_markdown,
        use_result_markdown,
        resolving_question_markdown,
        resolving_choice,
        preview_button,
        train_button,
    ]

    def _connect_reset(component, signal_attr):
        signal = getattr(component, signal_attr, None)
        if signal is None:
            return
        signal.connect(
            lambda *_: _apply_reset_outputs(
                reset_outputs,
                reset_train_coach_state(),
            )
        )

    from definers.ui.gui_shared import write_widget_value

    def _apply_reset_outputs(targets, values):
        for target, value in zip(targets, values):
            write_widget_value(target, value)

    if hasattr(entry_intent, "changed"):
        entry_intent.changed.connect(
            lambda *_: _apply_reset_outputs(
                reset_outputs, reset_train_coach_state()
            )
        )
    for component in (remote_src, revision, save_as):
        editor = getattr(component, "_editor", None)
        if editor is not None and hasattr(editor, "textChanged"):
            editor.textChanged.connect(
                lambda *_: _apply_reset_outputs(
                    reset_outputs, reset_train_coach_state()
                )
            )
    for picker in (uploaded_files, resume_artifact):
        if hasattr(picker, "selected"):
            picker.selected.connect(
                lambda *_: _apply_reset_outputs(
                    reset_outputs, reset_train_coach_state()
                )
            )

    bind_action(
        inspect_button,
        inspect_train_coach_request,
        inputs=inspect_inputs,
        outputs=reset_outputs,
        action_label="Inspect Inputs",
        steps=(
            "Check files",
            "Understand data",
            "Write recommendations",
            "Resolve one question",
        ),
        running_detail="Inspecting files, datasets, and resume artifacts.",
        success_detail="Guided inspection is ready.",
    )
    bind_action(
        auto_train_button,
        run_train_coach_auto_workflow,
        inputs=inspect_inputs,
        outputs=[
            state_payload,
            intake_summary,
            inspection_markdown,
            validation_markdown,
            use_result_markdown,
            resolving_question_markdown,
            resolving_choice,
            preview_button,
            train_button,
            train_output,
            training_plan,
            training_status,
            resume_artifact,
        ],
        action_label="Train Automatically",
        steps=(
            "Check files",
            "Understand data",
            "Resolve route",
            "Train model",
            "Save model",
            "Suggest next steps",
        ),
        running_detail=(
            "Inspecting the inputs and training when the guided route is ready."
        ),
        success_detail=(
            "Automatic guided training finished or reported the next decision."
        ),
    )
    bind_action(
        preview_button,
        preview_train_coach_plan,
        inputs=[state_payload],
        outputs=[training_plan],
        action_label="Review Guided Plan",
        steps=(
            "Check files",
            "Understand data",
            "Build plan",
        ),
        running_detail="Preparing the guided training plan.",
        success_detail="Guided plan preview is ready.",
    )
    bind_action(
        train_button,
        run_train_coach_workflow,
        inputs=[state_payload],
        outputs=[
            train_output,
            training_plan,
            training_status,
            use_result_markdown,
            resume_artifact,
        ],
        action_label="Train With Guided Defaults",
        steps=(
            "Check files",
            "Understand data",
            "Build plan",
            "Train model",
            "Save model",
            "Suggest next steps",
        ),
        running_detail="Running the guided training workflow.",
        success_detail="Guided training is complete.",
    )
    strip_button.clicked.connect(
        lambda *_: write_widget_value(
            train_output, strip_saved_model(train_output.value())
        )
    )


__all__ = ["build_train_guided_mode"]
