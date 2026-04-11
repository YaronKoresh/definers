from __future__ import annotations

from .coach import (
    build_train_coach_contract_markdown,
    train_coach_entry_intents,
    train_coach_step_names,
)


def _render_train_coach_steps_html() -> str:
    items = "".join(
        f"<li><span>{index}</span><strong>{step}</strong></li>"
        for index, step in enumerate(train_coach_step_names(), start=1)
    )
    return f'<section class="train-guided-steps"><ol>{items}</ol></section>'


def train_coach_css() -> str:
    return """
.train-guided-steps ol {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    list-style: none;
    margin: 0 0 22px 0;
    padding: 0;
}

.train-guided-steps li {
    display: grid;
    gap: 8px;
    padding: 18px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(255, 253, 248, 0.98), rgba(252, 245, 235, 0.94));
    border: 1px solid rgba(214, 199, 180, 0.95);
    box-shadow: 0 18px 42px rgba(28, 25, 23, 0.06);
}

.train-guided-steps li span {
    display: inline-flex;
    width: 32px;
    height: 32px;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    background: rgba(15, 118, 110, 0.12);
    color: #115e59;
    font-weight: 700;
}

.train-guided-steps li strong {
    color: #111827;
}
"""


def build_train_guided_mode(*, bind_action):
    import gradio as gr

    from .coach_handlers import (
        inspect_train_coach_request,
        preview_train_coach_plan,
        reset_train_coach_state,
        run_train_coach_workflow,
    )

    gr.HTML(_render_train_coach_steps_html())
    gr.Markdown(build_train_coach_contract_markdown())

    state_payload = gr.Textbox(value="", visible=False, label="Guided State")

    with gr.Row(elem_classes="studio-panel"):
        with gr.Column(scale=1):
            entry_intent = gr.Radio(
                label="Start Here",
                choices=[
                    (entry["title"], entry["id"])
                    for entry in train_coach_entry_intents()
                ],
                value="files",
            )
            uploaded_files = gr.File(
                label="Files",
                file_count="multiple",
                type="filepath",
            )
            local_collection_path = gr.Textbox(
                label="Folder Or Collection Path",
                placeholder="Optional local folder path for image, audio, or video collections",
            )
            remote_src = gr.Textbox(
                label="Remote Dataset",
                placeholder="owner/dataset or https://...",
            )
            revision = gr.Textbox(
                label="Revision",
                placeholder="main",
            )
            resume_artifact = gr.File(
                label="Previous Model Artifact",
                type="filepath",
            )
            save_as = gr.Textbox(
                label="Save Artifact As",
                value="guided-model.joblib",
            )
            inspect_button = gr.Button(
                "Inspect My Inputs",
                elem_classes="btn",
            )
            preview_button = gr.Button(
                "Review Guided Plan",
                interactive=False,
            )
            train_button = gr.Button(
                "Train With Guided Defaults",
                elem_classes="btn",
                interactive=False,
            )
        with gr.Column(scale=1):
            intake_summary = gr.Markdown(
                "## Guided Intake\n- Status: waiting for inspection."
            )
            inspection_markdown = gr.Markdown(
                "## Data Inspection\n- Status: add files or a dataset, then run inspection."
            )
            validation_markdown = gr.Markdown(
                "## Guided Validation\n- Status: inspection pending."
            )
    with gr.Row(elem_classes="studio-panel"):
        with gr.Column(scale=1):
            training_plan = gr.Markdown(
                "## Training Plan\n- Guided inspection will unlock the plan preview."
            )
        with gr.Column(scale=1):
            training_status = gr.Markdown("## Training\n- Status: idle")
            train_output = gr.File(label="Training Output")
            use_result_markdown = gr.Markdown(
                "## Use Result\n- Status: train a model to unlock the saved session, manifest, and next actions."
            )
    with gr.Row(elem_classes="studio-panel"):
        with gr.Column(scale=1):
            resolving_question_markdown = gr.Markdown(
                "## Quick Decision\n- Status: no clarification is needed."
            )
        with gr.Column(scale=1):
            resolving_choice = gr.Radio(
                label="Quick Decision",
                choices=[],
                value=None,
                visible=False,
            )

    inspect_inputs = [
        entry_intent,
        uploaded_files,
        remote_src,
        revision,
        resume_artifact,
        save_as,
        local_collection_path,
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
    for component in (
        entry_intent,
        uploaded_files,
        local_collection_path,
        remote_src,
        revision,
        resume_artifact,
        save_as,
    ):
        component.change(
            fn=reset_train_coach_state,
            outputs=reset_outputs,
            show_progress="hidden",
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
    return {
        "state_payload": state_payload,
        "intake_summary": intake_summary,
        "inspection_markdown": inspection_markdown,
        "validation_markdown": validation_markdown,
        "training_plan": training_plan,
        "training_status": training_status,
        "train_output": train_output,
        "use_result_markdown": use_result_markdown,
        "resolving_question_markdown": resolving_question_markdown,
        "resolving_choice": resolving_choice,
    }


__all__ = [
    "build_train_coach_contract_markdown",
    "build_train_guided_mode",
    "train_coach_css",
    "train_coach_entry_intents",
    "train_coach_step_names",
]
