from __future__ import annotations

TRAIN_TAB_NAMES = ("Studio", "Train", "Run", "Text Lab", "Ops")
RUN_TAB_NAMES = ("Predict", "Infer", "Answer")
TEXT_TAB_NAMES = ("Features", "Reconstruct", "Summaries", "Prompt")
OPS_TAB_NAMES = ("Bootstrap", "K-Means", "Lookup")
TRAIN_STUDIO_SECTIONS = (
    {
        "eyebrow": "Training",
        "title": "Training Orchestration",
        "description": "Preview the full training route before execution, train from local files or remote datasets, and control splits, labels, routing, and resume flows.",
        "items": (
            "Training plan preview",
            "Remote and local dataset training",
            "Resume existing model artifacts",
            "Batch, validation, and test split controls",
        ),
    },
    {
        "eyebrow": "Execution",
        "title": "Model Execution",
        "description": "Run saved artifacts with direct prediction, trigger task-based inference against registered models, and drive answer generation from the ML facade.",
        "items": (
            "Artifact prediction",
            "Task-based inference",
            "Answer runtime invocation",
            "Inline or file-backed results",
        ),
    },
    {
        "eyebrow": "Text",
        "title": "Text And Prompt Lab",
        "description": "Extract and reconstruct text features, run one-pass, map-reduce, and iterative summaries, and prepare prompts for image-oriented pipelines.",
        "items": (
            "Text feature extraction",
            "Feature-to-text reconstruction",
            "Summarize and map-reduce summary",
            "Prompt preprocessing and realism optimization",
        ),
    },
    {
        "eyebrow": "Diagnostics",
        "title": "Diagnostics And Routing",
        "description": "Inspect live ML health, validate runtime readiness, estimate K ranges for clustering, and surface language and checkpoint metadata quickly.",
        "items": (
            "ML health snapshot",
            "Runtime validation",
            "K-means advisor",
            "Language code and RVC checkpoint lookup",
        ),
    },
    {
        "eyebrow": "Bootstrap",
        "title": "Runtime Bootstrap",
        "description": "Initialize file-based model artifacts and load runtime models from the existing definers task catalog without leaving the studio.",
        "items": (
            "init_model_file surface",
            "init_pretrained_model surface",
            "Task catalog access",
            "Turbo toggle control",
        ),
    },
)


def train_studio_tab_names():
    return TRAIN_TAB_NAMES


def train_studio_sections():
    return TRAIN_STUDIO_SECTIONS


def build_capability_markdown():
    lines = ["## ML Studio Surface"]
    for section in TRAIN_STUDIO_SECTIONS:
        lines.append(f"### {section['title']}")
        lines.append(section["description"])
        for item in section["items"]:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _render_section_cards_html():
    cards = []
    for section in TRAIN_STUDIO_SECTIONS:
        items = "".join(f"<li>{item}</li>" for item in section["items"])
        cards.append(
            '<article class="capability-card">'
            f'<div class="capability-chip">{section["eyebrow"]}</div>'
            f"<h3>{section['title']}</h3>"
            f"<p>{section['description']}</p>"
            f"<ul>{items}</ul>"
            "</article>"
        )
    return '<section class="capability-grid">' + "".join(cards) + "</section>"


def _hero_html():
    capability_count = sum(
        len(section["items"]) for section in TRAIN_STUDIO_SECTIONS
    )
    return f"""
<section class="studio-hero">
  <div class="studio-hero__copy">
    <span class="studio-hero__label">Definers ML Studio</span>
    <h1>Train, run, inspect, and bootstrap models from one surface.</h1>
    <p>
      The training launcher is now a full ML cockpit for data-driven training,
      task inference, text tooling, runtime diagnostics, and bootstrap flows.
    </p>
  </div>
  <div class="studio-hero__signal">
    <div>
      <span>Capability Clusters</span>
      <strong>{len(TRAIN_STUDIO_SECTIONS)}</strong>
    </div>
    <div>
      <span>Interactive Tools</span>
      <strong>{capability_count}</strong>
    </div>
    <div>
      <span>Execution Modes</span>
      <strong>Train / Predict / Infer</strong>
    </div>
  </div>
</section>
"""


def train_theme():
    import gradio as gr

    return gr.themes.Soft(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        font=(
            gr.themes.GoogleFont("Space Grotesk"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono=(
            gr.themes.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ).set(
        body_background_fill="#f4efe6",
        block_background_fill="#fffaf1",
        block_border_width="1px",
        block_border_color="#d6c7b4",
        block_label_background_fill="#fffaf1",
        button_primary_background_fill="linear-gradient(135deg, #0f766e, #ea580c)",
        button_primary_text_color="#fffaf1",
        button_secondary_background_fill="#efe3d1",
        button_secondary_text_color="#1c1917",
        checkbox_label_background_fill="#fffaf1",
        input_background_fill="#fffdf8",
        slider_color="#0f766e",
    )


def train_css() -> str:
    return """
.studio-hero {
    display: grid;
    gap: 20px;
    grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr);
    align-items: stretch;
    margin: 18px 0 28px 0;
    padding: 28px;
    border-radius: 28px;
    background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 42%),
        radial-gradient(circle at bottom right, rgba(234, 88, 12, 0.2), transparent 38%),
        linear-gradient(135deg, #fff7ed 0%, #fffbf5 45%, #ecfeff 100%);
    border: 1px solid rgba(148, 163, 184, 0.35);
    box-shadow: 0 24px 80px rgba(28, 25, 23, 0.1);
}

.studio-hero__copy {
    text-align: left !important;
}

.studio-hero__label {
    display: inline-flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(15, 118, 110, 0.12);
    color: #115e59;
    font-size: 12px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.studio-hero h1 {
    margin: 14px 0 12px 0;
    color: #111827 !important;
    font-size: clamp(2rem, 3vw, 3.35rem);
    line-height: 1.02;
    text-align: left !important;
}

.studio-hero p {
    margin: 0;
    max-width: 58ch;
    color: #334155;
    font-size: 1rem;
    line-height: 1.6;
    text-align: left !important;
}

.studio-hero__signal {
    display: grid;
    gap: 12px;
    align-content: center;
}

.studio-hero__signal > div {
    padding: 16px 18px;
    border-radius: 20px;
    background: rgba(255, 250, 241, 0.92);
    border: 1px solid rgba(214, 199, 180, 0.85);
    text-align: left !important;
}

.studio-hero__signal span {
    display: block;
    color: #78716c;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}

.studio-hero__signal strong {
    display: block;
    margin-top: 6px;
    color: #111827;
    font-size: 1.25rem;
}

.capability-grid {
    display: grid;
    gap: 18px;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    margin: 6px 0 22px 0;
}

.capability-card {
    position: relative;
    overflow: hidden;
    padding: 20px;
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(255, 253, 248, 0.98), rgba(252, 245, 235, 0.94));
    border: 1px solid rgba(214, 199, 180, 0.95);
    box-shadow: 0 18px 42px rgba(28, 25, 23, 0.06);
}

.capability-card h3,
.capability-card p,
.capability-card li {
    text-align: left !important;
}

.capability-card h3 {
    margin: 14px 0 10px 0;
    color: #111827 !important;
}

.capability-card p {
    margin: 0 0 12px 0;
    color: #475569;
    line-height: 1.55;
}

.capability-card ul {
    margin: 0;
    padding-left: 18px;
}

.capability-chip {
    display: inline-flex;
    padding: 6px 10px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(234, 88, 12, 0.14));
    color: #115e59;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.studio-panel h2,
.studio-panel h3,
.studio-panel p,
.studio-panel li,
.studio-panel label,
.studio-panel textarea,
.studio-panel input {
    text-align: left !important;
}

.studio-panel .block {
    border-radius: 22px !important;
    box-shadow: 0 14px 32px rgba(28, 25, 23, 0.05) !important;
}

.studio-panel textarea,
.studio-panel input,
.studio-panel .wrap {
    font-family: "IBM Plex Mono", ui-monospace, monospace !important;
}

.studio-panel .tab-nav {
    gap: 8px;
}

.studio-panel button {
    min-height: 46px;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}

@media (max-width: 900px) {
    .studio-hero {
        grid-template-columns: 1fr;
        padding: 20px;
    }
}
"""


def build_train_app():
    import gradio as gr

    from definers.constants import MAX_INPUT_LENGTH, tasks
    from definers.presentation.apps.train_handlers import (
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
    )

    task_choices = sorted(tasks)
    dataset_choices = ["parquet", "json", "csv", "arrow", "webdataset", "txt"]
    model_type_choices = [
        "auto",
        "joblib",
        "onnx",
        "pkl",
        "pt",
        "pth",
        "safetensors",
    ]

    with gr.Blocks(title="Definers ML Studio") as app:
        gr.HTML(_hero_html())
        with gr.Tabs(elem_classes="studio-panel"):
            with gr.TabItem(TRAIN_TAB_NAMES[0]):
                gr.HTML(_render_section_cards_html())
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(build_capability_markdown())
                    with gr.Column(scale=1):
                        refresh_health_button = gr.Button(
                            "Refresh ML Health",
                            elem_classes="btn",
                        )
                        validate_health_button = gr.Button("Validate Runtime")
                        health_markdown = gr.Markdown(
                            "## ML Health\n- Status: click refresh to inspect the live runtime."
                        )
                        validation_status = gr.Markdown(
                            "## Validation\n- Status: idle"
                        )
                refresh_health_button.click(
                    fn=handle_ml_health_report,
                    outputs=[health_markdown, validation_status],
                )
                validate_health_button.click(
                    fn=handle_validate_ml_health,
                    outputs=[validation_status],
                )
            with gr.TabItem(TRAIN_TAB_NAMES[1]):
                with gr.Row():
                    with gr.Column(scale=1):
                        resume_model = gr.File(label="Resume Model Artifact")
                        save_as = gr.Textbox(
                            label="Save Artifact As",
                            placeholder="studio-model.joblib",
                        )
                        remote = gr.Textbox(
                            placeholder="owner/dataset or remote URL",
                            label="Remote Dataset",
                        )
                        revision = gr.Textbox(
                            placeholder="main",
                            label="Revision",
                        )
                        url_type = gr.Dropdown(
                            label="Remote Dataset Type",
                            value="parquet",
                            choices=dataset_choices,
                        )
                        local_features = gr.File(
                            label="Local Features",
                            file_count="multiple",
                            allow_reordering=True,
                        )
                        local_labels = gr.File(
                            label="Local Labels",
                            file_count="multiple",
                            allow_reordering=True,
                        )
                    with gr.Column(scale=1):
                        label_columns = gr.Textbox(
                            placeholder="label;target",
                            label="Label Columns",
                        )
                        drop_list = gr.Textbox(
                            placeholder="unused_column;debug_column",
                            label="Drop Columns",
                        )
                        selected_rows = gr.Textbox(
                            placeholder="1-100 125 130-180",
                            label="Selected Rows",
                            max_length=MAX_INPUT_LENGTH,
                        )
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            value=32,
                            label="Batch Size",
                        )
                        validation_split = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            step=0.05,
                            value=0.0,
                            label="Validation Split",
                        )
                        test_split = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            step=0.05,
                            value=0.0,
                            label="Test Split",
                        )
                        with gr.Accordion("Advanced Routing", open=False):
                            order_by = gr.Textbox(
                                placeholder="shuffle",
                                label="Order By",
                            )
                            stratify = gr.Textbox(
                                placeholder="label",
                                label="Stratify",
                            )
                        with gr.Row():
                            preview_plan_button = gr.Button(
                                "Preview Training Plan"
                            )
                            train_button = gr.Button(
                                "Train Model",
                                elem_classes="btn",
                            )
                    with gr.Column(scale=1):
                        training_status = gr.Markdown(
                            "## Training\n- Status: idle"
                        )
                        training_plan = gr.Markdown(
                            "## Training Plan\n- Build a plan to inspect the execution route."
                        )
                        train_output = gr.File(label="Training Output")
                preview_plan_button.click(
                    fn=build_training_plan_markdown,
                    inputs=[
                        local_features,
                        local_labels,
                        resume_model,
                        remote,
                        label_columns,
                        revision,
                        url_type,
                        drop_list,
                        selected_rows,
                        batch_size,
                        validation_split,
                        test_split,
                        order_by,
                        stratify,
                    ],
                    outputs=[training_plan],
                )
                train_button.click(
                    fn=handle_training,
                    inputs=[
                        local_features,
                        local_labels,
                        resume_model,
                        remote,
                        label_columns,
                        revision,
                        url_type,
                        drop_list,
                        selected_rows,
                        save_as,
                        batch_size,
                        validation_split,
                        test_split,
                        order_by,
                        stratify,
                    ],
                    outputs=[train_output, training_plan, training_status],
                )
            with gr.TabItem(TRAIN_TAB_NAMES[2]):
                with gr.Tabs():
                    with gr.TabItem(RUN_TAB_NAMES[0]):
                        with gr.Row():
                            with gr.Column(scale=1):
                                model_predict = gr.File(
                                    label="Saved Model Artifact"
                                )
                                prediction_data = gr.File(
                                    label="Prediction Input File"
                                )
                                prediction_payload = gr.Code(
                                    label="Or paste in-memory payload",
                                    language="json",
                                    value="[[0.2, 0.4], [0.8, 0.6]]",
                                )
                                predict_button = gr.Button(
                                    "Run Prediction",
                                    elem_classes="btn",
                                )
                            with gr.Column(scale=1):
                                predict_status = gr.Markdown(
                                    "## Prediction\n- Status: idle"
                                )
                                predict_output = gr.File(
                                    label="Prediction Artifact"
                                )
                                predict_preview = gr.Code(
                                    label="Prediction Preview",
                                    language="json",
                                )
                        predict_button.click(
                            fn=handle_prediction,
                            inputs=[
                                model_predict,
                                prediction_data,
                                prediction_payload,
                            ],
                            outputs=[
                                predict_output,
                                predict_status,
                                predict_preview,
                            ],
                        )
                    with gr.TabItem(RUN_TAB_NAMES[1]):
                        with gr.Row():
                            with gr.Column(scale=1):
                                inference_task = gr.Dropdown(
                                    label="Model Task",
                                    choices=task_choices,
                                    value="answer",
                                    allow_custom_value=True,
                                )
                                inference_model_type = gr.Dropdown(
                                    label="Model Type",
                                    choices=model_type_choices,
                                    value="auto",
                                )
                                inference_data = gr.File(
                                    label="Inference Input File"
                                )
                                infer_button = gr.Button(
                                    "Run Task Inference",
                                    elem_classes="btn",
                                )
                            with gr.Column(scale=1):
                                infer_status = gr.Markdown(
                                    "## Inference\n- Status: idle"
                                )
                                infer_output = gr.File(
                                    label="Inference Artifact"
                                )
                                infer_preview = gr.Code(
                                    label="Inference Preview",
                                    language="json",
                                )
                        infer_button.click(
                            fn=handle_inference,
                            inputs=[
                                inference_task,
                                inference_data,
                                inference_model_type,
                            ],
                            outputs=[infer_output, infer_status, infer_preview],
                        )
                    with gr.TabItem(RUN_TAB_NAMES[2]):
                        with gr.Row():
                            with gr.Column(scale=1):
                                answer_prompt = gr.Textbox(
                                    label="Prompt",
                                    lines=5,
                                    max_length=MAX_INPUT_LENGTH,
                                )
                                answer_history = gr.Code(
                                    label="Optional History JSON",
                                    language="json",
                                    value="[]",
                                )
                                answer_attachment = gr.File(
                                    label="Optional Image Or Audio Attachment"
                                )
                                answer_button = gr.Button(
                                    "Run Answer Runtime",
                                    elem_classes="btn",
                                )
                            with gr.Column(scale=1):
                                answer_output = gr.Textbox(
                                    label="Answer Output",
                                    lines=10,
                                    buttons=["copy"],
                                )
                        answer_button.click(
                            fn=handle_answer,
                            inputs=[
                                answer_prompt,
                                answer_history,
                                answer_attachment,
                            ],
                            outputs=[answer_output],
                        )
            with gr.TabItem(TRAIN_TAB_NAMES[3]):
                with gr.Tabs():
                    with gr.TabItem(TEXT_TAB_NAMES[0]):
                        with gr.Row():
                            with gr.Column(scale=1):
                                feature_text = gr.Textbox(
                                    label="Source Text",
                                    lines=8,
                                    max_length=MAX_INPUT_LENGTH,
                                )
                                extract_features_button = gr.Button(
                                    "Extract Text Features",
                                    elem_classes="btn",
                                )
                            with gr.Column(scale=1):
                                feature_summary = gr.Markdown(
                                    "## Text Features\n- Status: idle"
                                )
                                feature_output = gr.Code(
                                    label="Feature Matrix",
                                    language="json",
                                )
                        extract_features_button.click(
                            fn=handle_text_feature_extraction,
                            inputs=[feature_text],
                            outputs=[feature_output, feature_summary],
                        )
                    with gr.TabItem(TEXT_TAB_NAMES[1]):
                        with gr.Row():
                            with gr.Column(scale=1):
                                reconstructed_features = gr.Code(
                                    label="Feature Payload",
                                    language="json",
                                    value="[[1, 0, 0, 1]]",
                                )
                                vocabulary = gr.Textbox(
                                    label="Vocabulary",
                                    lines=4,
                                    placeholder="word_a, word_b, word_c",
                                )
                                reconstruct_button = gr.Button(
                                    "Reconstruct Text",
                                    elem_classes="btn",
                                )
                            with gr.Column(scale=1):
                                reconstructed_text = gr.Textbox(
                                    label="Reconstructed Text",
                                    lines=8,
                                    buttons=["copy"],
                                )
                        reconstruct_button.click(
                            fn=handle_features_to_text,
                            inputs=[reconstructed_features, vocabulary],
                            outputs=[reconstructed_text],
                        )
                    with gr.TabItem(TEXT_TAB_NAMES[2]):
                        with gr.Row():
                            with gr.Column(scale=1):
                                summary_text = gr.Textbox(
                                    label="Summary Source",
                                    lines=10,
                                )
                                max_words = gr.Slider(
                                    minimum=5,
                                    maximum=120,
                                    step=1,
                                    value=24,
                                    label="Max Words",
                                )
                                min_loops = gr.Slider(
                                    minimum=1,
                                    maximum=6,
                                    step=1,
                                    value=1,
                                    label="Minimum Loops",
                                )
                                with gr.Row():
                                    quick_summary_button = gr.Button(
                                        "One-Pass Summarize"
                                    )
                                    map_reduce_button = gr.Button(
                                        "Map-Reduce Summary"
                                    )
                                    iterative_summary_button = gr.Button(
                                        "Iterative Summary",
                                        elem_classes="btn",
                                    )
                            with gr.Column(scale=1):
                                quick_summary_output = gr.Textbox(
                                    label="One-Pass Output",
                                    lines=6,
                                )
                                map_reduce_output = gr.Textbox(
                                    label="Map-Reduce Output",
                                    lines=6,
                                )
                                iterative_summary_output = gr.Textbox(
                                    label="Iterative Output",
                                    lines=6,
                                )
                        quick_summary_button.click(
                            fn=handle_quick_summary,
                            inputs=[summary_text],
                            outputs=[quick_summary_output],
                        )
                        map_reduce_button.click(
                            fn=handle_map_reduce_summary,
                            inputs=[summary_text, max_words],
                            outputs=[map_reduce_output],
                        )
                        iterative_summary_button.click(
                            fn=handle_iterative_summary,
                            inputs=[summary_text, max_words, min_loops],
                            outputs=[iterative_summary_output],
                        )
                    with gr.TabItem(TEXT_TAB_NAMES[3]):
                        with gr.Row():
                            with gr.Column(scale=1):
                                prompt_input = gr.Textbox(
                                    label="Prompt",
                                    lines=8,
                                    max_length=MAX_INPUT_LENGTH,
                                )
                                prompt_button = gr.Button(
                                    "Optimize Prompt",
                                    elem_classes="btn",
                                )
                            with gr.Column(scale=1):
                                preprocessed_prompt = gr.Textbox(
                                    label="Preprocessed Prompt",
                                    lines=5,
                                )
                                optimized_prompt = gr.Textbox(
                                    label="Optimized Prompt",
                                    lines=8,
                                )
                        prompt_button.click(
                            fn=handle_prompt_optimization,
                            inputs=[prompt_input],
                            outputs=[preprocessed_prompt, optimized_prompt],
                        )
            with gr.TabItem(TRAIN_TAB_NAMES[4]):
                with gr.Tabs():
                    with gr.TabItem(OPS_TAB_NAMES[0]):
                        with gr.Row():
                            with gr.Column(scale=1):
                                bootstrap_task = gr.Dropdown(
                                    label="Task",
                                    choices=task_choices,
                                    value="answer",
                                    allow_custom_value=True,
                                )
                                turbo = gr.Checkbox(
                                    label="Turbo",
                                    value=True,
                                )
                                bootstrap_model_type = gr.Dropdown(
                                    label="Model Type",
                                    choices=model_type_choices,
                                    value="auto",
                                )
                                with gr.Row():
                                    init_model_files_button = gr.Button(
                                        "Init Model Files"
                                    )
                                    load_runtime_button = gr.Button(
                                        "Load Runtime Model",
                                        elem_classes="btn",
                                    )
                            with gr.Column(scale=1):
                                init_model_files_status = gr.Markdown(
                                    "## Model File Init\n- Status: idle"
                                )
                                load_runtime_status = gr.Markdown(
                                    "## Runtime Model Init\n- Status: idle"
                                )
                        init_model_files_button.click(
                            fn=handle_init_model_files,
                            inputs=[
                                bootstrap_task,
                                turbo,
                                bootstrap_model_type,
                            ],
                            outputs=[init_model_files_status],
                        )
                        load_runtime_button.click(
                            fn=handle_load_runtime_model,
                            inputs=[bootstrap_task, turbo],
                            outputs=[load_runtime_status],
                        )
                    with gr.TabItem(OPS_TAB_NAMES[1]):
                        with gr.Row():
                            with gr.Column(scale=1):
                                kmeans_matrix = gr.Code(
                                    label="Feature Matrix",
                                    language="json",
                                    value="[[0.1, 0.3], [0.2, 0.4], [0.9, 1.1], [1.0, 1.2]]",
                                )
                                k_min = gr.Slider(
                                    minimum=2,
                                    maximum=16,
                                    step=1,
                                    value=2,
                                    label="k Min",
                                )
                                k_max = gr.Slider(
                                    minimum=3,
                                    maximum=24,
                                    step=1,
                                    value=8,
                                    label="k Max",
                                )
                                random_state = gr.Number(
                                    label="Random State",
                                    value=42,
                                )
                                kmeans_button = gr.Button(
                                    "Suggest Cluster Counts",
                                    elem_classes="btn",
                                )
                            with gr.Column(scale=1):
                                kmeans_summary = gr.Markdown(
                                    "## K-Means Advisor\n- Status: idle"
                                )
                                kmeans_output = gr.Code(
                                    label="K-Means Metrics",
                                    language="json",
                                )
                        kmeans_button.click(
                            fn=handle_kmeans_suggestions,
                            inputs=[kmeans_matrix, k_min, k_max, random_state],
                            outputs=[kmeans_summary, kmeans_output],
                        )
                    with gr.TabItem(OPS_TAB_NAMES[2]):
                        with gr.Row():
                            with gr.Column(scale=1):
                                checkpoint_folder = gr.Textbox(
                                    label="RVC Checkpoint Folder",
                                    placeholder="./logs/model_name",
                                )
                                checkpoint_model_name = gr.Textbox(
                                    label="RVC Model Name",
                                    placeholder="voice_model",
                                )
                                checkpoint_button = gr.Button(
                                    "Find Latest Checkpoint"
                                )
                                checkpoint_status = gr.Markdown(
                                    "## RVC Checkpoint\n- Status: idle"
                                )
                            with gr.Column(scale=1):
                                language_code = gr.Textbox(
                                    label="Language Code",
                                    placeholder="en",
                                )
                                language_button = gr.Button(
                                    "Resolve Language",
                                    elem_classes="btn",
                                )
                                language_status = gr.Markdown(
                                    "## Language Lookup\n- Status: idle"
                                )
                        checkpoint_button.click(
                            fn=handle_rvc_checkpoint_lookup,
                            inputs=[checkpoint_folder, checkpoint_model_name],
                            outputs=[checkpoint_status],
                        )
                        language_button.click(
                            fn=handle_language_lookup,
                            inputs=[language_code],
                            outputs=[language_status],
                        )
    return app


__all__ = [
    "build_capability_markdown",
    "build_train_app",
    "train_css",
    "train_studio_sections",
    "train_studio_tab_names",
    "train_theme",
]
