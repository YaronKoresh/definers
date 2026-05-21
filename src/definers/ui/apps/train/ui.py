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


def build_train_app(
    visible_sections=None,
    *,
    app_title="Definers ML Studio",
    hero_label="Definers ML Studio",
    hero_heading="Train, run, inspect, and bootstrap models from one surface.",
    hero_description=(
        "The training launcher is a full ML cockpit for data-driven training, "
        "task inference, text tooling, runtime diagnostics, and bootstrap flows."
    ),
):
    from PySide6.QtWidgets import QPushButton

    from definers.constants import MAX_INPUT_LENGTH, tasks
    from definers.ui.gui_shared import (
        bind_progress_click,
        init_output_folder_controls,
        init_progress_tracker,
    )
    from definers.ui.qt_widgets import (
        Accordion,
        CardFrame,
        FilePicker,
        HeroSection,
        LabeledCheckbox,
        LabeledCombo,
        LabeledFloatSlider,
        LabeledLineEdit,
        LabeledSlider,
        LabeledSpin,
        LabeledTextbox,
        MarkdownLabel,
        Row,
        Section,
        build_main_window,
        build_tab_widget,
    )

    from .coach_ui import build_train_guided_mode
    from .handlers import (
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
    selected_sections = set(
        visible_sections or ("studio", "train", "run", "text", "ops")
    )

    body = Section()
    _capability_count = sum(len(s["items"]) for s in TRAIN_STUDIO_SECTIONS)
    hero = HeroSection()
    hero.addWidget(MarkdownLabel(f"**{hero_label.upper()}**"))
    hero.addWidget(MarkdownLabel(f"# {hero_heading}"))
    hero.addWidget(MarkdownLabel(hero_description))
    _stats_row = Row()
    for _stat_name, _stat_val in (
        ("Capability Clusters", str(len(TRAIN_STUDIO_SECTIONS))),
        ("Interactive Tools", str(_capability_count)),
        ("Execution Modes", "Train / Predict / Infer"),
    ):
        _stat_cell = CardFrame()
        _stat_cell.addWidget(MarkdownLabel(f"**{_stat_name}**\n\n{_stat_val}"))
        _stats_row.addWidget(_stat_cell)
    hero.addWidget(_stats_row)
    body.addWidget(hero)
    progress_status = init_progress_tracker(
        "ML studio ready",
        "Choose a task and execute it.",
    )
    body.addWidget(progress_status)
    _, output_toolbar = init_output_folder_controls(section="train")
    body.addWidget(output_toolbar)

    def bind_action(
        button,
        handler,
        *,
        inputs=None,
        outputs=None,
        action_label=None,
        steps=None,
        running_detail=None,
        success_detail=None,
    ):
        return bind_progress_click(
            button,
            handler,
            progress_output=progress_status,
            inputs=inputs,
            outputs=outputs,
            action_label=action_label,
            steps=steps,
            running_detail=running_detail,
            success_detail=success_detail,
        )

    tabs = build_tab_widget()
    body.addWidget(tabs)

    if "studio" in selected_sections:
        studio_tab = Section()
        for _section_data in TRAIN_STUDIO_SECTIONS:
            _card = CardFrame()
            _items_md = "\n".join(
                f"- {item}" for item in _section_data["items"]
            )
            _card.addWidget(
                MarkdownLabel(
                    f"**{_section_data['eyebrow'].upper()}**\n\n"
                    f"### {_section_data['title']}\n\n"
                    f"{_section_data['description']}\n\n"
                    f"{_items_md}"
                )
            )
            studio_tab.addWidget(_card)
        row = Row()
        mid = Section()
        refresh_health_button = QPushButton("Refresh ML Health")
        validate_health_button = QPushButton("Validate Runtime")
        health_markdown = MarkdownLabel(
            "## ML Health\n- Status: click refresh to inspect the live runtime."
        )
        validation_status = MarkdownLabel("## Validation\n- Status: idle")
        mid.addWidget(refresh_health_button)
        mid.addWidget(validate_health_button)
        mid.addWidget(health_markdown)
        mid.addWidget(validation_status)
        row.addWidget(mid)
        studio_tab.addWidget(row)
        tabs.addTab(studio_tab, TRAIN_TAB_NAMES[0])

        bind_action(
            refresh_health_button,
            handle_ml_health_report,
            outputs=[health_markdown, validation_status],
            action_label="Refresh ML Health",
            steps=(
                "Validate runtime",
                "Collect health snapshot",
                "Publish report",
            ),
            running_detail="Collecting the live ML health snapshot.",
            success_detail="ML health snapshot is ready.",
        )
        bind_action(
            validate_health_button,
            handle_validate_ml_health,
            outputs=[validation_status],
            action_label="Validate Runtime",
            steps=(
                "Validate request",
                "Inspect dependencies",
                "Publish validation",
            ),
            running_detail="Validating the runtime dependencies.",
            success_detail="Runtime validation is complete.",
        )

    if "train" in selected_sections:
        train_tab = Section()
        train_tab.addWidget(
            MarkdownLabel(
                "Use Guided Mode for the beginner-first route or Advanced Mode"
                " for direct control over columns, routing, and splits."
            )
        )
        train_subtabs = build_tab_widget()
        train_tab.addWidget(train_subtabs)

        guided_panel = Section()
        build_train_guided_mode(
            container=guided_panel,
            bind_action=bind_action,
        )
        train_subtabs.addTab(guided_panel, "Guided Mode")

        advanced_panel = Section()
        adv_row = Row()
        left = Section()
        resume_model = FilePicker("Resume Model Artifact")
        save_as = LabeledLineEdit(
            "Save Artifact As",
            placeholder="studio-model.joblib",
        )
        remote = LabeledLineEdit(
            "Remote Dataset",
            placeholder="owner/dataset or remote URL",
        )
        revision = LabeledLineEdit("Revision", placeholder="main")
        url_type = LabeledCombo(
            "Remote Dataset Type",
            choices=dataset_choices,
            value="parquet",
        )
        local_features = FilePicker("Local Features")
        local_labels = FilePicker("Local Labels")
        for widget in (
            resume_model,
            save_as,
            remote,
            revision,
            url_type,
            local_features,
            local_labels,
        ):
            left.addWidget(widget)

        mid = Section()
        label_columns = LabeledLineEdit(
            "Label Columns", placeholder="label;target"
        )
        drop_list = LabeledLineEdit(
            "Drop Columns", placeholder="unused_column;debug_column"
        )
        selected_rows = LabeledLineEdit(
            "Selected Rows", placeholder="1-100 125 130-180"
        )
        batch_size = LabeledSlider(
            "Batch Size",
            minimum=1,
            maximum=512,
            step=1,
            value=32,
        )
        validation_split = LabeledFloatSlider(
            "Validation Split",
            minimum=0.0,
            maximum=0.5,
            step=0.05,
            value=0.0,
        )
        test_split = LabeledFloatSlider(
            "Test Split",
            minimum=0.0,
            maximum=0.5,
            step=0.05,
            value=0.0,
        )
        routing = Accordion("Advanced Routing", open=False)
        order_by = LabeledLineEdit("Order By", placeholder="shuffle")
        stratify = LabeledLineEdit("Stratify", placeholder="label")
        routing.addWidget(order_by)
        routing.addWidget(stratify)
        preview_plan_button = QPushButton("Preview Training Plan")
        train_button = QPushButton("Train Model")
        train_button.setProperty("definersPrimary", True)
        for widget in (
            label_columns,
            drop_list,
            selected_rows,
            batch_size,
            validation_split,
            test_split,
            routing,
        ):
            mid.addWidget(widget)
        btn_row = Row()
        btn_row.addWidget(preview_plan_button)
        btn_row.addWidget(train_button)
        mid.addWidget(btn_row)

        right = Section()
        training_status = MarkdownLabel("## Training\n- Status: idle")
        training_plan = MarkdownLabel(
            "## Training Plan\n- Build a plan to inspect the execution route."
        )
        train_output = FilePicker("Training Output")
        right.addWidget(training_status)
        right.addWidget(training_plan)
        right.addWidget(train_output)

        adv_row.addWidget(left)
        adv_row.addWidget(mid)
        adv_row.addWidget(right)
        advanced_panel.addWidget(adv_row)
        train_subtabs.addTab(advanced_panel, "Advanced Mode")
        tabs.addTab(train_tab, TRAIN_TAB_NAMES[1])

        bind_action(
            preview_plan_button,
            build_training_plan_markdown,
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
            action_label="Preview Training Plan",
            steps=(
                "Validate data sources",
                "Build training plan",
                "Publish plan",
            ),
            running_detail="Building the training plan preview.",
            success_detail="Training plan preview is ready.",
        )
        bind_action(
            train_button,
            handle_training,
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
            action_label="Train Model",
            steps=(
                "Validate training inputs",
                "Run training job",
                "Publish artifacts",
            ),
            running_detail="Running the training workflow.",
            success_detail="Training run is complete.",
        )

    if "run" in selected_sections:
        run_tab = Section()
        run_subtabs = build_tab_widget()
        run_tab.addWidget(run_subtabs)

        predict_panel = Section()
        pr = Row()
        left = Section()
        model_predict = FilePicker("Saved Model Artifact")
        predict_model_type = LabeledCombo(
            "Model Type", choices=model_type_choices, value="auto"
        )
        prediction_data = FilePicker("Prediction Input File")
        prediction_payload = LabeledTextbox(
            "Or paste in-memory payload",
            value="[[0.2, 0.4], [0.8, 0.6]]",
            lines=4,
        )
        predict_button = QPushButton("Run Prediction")
        predict_button.setProperty("definersPrimary", True)
        for widget in (
            model_predict,
            predict_model_type,
            prediction_data,
            prediction_payload,
            predict_button,
        ):
            left.addWidget(widget)
        right = Section()
        predict_status = MarkdownLabel("## Prediction\n- Status: idle")
        predict_output = FilePicker("Prediction Artifact")
        predict_preview = LabeledTextbox("Prediction Preview", lines=8)
        right.addWidget(predict_status)
        right.addWidget(predict_output)
        right.addWidget(predict_preview)
        pr.addWidget(left)
        pr.addWidget(right)
        predict_panel.addWidget(pr)
        run_subtabs.addTab(predict_panel, RUN_TAB_NAMES[0])

        bind_action(
            predict_button,
            handle_prediction,
            inputs=[
                model_predict,
                predict_model_type,
                prediction_data,
                prediction_payload,
            ],
            outputs=[predict_output, predict_status, predict_preview],
            action_label="Run Prediction",
            steps=(
                "Validate model inputs",
                "Run prediction",
                "Publish artifact",
            ),
            running_detail="Running the saved model prediction.",
            success_detail="Prediction result is ready.",
        )

        infer_panel = Section()
        ir = Row()
        left = Section()
        inference_task = LabeledCombo(
            "Model Task",
            choices=task_choices,
            value="answer",
        )
        inference_model_type = LabeledCombo(
            "Model Type", choices=model_type_choices, value="auto"
        )
        inference_data = FilePicker("Inference Input File")
        infer_button = QPushButton("Run Task Inference")
        infer_button.setProperty("definersPrimary", True)
        for widget in (
            inference_task,
            inference_model_type,
            inference_data,
            infer_button,
        ):
            left.addWidget(widget)
        right = Section()
        infer_status = MarkdownLabel("## Inference\n- Status: idle")
        infer_output = FilePicker("Inference Artifact")
        infer_preview = LabeledTextbox("Inference Preview", lines=8)
        right.addWidget(infer_status)
        right.addWidget(infer_output)
        right.addWidget(infer_preview)
        ir.addWidget(left)
        ir.addWidget(right)
        infer_panel.addWidget(ir)
        run_subtabs.addTab(infer_panel, RUN_TAB_NAMES[1])

        bind_action(
            infer_button,
            handle_inference,
            inputs=[inference_task, inference_data, inference_model_type],
            outputs=[infer_output, infer_status, infer_preview],
            action_label="Run Task Inference",
            steps=(
                "Validate inference inputs",
                "Run inference",
                "Publish artifact",
            ),
            running_detail="Running task-based inference.",
            success_detail="Inference result is ready.",
        )

        answer_panel = Section()
        ar = Row()
        left = Section()
        answer_prompt = LabeledTextbox(
            "Prompt", lines=5, max_length=MAX_INPUT_LENGTH
        )
        answer_history = LabeledTextbox(
            "Optional History JSON", value="[]", lines=4
        )
        answer_attachment = FilePicker("Optional Image Or Audio Attachment")
        answer_button = QPushButton("Run Answer Runtime")
        answer_button.setProperty("definersPrimary", True)
        for widget in (
            answer_prompt,
            answer_history,
            answer_attachment,
            answer_button,
        ):
            left.addWidget(widget)
        right = Section()
        answer_output = LabeledTextbox("Answer Output", lines=10)
        right.addWidget(answer_output)
        ar.addWidget(left)
        ar.addWidget(right)
        answer_panel.addWidget(ar)
        run_subtabs.addTab(answer_panel, RUN_TAB_NAMES[2])

        bind_action(
            answer_button,
            handle_answer,
            inputs=[answer_prompt, answer_history, answer_attachment],
            outputs=[answer_output],
            action_label="Run Answer Runtime",
            steps=(
                "Validate prompt payload",
                "Run answer runtime",
                "Publish answer",
            ),
            running_detail="Running the answer runtime.",
            success_detail="Answer output is ready.",
        )
        tabs.addTab(run_tab, TRAIN_TAB_NAMES[2])

    if "text" in selected_sections:
        text_tab = Section()
        text_subtabs = build_tab_widget()
        text_tab.addWidget(text_subtabs)

        feat_panel = Section()
        fr = Row()
        left = Section()
        feature_text = LabeledTextbox(
            "Source Text", lines=8, max_length=MAX_INPUT_LENGTH
        )
        extract_features_button = QPushButton("Extract Text Features")
        extract_features_button.setProperty("definersPrimary", True)
        left.addWidget(feature_text)
        left.addWidget(extract_features_button)
        right = Section()
        feature_summary = MarkdownLabel("## Text Features\n- Status: idle")
        feature_output = LabeledTextbox("Feature Matrix", lines=8)
        right.addWidget(feature_summary)
        right.addWidget(feature_output)
        fr.addWidget(left)
        fr.addWidget(right)
        feat_panel.addWidget(fr)
        text_subtabs.addTab(feat_panel, TEXT_TAB_NAMES[0])

        bind_action(
            extract_features_button,
            handle_text_feature_extraction,
            inputs=[feature_text],
            outputs=[feature_output, feature_summary],
            action_label="Extract Text Features",
            steps=(
                "Validate source text",
                "Extract feature matrix",
                "Publish features",
            ),
            running_detail="Extracting the text feature matrix.",
            success_detail="Text features are ready.",
        )

        reconstruct_panel = Section()
        rr = Row()
        left = Section()
        reconstructed_features = LabeledTextbox(
            "Feature Payload", value="[[1, 0, 0, 1]]", lines=4
        )
        vocabulary = LabeledTextbox(
            "Vocabulary", lines=4, placeholder="word_a, word_b, word_c"
        )
        reconstruct_button = QPushButton("Reconstruct Text")
        reconstruct_button.setProperty("definersPrimary", True)
        for widget in (
            reconstructed_features,
            vocabulary,
            reconstruct_button,
        ):
            left.addWidget(widget)
        right = Section()
        reconstructed_text = LabeledTextbox("Reconstructed Text", lines=8)
        right.addWidget(reconstructed_text)
        rr.addWidget(left)
        rr.addWidget(right)
        reconstruct_panel.addWidget(rr)
        text_subtabs.addTab(reconstruct_panel, TEXT_TAB_NAMES[1])

        bind_action(
            reconstruct_button,
            handle_features_to_text,
            inputs=[reconstructed_features, vocabulary],
            outputs=[reconstructed_text],
            action_label="Reconstruct Text",
            steps=(
                "Validate feature payload",
                "Reconstruct text",
                "Publish text",
            ),
            running_detail="Reconstructing text from features.",
            success_detail="Reconstructed text is ready.",
        )

        summary_panel = Section()
        sr = Row()
        left = Section()
        summary_text = LabeledTextbox("Summary Source", lines=10)
        max_words = LabeledSlider(
            "Max Words", minimum=5, maximum=120, step=1, value=24
        )
        min_loops = LabeledSlider(
            "Minimum Loops", minimum=1, maximum=6, step=1, value=1
        )
        quick_summary_button = QPushButton("One-Pass Summarize")
        map_reduce_button = QPushButton("Map-Reduce Summary")
        iterative_summary_button = QPushButton("Iterative Summary")
        iterative_summary_button.setProperty("definersPrimary", True)
        btn_row = Row()
        btn_row.addWidget(quick_summary_button)
        btn_row.addWidget(map_reduce_button)
        btn_row.addWidget(iterative_summary_button)
        for widget in (summary_text, max_words, min_loops):
            left.addWidget(widget)
        left.addWidget(btn_row)
        right = Section()
        quick_summary_output = LabeledTextbox("One-Pass Output", lines=6)
        map_reduce_output = LabeledTextbox("Map-Reduce Output", lines=6)
        iterative_summary_output = LabeledTextbox("Iterative Output", lines=6)
        right.addWidget(quick_summary_output)
        right.addWidget(map_reduce_output)
        right.addWidget(iterative_summary_output)
        sr.addWidget(left)
        sr.addWidget(right)
        summary_panel.addWidget(sr)
        text_subtabs.addTab(summary_panel, TEXT_TAB_NAMES[2])

        bind_action(
            quick_summary_button,
            handle_quick_summary,
            inputs=[summary_text],
            outputs=[quick_summary_output],
            action_label="One-Pass Summarize",
            steps=(
                "Validate source text",
                "Create summary",
                "Publish summary",
            ),
            running_detail="Creating a one-pass summary.",
            success_detail="One-pass summary is ready.",
        )
        bind_action(
            map_reduce_button,
            handle_map_reduce_summary,
            inputs=[summary_text, max_words],
            outputs=[map_reduce_output],
            action_label="Map-Reduce Summary",
            steps=(
                "Validate source text",
                "Run map-reduce summary",
                "Publish summary",
            ),
            running_detail="Running the map-reduce summary flow.",
            success_detail="Map-reduce summary is ready.",
        )
        bind_action(
            iterative_summary_button,
            handle_iterative_summary,
            inputs=[summary_text, max_words, min_loops],
            outputs=[iterative_summary_output],
            action_label="Iterative Summary",
            steps=(
                "Validate source text",
                "Run iterative summary",
                "Publish summary",
            ),
            running_detail="Running the iterative summary flow.",
            success_detail="Iterative summary is ready.",
        )

        prompt_panel = Section()
        pr = Row()
        left = Section()
        prompt_input = LabeledTextbox(
            "Prompt", lines=8, max_length=MAX_INPUT_LENGTH
        )
        prompt_button = QPushButton("Optimize Prompt")
        prompt_button.setProperty("definersPrimary", True)
        left.addWidget(prompt_input)
        left.addWidget(prompt_button)
        right = Section()
        preprocessed_prompt = LabeledTextbox("Preprocessed Prompt", lines=5)
        optimized_prompt = LabeledTextbox("Optimized Prompt", lines=8)
        right.addWidget(preprocessed_prompt)
        right.addWidget(optimized_prompt)
        pr.addWidget(left)
        pr.addWidget(right)
        prompt_panel.addWidget(pr)
        text_subtabs.addTab(prompt_panel, TEXT_TAB_NAMES[3])

        bind_action(
            prompt_button,
            handle_prompt_optimization,
            inputs=[prompt_input],
            outputs=[preprocessed_prompt, optimized_prompt],
            action_label="Optimize Prompt",
            steps=(
                "Validate prompt",
                "Optimize prompt",
                "Publish result",
            ),
            running_detail="Optimizing the prompt payload.",
            success_detail="Prompt optimization is ready.",
        )
        tabs.addTab(text_tab, TRAIN_TAB_NAMES[3])

    if "ops" in selected_sections:
        ops_tab = Section()
        ops_subtabs = build_tab_widget()
        ops_tab.addWidget(ops_subtabs)

        bootstrap_panel = Section()
        br = Row()
        left = Section()
        bootstrap_task = LabeledCombo(
            "Task", choices=task_choices, value="answer"
        )
        turbo = LabeledCheckbox("Turbo", value=True)
        bootstrap_model_type = LabeledCombo(
            "Model Type", choices=model_type_choices, value="auto"
        )
        init_model_files_button = QPushButton("Init Model Files")
        load_runtime_button = QPushButton("Load Runtime Model")
        load_runtime_button.setProperty("definersPrimary", True)
        for widget in (bootstrap_task, turbo, bootstrap_model_type):
            left.addWidget(widget)
        btn_row = Row()
        btn_row.addWidget(init_model_files_button)
        btn_row.addWidget(load_runtime_button)
        left.addWidget(btn_row)
        right = Section()
        init_model_files_status = MarkdownLabel(
            "## Model File Init\n- Status: idle"
        )
        load_runtime_status = MarkdownLabel(
            "## Runtime Model Init\n- Status: idle"
        )
        right.addWidget(init_model_files_status)
        right.addWidget(load_runtime_status)
        br.addWidget(left)
        br.addWidget(right)
        bootstrap_panel.addWidget(br)
        ops_subtabs.addTab(bootstrap_panel, OPS_TAB_NAMES[0])

        bind_action(
            init_model_files_button,
            handle_init_model_files,
            inputs=[bootstrap_task, turbo, bootstrap_model_type],
            outputs=[init_model_files_status],
            action_label="Init Model Files",
            steps=(
                "Validate bootstrap config",
                "Initialize model files",
                "Publish status",
            ),
            running_detail="Initializing model files for the selected task.",
            success_detail="Model file initialization is complete.",
        )
        bind_action(
            load_runtime_button,
            handle_load_runtime_model,
            inputs=[bootstrap_task, turbo],
            outputs=[load_runtime_status],
            action_label="Load Runtime Model",
            steps=(
                "Validate bootstrap config",
                "Load runtime model",
                "Publish status",
            ),
            running_detail="Loading the runtime model.",
            success_detail="Runtime model is ready.",
        )

        kmeans_panel = Section()
        kr = Row()
        left = Section()
        kmeans_matrix = LabeledTextbox(
            "Feature Matrix",
            value="[[0.1, 0.3], [0.2, 0.4], [0.9, 1.1], [1.0, 1.2]]",
            lines=4,
        )
        k_min = LabeledSlider("k Min", minimum=2, maximum=16, step=1, value=2)
        k_max = LabeledSlider("k Max", minimum=3, maximum=24, step=1, value=8)
        random_state = LabeledSpin("Random State", value=42)
        kmeans_button = QPushButton("Suggest Cluster Counts")
        kmeans_button.setProperty("definersPrimary", True)
        for widget in (
            kmeans_matrix,
            k_min,
            k_max,
            random_state,
            kmeans_button,
        ):
            left.addWidget(widget)
        right = Section()
        kmeans_summary = MarkdownLabel("## K-Means Advisor\n- Status: idle")
        kmeans_output = LabeledTextbox("K-Means Metrics", lines=8)
        right.addWidget(kmeans_summary)
        right.addWidget(kmeans_output)
        kr.addWidget(left)
        kr.addWidget(right)
        kmeans_panel.addWidget(kr)
        ops_subtabs.addTab(kmeans_panel, OPS_TAB_NAMES[1])

        bind_action(
            kmeans_button,
            handle_kmeans_suggestions,
            inputs=[kmeans_matrix, k_min, k_max, random_state],
            outputs=[kmeans_summary, kmeans_output],
            action_label="Suggest Cluster Counts",
            steps=(
                "Validate feature matrix",
                "Score cluster counts",
                "Publish metrics",
            ),
            running_detail="Scoring candidate cluster counts.",
            success_detail="K-means suggestions are ready.",
        )

        lookup_panel = Section()
        lr = Row()
        left = Section()
        checkpoint_folder = LabeledLineEdit(
            "RVC Checkpoint Folder", placeholder="./logs/model_name"
        )
        checkpoint_model_name = LabeledLineEdit(
            "RVC Model Name", placeholder="voice_model"
        )
        checkpoint_button = QPushButton("Find Latest Checkpoint")
        checkpoint_status = MarkdownLabel("## RVC Checkpoint\n- Status: idle")
        for widget in (
            checkpoint_folder,
            checkpoint_model_name,
            checkpoint_button,
            checkpoint_status,
        ):
            left.addWidget(widget)
        right = Section()
        language_code = LabeledLineEdit("Language Code", placeholder="en")
        language_button = QPushButton("Resolve Language")
        language_button.setProperty("definersPrimary", True)
        language_status = MarkdownLabel("## Language Lookup\n- Status: idle")
        for widget in (language_code, language_button, language_status):
            right.addWidget(widget)
        lr.addWidget(left)
        lr.addWidget(right)
        lookup_panel.addWidget(lr)
        ops_subtabs.addTab(lookup_panel, OPS_TAB_NAMES[2])

        bind_action(
            checkpoint_button,
            handle_rvc_checkpoint_lookup,
            inputs=[checkpoint_folder, checkpoint_model_name],
            outputs=[checkpoint_status],
            action_label="Find Latest Checkpoint",
            steps=(
                "Validate checkpoint path",
                "Resolve latest checkpoint",
                "Publish status",
            ),
            running_detail="Looking up the latest RVC checkpoint.",
            success_detail="Checkpoint lookup is complete.",
        )
        bind_action(
            language_button,
            handle_language_lookup,
            inputs=[language_code],
            outputs=[language_status],
            action_label="Resolve Language",
            steps=(
                "Validate language code",
                "Resolve language",
                "Publish status",
            ),
            running_detail="Resolving the language code.",
            success_detail="Language lookup is ready.",
        )
        tabs.addTab(ops_tab, TRAIN_TAB_NAMES[4])

    return build_main_window(app_title, body)


__all__ = [
    "build_train_app",
    "train_studio_sections",
    "train_studio_tab_names",
]
