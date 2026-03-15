from definers.constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH
from definers.presentation.gradio_shared import launch_blocks


def launch_train_app():
    import gradio as gr

    from definers import infer, system, train

    system.install_ffmpeg()

    def handle_training(
        features,
        labels,
        model_path,
        remote_src,
        dataset_label_columns,
        revision,
        url_type,
        drop_list,
        selected_rows,
    ):
        if selected_rows is not None:
            from definers.ml import simple_text

            if len(selected_rows) > MAX_INPUT_LENGTH:
                raise gr.Error(
                    f"Selected rows input too long ({len(selected_rows)} > {MAX_INPUT_LENGTH})"
                )
            if " " * (MAX_CONSECUTIVE_SPACES + 1) in selected_rows:
                raise gr.Error(
                    "Selected rows contains too many consecutive spaces"
                )
            selected_rows = simple_text(selected_rows)
        return train(
            model_path,
            remote_src,
            revision,
            url_type,
            features,
            labels,
            dataset_label_columns,
            drop_list,
            selected_rows,
        )

    with gr.Blocks() as app:
        gr.Markdown("# Train your own models")
        with gr.Tabs():
            with gr.TabItem("Train"):
                with gr.Row():
                    with gr.Column():
                        model_train = gr.File(
                            label="Upload Model (for re-training)"
                        )
                        remote = gr.Textbox(
                            placeholder="Remote Dataset",
                            label="HuggingFace name or URL",
                        )
                        revision = gr.Textbox(
                            placeholder="Dataset Revision",
                            label="Revision",
                        )
                        url_type = gr.Dropdown(
                            label="Remote Dataset Type",
                            choices=[
                                "parquet",
                                "json",
                                "csv",
                                "arrow",
                                "webdataset",
                                "txt",
                            ],
                        )
                        drop_list = gr.Textbox(
                            placeholder="Ignored Columns (semi-colon separated)",
                            label="Drop List",
                        )
                        label_columns = gr.Textbox(
                            placeholder="Label Columns (semi-colon separated)",
                            label="Label Columns",
                        )
                        selected_rows = gr.Textbox(
                            placeholder="Single rows and ranges (space separated, use a hyphen to choose a range or rows)",
                            label="Selected Rows",
                            max_length=MAX_INPUT_LENGTH,
                        )
                    with gr.Column():
                        local_features = gr.File(
                            label="Local Features",
                            file_count="multiple",
                            allow_reordering=True,
                        )
                        local_labels = gr.File(
                            label="Local Labels (for supervised training)",
                            file_count="multiple",
                            allow_reordering=True,
                        )
                        train_button = gr.Button("Train", elem_classes="btn")
                        train_output = gr.File(label="Trained Model Output")
                train_button.click(
                    fn=handle_training,
                    inputs=[
                        local_features,
                        local_labels,
                        model_train,
                        remote,
                        label_columns,
                        revision,
                        url_type,
                        drop_list,
                        selected_rows,
                    ],
                    outputs=[train_output],
                )
            with gr.TabItem("Predict"):
                with gr.Row():
                    with gr.Column():
                        model_predict = gr.File(
                            label="Upload Model (for prediction)"
                        )
                        prediction_data = gr.File(label="Prediction Data")
                    with gr.Column():
                        predict_button = gr.Button(
                            "Predict",
                            elem_classes="btn",
                        )
                        predict_output = gr.File(label="Prediction Output")
                predict_button.click(
                    fn=infer,
                    inputs=[model_predict, prediction_data],
                    outputs=[predict_output],
                )
    launch_blocks(app)
