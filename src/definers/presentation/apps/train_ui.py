from __future__ import annotations


def build_train_app():
    import gradio as gr

    from definers.constants import MAX_INPUT_LENGTH
    from definers.presentation.apps.train_handlers import (
        build_training_plan_markdown,
        handle_prediction,
        handle_training,
    )
    from definers.system import install_ffmpeg

    install_ffmpeg()

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
                        with gr.Row():
                            preview_plan_button = gr.Button("Preview Plan")
                            train_button = gr.Button(
                                "Train", elem_classes="btn"
                            )
                        train_output = gr.File(label="Trained Model Output")
                        training_plan = gr.Markdown(label="Training Plan")
                preview_plan_button.click(
                    fn=build_training_plan_markdown,
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
                    outputs=[training_plan],
                )
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
                    outputs=[train_output, training_plan],
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
                            "Predict", elem_classes="btn"
                        )
                        predict_output = gr.File(label="Prediction Output")
                predict_button.click(
                    fn=handle_prediction,
                    inputs=[model_predict, prediction_data],
                    outputs=[predict_output],
                )
    return app
