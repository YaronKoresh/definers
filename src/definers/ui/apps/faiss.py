class FaissApp:
    @staticmethod
    def launch_faiss_app():
        import gradio as gr

        from definers.ml import build_faiss
        from definers.ui.gradio_shared import (
            bind_progress_click,
            init_output_folder_controls,
            init_progress_tracker,
            launch_blocks,
        )

        with gr.Blocks() as app:
            gr.Markdown("# FAISS Wheel Builder")
            progress_status = init_progress_tracker(
                "FAISS builder ready",
                "Click build when you want to prepare the wheel.",
            )
            init_output_folder_controls(section="faiss")
            build_button = gr.Button("Build FAISS Wheel", variant="primary")
            wheel_output = gr.File(label="Download faiss wheel")
            bind_progress_click(
                build_button,
                build_faiss,
                progress_output=progress_status,
                outputs=[wheel_output],
                action_label="Build FAISS Wheel",
                steps=(
                    "Validate environment",
                    "Build wheel",
                    "Publish artifact",
                ),
                running_detail="Building the FAISS wheel artifact.",
                success_detail="FAISS wheel is ready.",
            )
        launch_blocks(app)


launch_faiss_app = FaissApp.launch_faiss_app
