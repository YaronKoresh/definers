class FaissApp:
    @staticmethod
    def launch_faiss_app():
        import gradio as gr

        from definers.ml import build_faiss
        from definers.presentation.gradio_shared import launch_blocks

        wheel_path = build_faiss()

        with gr.Blocks() as app:
            gr.File(label="Download faiss wheel", value=wheel_path)
        launch_blocks(app)


launch_faiss_app = FaissApp.launch_faiss_app
