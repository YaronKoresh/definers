from definers.ml import build_faiss
from definers.presentation.gradio_shared import launch_blocks


def launch_faiss_app():
    import gradio as gr

    whl = build_faiss()

    with gr.Blocks() as app:
        gr.File(label="Download faiss wheel", value=whl)
    launch_blocks(app)
