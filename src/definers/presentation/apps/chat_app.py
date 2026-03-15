def launch_chat_app():
    import gradio as gr

    from definers import system
    from definers.ml import init_pretrained_model
    from definers.presentation.chat_handlers import get_chat_response
    from definers.presentation.gradio_shared import init_chat, launch_blocks

    init_pretrained_model("summary", True)
    init_pretrained_model("answer", True)
    init_pretrained_model("translate", True)
    system.install_ffmpeg()

    with gr.Blocks(title="AI Chatbot") as app:
        init_chat("AI Chatbot", get_chat_response)
    launch_blocks(app)
