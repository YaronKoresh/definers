class ChatApp:
    @staticmethod
    def launch_chat_app():
        import gradio as gr

        from definers.ui.chat_handlers import get_chat_response_stream
        from definers.ui.gradio_shared import (
            init_chat,
            init_output_folder_controls,
            init_progress_tracker,
            launch_blocks,
        )

        with gr.Blocks(title="Definers Chat") as app:
            gr.Markdown("# Definers Chat")
            gr.Markdown(
                "Production-grade assistant for generation, analysis, debugging, and delivery decisions."
            )
            init_progress_tracker(
                "Assistant ready",
                "Describe the task, attach context if needed, and send the request.",
            )
            init_output_folder_controls(section="chat")
            init_chat("Definers Chat", get_chat_response_stream)
        launch_blocks(app)


launch_chat_app = ChatApp.launch_chat_app
