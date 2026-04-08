class ChatApp:
    @staticmethod
    def launch_chat_app():
        import gradio as gr

        from definers.ui.chat_handlers import get_chat_response
        from definers.ui.gradio_shared import init_chat, launch_blocks

        with gr.Blocks(title="AI Chatbot") as app:
            init_chat("AI Chatbot", get_chat_response)
        launch_blocks(app)


launch_chat_app = ChatApp.launch_chat_app
