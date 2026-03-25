class GradioShared:
    @staticmethod
    def theme():
        import gradio as gr

        return gr.themes.Base(
            primary_hue=gr.themes.colors.slate,
            secondary_hue=gr.themes.colors.indigo,
            font=(
                gr.themes.GoogleFont("Inter"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ),
        ).set(
            body_background_fill_dark="#111827",
            block_background_fill_dark="#1f2937",
            block_border_width="1px",
            block_title_background_fill_dark="#374151",
            button_primary_background_fill_dark="linear-gradient(90deg, #4f46e5, #7c3aed)",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill_dark="#374151",
            button_secondary_text_color_dark="#ffffff",
            slider_color_dark="#6366f1",
        )

    @staticmethod
    def css() -> str:
        return """
video {
    border-radius: 20px;
}

div:has(>video) {
    padding: 20px 20px 0px 20px !important;
}

span {
    margin-block: 0 !important;
}

label.container > span {
    width: 100% !important;
}

html > body .gradio-container > main *:not(img, svg, span, :has(>svg)):is(*, *::placeholder) {
    scrollbar-width: none !important;
    text-align: center !important;
    max-width: 100% !important;
}

div:not(.styler) > :is(.block:has(+ .block), .block + .block):has(:not(span, div, h1, h2, h3, h4, h5, h6, p, strong)) {
    border: 1px dotted slategray !important;
    margin-block: 10px !important;
}

.row {
    padding-block: 20px !important;
}

label > input[type=\"radio\"] {
    border: 2px ridge black !important;
    flex-grow: 0 !important;
}

label.selected > input[type=\"radio\"] {
    background: lime !important;
}

label:has(>input[type=\"radio\"]) {
    flex-grow: 0 !important;
}

div.form:has(>fieldset.block) {
    margin-block: 10px !important;
}

div.controls {
    width: 100% !important;
}

div.controls > * {
    flex-grow: 1 !important;
}

html > body .gradio-container {
    padding: 0 !important;
}

html > body footer {
    opacity: 0 !important;
    visibility: hidden !important;
    width: 0px !important;
    height: 0px !important;
}

tr.file > td.download {
    min-width: unset !important;
    width: auto !important;
}

html > body main {
    padding-inline: 20px !important;
}

button {
    border-radius: 2mm !important;
    border: none !important;
    cursor: pointer !important;
    margin-inline: 8px !important;
}

button:not(:has(svg)) {
    width: auto !important;
}

textarea {
    border: 1px solid #ccc !important;
    border-radius: 5px !important;
    padding: 8px !important;
}

textarea:focus {
    border-color: #4CAF50 !important;
    outline: none !important;
    box-shadow: 0 0 5px rgba(76, 175, 80, 0.5) !important;
}

h1 {
    color: #333 !important;
}

h2 {
    color: #444 !important;
}

h3 {
    color: #555 !important;
}
"""

    @staticmethod
    def init_chat(title: str = "Chatbot", handler=None):
        import gradio as gr

        from definers.presentation.chat_handlers import get_chat_response

        active_handler = get_chat_response if handler is None else handler
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            type="messages",
            show_copy_button=True,
        )
        return gr.ChatInterface(
            fn=active_handler,
            type="messages",
            chatbot=chatbot,
            multimodal=True,
            title=title,
            save_history=True,
            show_progress="hidden",
            concurrency_limit=None,
        )

    @classmethod
    def launch_blocks(
        cls,
        app,
        *,
        custom_css: str | None = None,
        custom_theme=None,
        server_port: int = 7860,
    ):
        app.launch(
            server_name="0.0.0.0",
            theme=cls.theme() if custom_theme is None else custom_theme,
            css=cls.css() if custom_css is None else custom_css,
            server_port=server_port,
        )


ChatHandler = object
theme = GradioShared.theme
css = GradioShared.css
init_chat = GradioShared.init_chat
launch_blocks = GradioShared.launch_blocks
