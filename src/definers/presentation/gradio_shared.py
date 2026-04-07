class GradioShared:
    @staticmethod
    def theme():
        import gradio as gr

        return gr.themes.Base(
            primary_hue=gr.themes.colors.emerald,
            secondary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.stone,
            font=(
                gr.themes.GoogleFont("Space Grotesk"),
                gr.themes.GoogleFont("IBM Plex Sans"),
                "ui-sans-serif",
                "sans-serif",
            ),
            font_mono=(
                gr.themes.GoogleFont("IBM Plex Mono"),
                "ui-monospace",
                "monospace",
            ),
        ).set(
            body_background_fill="radial-gradient(circle at top, #f7efe2 0%, #eef2ef 52%, #e6edf1 100%)",
            body_background_fill_dark="radial-gradient(circle at top, #0f172a 0%, #111827 52%, #0b1120 100%)",
            block_background_fill="rgba(255, 255, 255, 0.84)",
            block_background_fill_dark="rgba(15, 23, 42, 0.84)",
            block_border_width="1px",
            block_border_color="rgba(23, 32, 51, 0.12)",
            block_border_color_dark="#334155",
            block_title_text_color="#172033",
            block_title_text_color_dark="#f8fafc",
            body_text_color="#172033",
            body_text_color_dark="#e2e8f0",
            input_background_fill="rgba(255, 255, 255, 0.92)",
            input_background_fill_dark="rgba(15, 23, 42, 0.72)",
            button_primary_background_fill="linear-gradient(135deg, #0f766e 0%, #155e75 100%)",
            button_primary_background_fill_hover="linear-gradient(135deg, #115e59 0%, #164e63 100%)",
            button_primary_background_fill_dark="linear-gradient(135deg, #0f766e 0%, #155e75 100%)",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="#f5f1e9",
            button_secondary_background_fill_hover="#ede5d8",
            button_secondary_background_fill_dark="#1e293b",
            button_secondary_text_color="#172033",
            button_secondary_text_color_dark="#ffffff",
            slider_color="#0f766e",
            slider_color_dark="#14b8a6",
        )

    @staticmethod
    def css() -> str:
        return """
:root {
    --definers-ink: #172033;
    --definers-muted: #5b6679;
    --definers-accent: #0f766e;
    --definers-accent-strong: #155e75;
    --definers-surface: rgba(255, 255, 255, 0.78);
    --definers-border: rgba(23, 32, 51, 0.12);
    --definers-shadow: 0 18px 60px rgba(15, 23, 42, 0.12);
}

.audio-container .timestamps * {
    margin-top: 4mm;
    margin-inline: 2mm;
    color: black !important;
}

html > body > gradio-app > .gradio-container > .main .contain .block {
    margin-bottom: 0 !important;
}

.icon-button-wrapper {
    background: transparent !important;
}

html > body > gradio-app > .gradio-container > .main .contain :is(.settings-wrapper, .control-wrapper) > button {
    min-height: 4mm !important;
    min-width: 4mm !important;
    padding: 0 !important;
}

.label-wrap.open {
    margin-bottom: 0 !important;
}

.app {
    padding: 0 !important;
}

.row > button {
    margin-inline: 1mm;
}

.row div:not(.icon-wrap) {
    min-width: 100% !important;
}

.main {
    margin-block: 1mm !important;
    max-width: 99% !important;
}

h1, h2, p {
    padding: 4mm !important;
}

html > body {
    background: radial-gradient(circle at top, #f7efe2 0%, #eef2ef 52%, #e6edf1 100%);
}

html > body .gradio-container {
    padding: 0 !important;
}

html > body .gradio-container > main {
    max-width: 1440px;
    margin: 0 auto;
}

.controls button {
    margin: auto !important;
}

html > body > gradio-app > .gradio-container > .main .contain button {
    padding: 2mm !important;
    min-height: 0 !important;
    border: none !important;
    color: black !important;
}

html > body > gradio-app > .gradio-container > .main .contain button.primary {
    color: white !important;
}

html > body footer {
    opacity: 0 !important;
    visibility: hidden !important;
    width: 0px !important;
    height: 0px !important;
}

#header,
.audio-hero {
    position: relative;
    overflow: hidden;
    margin-bottom: 18px;
    padding: 30px 34px;
    border: 1px solid var(--definers-border);
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.94) 0%, rgba(233, 245, 243, 0.82) 55%, rgba(229, 236, 241, 0.88) 100%);
    box-shadow: var(--definers-shadow);
}

#header::after,
.audio-hero::after {
    content: "";
    position: absolute;
    inset: auto -8% -35% auto;
    width: 260px;
    height: 260px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(15, 118, 110, 0.18), rgba(21, 94, 117, 0));
}

.audio-hero .eyebrow,
#header .eyebrow {
    margin: 0 0 10px !important;
    color: var(--definers-accent-strong);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.audio-hero h1,
#header h1 {
    margin: 0;
    color: var(--definers-ink) !important;
    font-size: clamp(2.1rem, 3vw, 3.4rem);
    line-height: 1.02;
}

.audio-hero p:last-child,
#header p:last-child {
    max-width: 760px;
    margin: 12px 0 0 !important;
    color: var(--definers-muted);
    font-size: 1rem;
    line-height: 1.6;
}

.tool-container {
    padding: 10px 12px !important;
    border: 1px solid var(--definers-border) !important;
    border-radius: 24px !important;
    background: var(--definers-surface) !important;
    box-shadow: var(--definers-shadow);
    backdrop-filter: blur(18px);
}

.tool-container h2,
.tool-container h3 {
    color: var(--definers-ink) !important;
    letter-spacing: 0.01em;
}

.row {
    gap: 16px !important;
    align-items: stretch !important;
    padding-block: 12px !important;
}

button {
    min-height: 44px !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
}

button.secondary {
    box-shadow: none;
}

html > body > gradio-app > .gradio-container > .main .contain .source-selection,
html > body > gradio-app > .gradio-container > .main .contain .source-selection > button {
    min-height: 1cm !important;
    min-width: 1cm !important;
    box-shadow: none;
}

textarea,
input,
select {
    border-radius: 18px !important;
}

textarea {
    padding: 12px 14px !important;
    border: 1px solid var(--definers-border) !important;
    background: rgba(255, 255, 255, 0.92) !important;
}

textarea:focus,
input:focus,
select:focus {
    border-color: rgba(15, 118, 110, 0.5) !important;
    outline: none !important;
    box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.14) !important;
}

.form,
.block,
fieldset,
.panel,
.accordion {
    border-radius: 0 !important;
}

.accordion {
    border: 1px solid rgba(23, 32, 51, 0.08) !important;
    background: rgba(249, 250, 251, 0.78) !important;
}

video,
img {
    border-radius: 20px;
}

tr.file > td.download {
    min-width: unset !important;
    width: auto !important;
}

.definers-chat-shell {
    padding: 18px !important;
    border: 1px solid var(--definers-border) !important;
    border-radius: 28px !important;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(243, 246, 248, 0.82) 100%) !important;
    box-shadow: var(--definers-shadow);
}

.definers-chat-shell h1 {
    color: var(--definers-ink) !important;
}

.definers-chat-shell p {
    color: var(--definers-muted) !important;
}

.definers-chatbot {
    border: none !important;
    background: transparent !important;
}

.definers-chatbot [class*="message"] {
    border: 1px solid rgba(23, 32, 51, 0.08) !important;
    border-radius: 22px !important;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
}

.definers-chatbot [class*="user"] [class*="message"] {
    background: linear-gradient(135deg, #0f766e 0%, #155e75 100%) !important;
    color: #f8fafc !important;
}

.definers-chatbot [class*="bot"] [class*="message"],
.definers-chatbot [class*="assistant"] [class*="message"] {
    background: rgba(255, 255, 255, 0.94) !important;
    color: var(--definers-ink) !important;
}

.definers-chat-input {
    border: 1px solid rgba(23, 32, 51, 0.1) !important;
    border-radius: 24px !important;
    background: rgba(255, 255, 255, 0.82) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
}

.definers-chat-input textarea {
    min-height: 92px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.block {
    border: 1px solid var(--definers-border) !important;
    background: var(--definers-surface) !important;
    box-shadow: var(--definers-shadow);
    margin-bottom: 18px !important;
    overflow-x: hidden !important;
}

html > body > gradio-app > .gradio-container > .main .contain .styler {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.form {
    overflow-x: hidden !important;
}

@media (max-width: 900px) {
    html > body .gradio-container {
        padding: 14px 14px 24px !important;
    }

    #header,
    .audio-hero,
    .definers-chat-shell {
        padding: 22px 18px !important;
        border-radius: 22px !important;
    }
}
"""

    @staticmethod
    def init_chat(title: str = "Chatbot", handler=None):
        import gradio as gr

        from definers.presentation.chat_handlers import get_chat_response

        active_handler = get_chat_response if handler is None else handler
        with gr.Group(elem_classes="definers-chat-shell"):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                elem_classes="definers-chatbot",
                height=560,
                buttons=["copy", "copy_all"],
            )
            textbox = gr.MultimodalTextbox(
                elem_classes="definers-chat-input",
                placeholder="Describe the task, attach context if needed, then send.",
                show_label=False,
                submit_btn="Send",
                lines=3,
                max_lines=8,
                sources=["upload", "microphone"],
            )
            return gr.ChatInterface(
                fn=active_handler,
                chatbot=chatbot,
                textbox=textbox,
                multimodal=True,
                title=title,
                description="Production-grade assistant for generation, analysis, debugging, and delivery decisions.",
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
