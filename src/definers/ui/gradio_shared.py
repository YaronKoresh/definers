class GradioShared:
    @staticmethod
    def theme():
        import gradio as gr

        return gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.amber,
            neutral_hue=gr.themes.colors.slate,
            font=(
                gr.themes.GoogleFont("Oxanium"),
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
            body_background_fill="radial-gradient(circle at top, #293447 0%, #111722 24%, #090d15 54%, #03050a 100%)",
            body_background_fill_dark="radial-gradient(circle at top, #293447 0%, #111722 24%, #090d15 54%, #03050a 100%)",
            block_background_fill="linear-gradient(180deg, rgba(74, 84, 99, 0.2) 0%, rgba(30, 36, 46, 0.94) 12%, rgba(12, 15, 22, 0.98) 100%)",
            block_background_fill_dark="linear-gradient(180deg, rgba(74, 84, 99, 0.2) 0%, rgba(30, 36, 46, 0.94) 12%, rgba(12, 15, 22, 0.98) 100%)",
            block_border_width="1px",
            block_border_color="rgba(151, 164, 184, 0.28)",
            block_border_color_dark="rgba(151, 164, 184, 0.28)",
            block_title_text_color="#f3f8ff",
            block_title_text_color_dark="#f3f8ff",
            body_text_color="#d6e1ee",
            body_text_color_dark="#d6e1ee",
            input_background_fill="linear-gradient(180deg, rgba(55, 65, 81, 0.22) 0%, rgba(16, 20, 28, 0.96) 100%)",
            input_background_fill_dark="linear-gradient(180deg, rgba(55, 65, 81, 0.22) 0%, rgba(16, 20, 28, 0.96) 100%)",
            button_primary_background_fill="linear-gradient(180deg, #7be7ff 0%, #27b8ea 16%, #0e597b 54%, #0a2433 100%)",
            button_primary_background_fill_hover="linear-gradient(180deg, #a0f1ff 0%, #42d0ff 18%, #1276a3 54%, #0b2d40 100%)",
            button_primary_background_fill_dark="linear-gradient(180deg, #7be7ff 0%, #27b8ea 16%, #0e597b 54%, #0a2433 100%)",
            button_primary_text_color_dark="#031018",
            button_secondary_background_fill="linear-gradient(180deg, #687385 0%, #313946 18%, #171c25 52%, #0b0f16 100%)",
            button_secondary_background_fill_hover="linear-gradient(180deg, #7a869a 0%, #404a59 18%, #1b212c 52%, #0d121a 100%)",
            button_secondary_background_fill_dark="linear-gradient(180deg, #687385 0%, #313946 18%, #171c25 52%, #0b0f16 100%)",
            button_secondary_text_color="#eef6ff",
            button_secondary_text_color_dark="#eef6ff",
            slider_color="#3fdcff",
            slider_color_dark="#3fdcff",
        )

    @staticmethod
    def css() -> str:
        return """
:root {
    color-scheme: dark;
    --definers-ink: #eaf3ff;
    --definers-ink-strong: #f9fcff;
    --definers-muted: #8d9caf;
    --definers-accent: #54e2ff;
    --definers-accent-strong: #bdf6ff;
    --definers-surface: linear-gradient(180deg, rgba(41, 49, 61, 0.2) 0%, rgba(16, 20, 28, 0.94) 18%, rgba(7, 9, 14, 0.99) 100%);
    --definers-surface-raised: linear-gradient(180deg, rgba(86, 99, 118, 0.24) 0%, rgba(24, 29, 39, 0.96) 12%, rgba(9, 12, 18, 0.99) 100%);
    --definers-surface-deep: linear-gradient(180deg, rgba(48, 57, 70, 0.18) 0%, rgba(12, 16, 23, 0.97) 100%);
    --definers-border: rgba(145, 160, 182, 0.2);
    --definers-border-strong: rgba(84, 226, 255, 0.28);
    --definers-shadow: 0 18px 42px rgba(0, 0, 0, 0.46), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    --definers-shadow-soft: 0 10px 24px rgba(0, 0, 0, 0.34);
    --definers-glow: 0 0 0 1px rgba(84, 226, 255, 0.1), 0 0 28px rgba(84, 226, 255, 0.08);
    --definers-grid: rgba(122, 140, 165, 0.08);
}

@keyframes definers-cockpit-rise {
    from {
        opacity: 0;
        transform: translateY(10px) scale(0.985);
    }

    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes definers-panel-sheen {
    from {
        transform: translateX(-120%);
    }

    to {
        transform: translateX(120%);
    }
}

.audio-container .timestamps time {
    display: flex;
    margin-top: 4mm;
    color: var(--definers-muted);
}

html > body > gradio-app > .gradio-container > .main .contain .block {
    margin-bottom: 18px !important;
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

.row > div:not(.icon-wrap) {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}

.main {
    margin-block: 1mm !important;
    max-width: 99% !important;
    position: relative;
    animation: definers-cockpit-rise 360ms ease-out both;
}

h1, h2, p {
    padding: 4mm !important;
}

html > body {
    background: var(--definers-surface) !important;
    color: var(--definers-ink) !important;
    position: relative;
}

html > body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        radial-gradient(circle at top, rgba(84, 226, 255, 0.12) 0%, rgba(84, 226, 255, 0) 34%),
        radial-gradient(circle at 82% 14%, rgba(255, 191, 102, 0.08) 0%, rgba(255, 191, 102, 0) 20%),
        repeating-linear-gradient(
            90deg,
            transparent 0,
            transparent 22px,
            var(--definers-grid) 23px,
            transparent 24px
        ),
        repeating-linear-gradient(
            180deg,
            transparent 0,
            transparent 22px,
            rgba(84, 226, 255, 0.03) 23px,
            transparent 24px
        );
    opacity: 0.75;
}

html > body::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0));
    mix-blend-mode: screen;
}

html > body .gradio-container {
    padding: 4mm !important;
    background: transparent !important;
}

html > body .gradio-container > main {
    max-width: 1440px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

.controls button {
    margin: auto !important;
}

html > body > gradio-app > .gradio-container > .main .contain button {
    padding: 2mm !important;
    min-height: 0 !important;
    border: 1px solid var(--definers-border) !important;
    color: var(--definers-ink-strong) !important;
    background: var(--definers-surface-deep) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 8px 20px rgba(0, 0, 0, 0.34) !important;
    transition:
        transform 160ms ease,
        filter 160ms ease,
        box-shadow 180ms ease,
        border-color 180ms ease,
        background 180ms ease;
}

html > body > gradio-app > .gradio-container > .main .contain button.primary {
    filter: brightness(1.5) sepia(1);
    text-shadow: none !important;
    box-shadow: 0 0 0 1px rgba(84, 226, 255, 0.22), 0 14px 28px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.32) !important;
}

button:hover {
    transform: translateY(-1px);
    filter: brightness(1.05);
}

button:active {
    transform: translateY(0);
    filter: brightness(0.98);
}

html > body footer {
    display: none !important;
}

.options, .options * {
    position: static !important;
    background: white !important;
    color: black !important;
}

#header,
.audio-hero,
.tool-container,
.definers-chat-shell,
.block,
.accordion {
    position: relative;
    overflow: hidden;
    margin-bottom: 18px;
    padding: 30px 34px;
    border: 1px solid var(--definers-border);
    border-radius: 28px;
    background: var(--definers-surface-raised);
    box-shadow: var(--definers-shadow);
    animation: definers-cockpit-rise 420ms ease-out both;
}

#header::before,
.audio-hero::before,
.tool-container::before,
.definers-chat-shell::before,
.block::before,
.accordion::before {
    content: "";
    position: absolute;
    inset: 1px;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0) 18%, rgba(84, 226, 255, 0.08) 58%, rgba(0, 0, 0, 0) 100%);
    pointer-events: none;
}

#header::after,
.audio-hero::after,
.tool-container::after,
.definers-chat-shell::after {
    content: "";
    position: absolute;
    inset: auto -8% -35% auto;
    width: 260px;
    height: 260px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(84, 226, 255, 0.18), rgba(84, 226, 255, 0));
    filter: blur(2px);
    pointer-events: none;
}

.audio-hero .eyebrow,
#header .eyebrow {
    margin: 0 0 10px !important;
    color: var(--definers-accent-strong);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    text-shadow: 0 0 18px rgba(84, 226, 255, 0.18);
}

.audio-hero h1,
#header h1 {
    margin: 0;
    color: var(--definers-ink-strong) !important;
    font-size: clamp(2.1rem, 3vw, 3.4rem);
    line-height: 1.02;
    text-shadow: 0 10px 26px rgba(0, 0, 0, 0.34);
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
    border-radius: 24px !important;
    backdrop-filter: blur(18px);
    box-shadow: var(--definers-shadow), var(--definers-glow);
}

.tool-container h2,
.tool-container h3 {
    color: var(--definers-ink-strong) !important;
    letter-spacing: 0.01em;
}

.row {
    gap: 16px !important;
    align-items: stretch !important;
    padding-block: 12px !important;
}

button {
    min-height: 44px !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
}

button.secondary {
    box-shadow: var(--definers-shadow-soft), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

html > body > gradio-app > .gradio-container > .main .contain .source-selection,
html > body > gradio-app > .gradio-container > .main .contain .source-selection > button {
    min-height: 1cm !important;
    min-width: 1cm !important;
    box-shadow: none;
}

code {
    background: transparent !important;
    border: none !important;
}

textarea,
input:not([type="radio"]):not([type="checkbox"]):not([type="range"]):not([type="color"]):not([role="listbox"]),
select {
    border-radius: 18px !important;
    border: 1px solid var(--definers-border) !important;
    background: var(--definers-surface-deep) !important;
    color: var(--definers-ink) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    transition:
        transform 160ms ease,
        border-color 180ms ease,
        box-shadow 180ms ease,
        background 180ms ease;
}

textarea {
    padding: 12px 14px !important;
}

textarea:focus,
input:focus,
select:focus {
    border-color: var(--definers-border-strong) !important;
    outline: none !important;
    box-shadow: 0 0 0 4px rgba(84, 226, 255, 0.12), 0 0 12px rgba(84, 226, 255, 0.08) !important;
    transform: translateY(-1px);
}

.form,
.block,
fieldset,
.panel,
.accordion {
    border-radius: 24px !important;
}

.accordion {
    border: 1px solid var(--definers-border) !important;
    background: var(--definers-surface-deep) !important;
    box-shadow: var(--definers-shadow-soft) !important;
}

video,
img {
    border-radius: 20px;
    box-shadow: var(--definers-shadow-soft);
}

tr.file > td.download {
    min-width: unset !important;
    width: auto !important;
}

.definers-chat-shell {
    padding: 18px !important;
    border-radius: 28px !important;
    box-shadow: var(--definers-shadow), var(--definers-glow);
}

.definers-chat-shell h1 {
    color: var(--definers-ink-strong) !important;
}

.definers-chat-shell p {
    color: var(--definers-muted) !important;
}

.definers-chatbot {
    border: none !important;
    background: transparent !important;
}

.definers-chatbot [class*="message"] {
    border: 1px solid var(--definers-border) !important;
    border-radius: 22px !important;
    box-shadow: 0 12px 26px rgba(0, 0, 0, 0.34);
    backdrop-filter: blur(18px);
    transition:
        transform 180ms ease,
        box-shadow 180ms ease,
        border-color 180ms ease;
}

.definers-chatbot [class*="user"] [class*="message"] {
    background: linear-gradient(135deg, rgba(86, 235, 255, 0.98) 0%, rgba(46, 181, 232, 0.94) 34%, rgba(14, 77, 113, 0.98) 100%) !important;
    color: #f8fdff !important;
    border-color: rgba(124, 239, 255, 0.28) !important;
}

.definers-chatbot [class*="bot"] [class*="message"],
.definers-chatbot [class*="assistant"] [class*="message"] {
    background: linear-gradient(180deg, rgba(60, 70, 85, 0.2) 0%, rgba(14, 18, 26, 0.96) 100%) !important;
    color: var(--definers-ink) !important;
}

.definers-chatbot [class*="message"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(84, 226, 255, 0.08);
}

.definers-chat-input {
    border: 1px solid var(--definers-border) !important;
    border-radius: 24px !important;
    background: var(--definers-surface-raised) !important;
    box-shadow: var(--definers-shadow-soft), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.definers-chat-input textarea {
    min-height: 92px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    color: var(--definers-ink) !important;
}

.block {
    border: 1px solid var(--definers-border) !important;
    background: var(--definers-surface) !important;
    box-shadow: var(--definers-shadow), var(--definers-glow);
    margin-bottom: 18px !important;
    overflow-x: hidden !important;
}

html > body > gradio-app > .gradio-container > .main .contain .styler {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.gr-group {
    border: none !important;
    padding: 6px;
    margin-top: 12px;
    border-radius: 16px;
}

div.form {
    background: transparent !important;
    overflow-x: hidden !important;
}

label + .tab-like-container {
    border: none !important;
    padding-inline: 2px;
    filter: drop-shadow(2px 4px 6px black);
}

.tool-container:hover,
.definers-chat-shell:hover,
.block:hover,
.accordion:hover {
    border-color: var(--definers-border-strong) !important;
    box-shadow: var(--definers-shadow), 0 0 0 1px rgba(84, 226, 255, 0.08), 0 0 34px rgba(84, 226, 255, 0.08) !important;
}

.tool-container:hover::before,
.definers-chat-shell:hover::before,
#header:hover::before,
.audio-hero:hover::before {
    animation: definers-panel-sheen 900ms ease;
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
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

        from definers.ui.chat_handlers import get_chat_response

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
