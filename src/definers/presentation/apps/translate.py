class TranslateApp:
    @staticmethod
    def target_language_code(target_language, language_codes):
        normalized_target_language = str(target_language).strip().lower()
        for code, language_name in language_codes.items():
            if str(language_name).strip().lower() == normalized_target_language:
                return code
        return normalized_target_language

    @staticmethod
    def translate_text(txt, tgt_lang):
        import definers.text as text
        from definers.constants import language_codes

        return text.ai_translate(
            txt,
            TranslateApp.target_language_code(tgt_lang, language_codes),
        )

    @staticmethod
    def launch_translate_app():
        import gradio as gr

        from definers.constants import language_codes
        from definers.presentation.gradio_shared import launch_blocks
        from definers.system import unique

        with gr.Blocks() as app:
            gr.Markdown("# AI Translator")
            gr.Markdown(
                "### An AI-based translation software for the community"
            )
            with gr.Row():
                with gr.Column():
                    txt = gr.Textbox(
                        placeholder="Some text...",
                        value="",
                        lines=4,
                        label="Input",
                        container=True,
                        max_length=2000,
                    )
                    lang = gr.Dropdown(
                        choices=unique(language_codes.values()),
                        value="english",
                    )
                with gr.Column():
                    res = gr.Textbox(
                        label="Results",
                        container=True,
                        value="",
                        lines=6,
                        show_copy_button=True,
                    )
            btn = gr.Button(value="Translate")
            btn.click(
                fn=TranslateApp.translate_text,
                inputs=[txt, lang],
                outputs=[res],
            )
        launch_blocks(app)


launch_translate_app = TranslateApp.launch_translate_app
