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
        from definers.system.download_activity import (
            create_activity_reporter,
        )

        report = create_activity_reporter(3)
        report(
            1,
            "Validate text",
            detail="Checking the translation input.",
        )
        target_code = TranslateApp.target_language_code(
            tgt_lang, language_codes
        )
        report(
            2,
            "Resolve target language",
            detail=f"Using target language '{target_code}'.",
        )
        report(
            3,
            "Translate paragraphs",
            detail="Running the translation workflow.",
        )
        return text.ai_translate(txt, target_code)

    @staticmethod
    def launch_translate_app():
        import gradio as gr

        from definers.constants import language_codes
        from definers.system import unique
        from definers.ui.gradio_shared import (
            bind_progress_click,
            init_output_folder_controls,
            init_progress_tracker,
            launch_blocks,
        )

        with gr.Blocks() as app:
            gr.Markdown("# AI Translator")
            gr.Markdown(
                "### An AI-based translation software for the community"
            )
            progress_status = init_progress_tracker(
                "Translator ready",
                "Enter text and choose a target language.",
            )
            init_output_folder_controls(section="translate")
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
                        buttons=["copy"],
                    )
            btn = gr.Button(value="Translate")
            bind_progress_click(
                btn,
                TranslateApp.translate_text,
                progress_output=progress_status,
                inputs=[txt, lang],
                outputs=[res],
                action_label="Translate",
                steps=(
                    "Validate text",
                    "Translate content",
                    "Publish result",
                ),
                running_detail="Translating the text.",
                success_detail="Translation is ready.",
            )
        launch_blocks(app)


launch_translate_app = TranslateApp.launch_translate_app
