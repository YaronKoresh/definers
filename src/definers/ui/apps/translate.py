from __future__ import annotations


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
        from PySide6.QtWidgets import QPushButton

        from definers.constants import language_codes
        from definers.system import unique
        from definers.ui.gui_shared import (
            bind_progress_click,
            init_output_folder_controls,
            init_progress_tracker,
            launch_blocks,
            qt_app,
        )
        from definers.ui.qt_widgets import (
            LabeledCombo,
            LabeledTextbox,
            MarkdownLabel,
            Row,
            Section,
            build_main_window,
        )

        qt_app()
        body = Section()
        body.addWidget(MarkdownLabel("# AI Translator"))
        body.addWidget(
            MarkdownLabel(
                "### An AI-based translation software for the community"
            )
        )
        progress_status = init_progress_tracker(
            "Translator ready",
            "Enter text and choose a target language.",
        )
        body.addWidget(progress_status)
        _, output_toolbar = init_output_folder_controls(section="translate")
        body.addWidget(output_toolbar)

        row = Row()
        left = Section()
        txt = LabeledTextbox(
            "Input",
            placeholder="Some text...",
            lines=4,
            max_length=2000,
        )
        lang = LabeledCombo(
            "Target Language",
            choices=unique(language_codes.values()),
            value="english",
        )
        left.addWidget(txt)
        left.addWidget(lang)
        right = Section()
        res = LabeledTextbox("Results", lines=6)
        right.addWidget(res)
        row.addWidget(left)
        row.addWidget(right)
        body.addWidget(row)

        btn = QPushButton("Translate")
        btn.setProperty("definersPrimary", True)
        body.addWidget(btn)

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

        window = build_main_window("Definers Translator", body)
        launch_blocks(window)
