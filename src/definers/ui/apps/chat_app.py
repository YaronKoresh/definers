from __future__ import annotations


def launch_chat_app():
    from definers.ui.chat_handlers import get_chat_response_stream
    from definers.ui.gui_shared import (
        init_chat,
        init_output_folder_controls,
        init_progress_tracker,
        launch_blocks,
        qt_app,
    )
    from definers.ui.qt_widgets import (
        MarkdownLabel,
        Section,
        build_main_window,
    )

    qt_app()
    body = Section()
    body.addWidget(MarkdownLabel("# Definers Chat"))
    body.addWidget(
        MarkdownLabel(
            "Production-grade assistant for generation, analysis,"
            " debugging, and delivery decisions."
        )
    )
    body.addWidget(
        init_progress_tracker(
            "Assistant ready",
            "Describe the task, attach context if needed, and send the"
            " request.",
        )
    )
    _, output_toolbar = init_output_folder_controls(section="chat")
    body.addWidget(output_toolbar)
    body.addWidget(init_chat("Definers Chat", get_chat_response_stream))

    window = build_main_window("Definers Chat", body)
    launch_blocks(window)
