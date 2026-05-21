from __future__ import annotations


def launch_faiss_app():
    from PySide6.QtWidgets import QPushButton

    from definers.ml import build_faiss
    from definers.ui.gui_shared import (
        bind_progress_click,
        init_output_folder_controls,
        init_progress_tracker,
        launch_blocks,
        qt_app,
    )
    from definers.ui.qt_widgets import (
        FileOutput,
        MarkdownLabel,
        Section,
        build_main_window,
    )

    qt_app()
    body = Section()
    body.addWidget(MarkdownLabel("# FAISS Wheel Builder"))
    progress_status = init_progress_tracker(
        "FAISS builder ready",
        "Click build when you want to prepare the wheel.",
    )
    body.addWidget(progress_status)
    _, output_toolbar = init_output_folder_controls(section="faiss")
    body.addWidget(output_toolbar)

    build_button = QPushButton("Build FAISS Wheel")
    build_button.setProperty("definersPrimary", True)
    body.addWidget(build_button)

    wheel_output = FileOutput("Download faiss wheel")
    body.addWidget(wheel_output)

    bind_progress_click(
        build_button,
        build_faiss,
        progress_output=progress_status,
        outputs=[wheel_output],
        action_label="Build FAISS Wheel",
        steps=(
            "Validate environment",
            "Build wheel",
            "Publish artifact",
        ),
        running_detail="Building the FAISS wheel artifact.",
        success_detail="FAISS wheel is ready.",
    )

    window = build_main_window("Definers FAISS Builder", body)
    launch_blocks(window)
