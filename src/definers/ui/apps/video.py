from __future__ import annotations


def launch_video_app(
    visible_tabs=None,
    *,
    app_title="AI VIDEO ARCHITECT",
    hero_eyebrow="Definers Video",
    hero_description=(
        "Compose videos, generate lyric videos, or build a music"
        " visualizer from one focused surface."
    ),
):
    from html import escape

    from PySide6.QtWidgets import QPushButton

    from definers.constants import STYLES_DB
    from definers.ui.gui_shared import (
        bind_progress_click,
        init_output_folder_controls,
        init_progress_tracker,
        launch_blocks,
        qt_app,
        write_widget_value,
    )
    from definers.ui.lyric_video_service import lyric_video
    from definers.ui.music_video_service import music_video
    from definers.ui.qt_widgets import (
        DividerLabel,
        FilePicker,
        HeroCard,
        ImageInput,
        LabeledCheckboxGroup,
        LabeledCombo,
        LabeledFloatSlider,
        LabeledLineEdit,
        LabeledSlider,
        LabeledSpin,
        LabeledTextbox,
        MarkdownLabel,
        Row,
        Section,
        VideoOutput,
        build_main_window,
        build_tab_widget,
    )
    from definers.video.gui import filter_styles, generate_video_handler

    qt_app()

    selected_tabs = set(visible_tabs or ("composer", "lyrics", "visualizer"))

    body = Section()
    body.addWidget(
        HeroCard(
            f"<div>"
            f'<p style="color:#bdf6ff;font-size:11px;font-weight:700;'
            f'letter-spacing:3px;margin:0 0 10px 0;">'
            f"{escape(hero_eyebrow)}</p>"
            f'<div style="color:#f9fcff;font-size:24px;font-weight:700;'
            f'margin:0 0 8px 0;">{escape(app_title)}</div>'
            f'<p style="color:#c8d3e2;font-size:14px;line-height:1.55;'
            f'margin:0;">{escape(hero_description)}</p>'
            f"</div>"
        )
    )
    progress_status = init_progress_tracker(
        "Video workspace ready",
        "Pick a render workflow and start it.",
    )
    body.addWidget(progress_status)
    _, output_toolbar = init_output_folder_controls(section="video")
    body.addWidget(output_toolbar)

    tabs = build_tab_widget()
    body.addWidget(tabs)

    if "composer" in selected_tabs:
        composer = Section()
        composer.addWidget(
            MarkdownLabel("### Advanced Composition & Layout Engine")
        )

        media_header = DividerLabel("### Media & Style")
        composer.addWidget(media_header)

        audio_in = FilePicker(
            "Audio File",
            file_filter="Audio (*.wav *.mp3 *.flac *.ogg *.m4a)",
        )
        composer.addWidget(audio_in)

        search_cat_row = Row()
        search_txt = LabeledLineEdit("Search", placeholder="Search styles...")
        cat_filter = LabeledCombo(
            "Category Filter",
            choices=(
                "All",
                "Abstract",
                "Cyberpunk",
                "Retro",
                "Simple",
                "Sci-Fi",
            ),
            value="All",
        )
        search_cat_row.addWidget(search_txt)
        search_cat_row.addWidget(cat_filter)
        composer.addWidget(search_cat_row)

        style_dd = LabeledCombo(
            "Select Style (Base Layer)",
            choices=list(STYLES_DB.keys()),
            value="Psychedelic Geometry",
        )
        composer.addWidget(style_dd)

        def _refresh_style_choices(*_args: object) -> None:
            payload = filter_styles(search_txt.value(), cat_filter.value())
            write_widget_value(style_dd, payload)

        search_txt._editor.textChanged.connect(_refresh_style_choices)
        cat_filter.changed.connect(_refresh_style_choices)

        controls_row = Row()
        left_col = Section()
        image_in = ImageInput("Background Image", height=220)
        resolution = LabeledCombo(
            "Resolution",
            choices=(
                "Square (1:1)",
                "Portrait (9:16)",
                "Landscape (16:9)",
            ),
            value="Landscape (16:9)",
        )
        fps = LabeledSlider("FPS", minimum=1, maximum=60, value=20, step=1)
        sensitivity = LabeledFloatSlider(
            "Sensitivity",
            minimum=0.1,
            maximum=5.0,
            value=1.0,
            step=0.1,
        )
        reactivity = LabeledCombo(
            "Reactivity Band",
            choices=("Low", "Mid", "High", "Full"),
            value="Full",
        )
        left_col.addWidget(image_in)
        left_col.addWidget(resolution)
        left_col.addWidget(fps)
        left_col.addWidget(sensitivity)
        left_col.addWidget(reactivity)

        right_col = Section()
        palette = LabeledCombo(
            "Color Palette",
            choices=list(STYLES_DB.keys()),
            value=list(STYLES_DB.keys())[0],
        )
        overlays = LabeledCheckboxGroup(
            "Overlays",
            choices=(
                "Neon Border",
                "Progress Bar",
                "Audio Waveform",
                "Timer",
            ),
        )
        effects = LabeledCheckboxGroup(
            "Post Effects",
            choices=("Vignette", "Scanlines", "Noise"),
        )
        custom_elem = LabeledCombo(
            "Custom Element",
            choices=(
                "None",
                "Custom Text",
                "Logo Image",
                "Spectrum Circle",
            ),
            value="None",
        )
        right_col.addWidget(palette)
        right_col.addWidget(overlays)
        right_col.addWidget(effects)
        right_col.addWidget(custom_elem)

        controls_row.addWidget(left_col)
        controls_row.addWidget(right_col)
        composer.addWidget(controls_row)

        ce_x = LabeledFloatSlider(
            "Element X",
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.01,
        )
        ce_y = LabeledFloatSlider(
            "Element Y",
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.01,
        )
        ce_scale = LabeledFloatSlider(
            "Element Scale",
            minimum=0.1,
            maximum=5.0,
            value=1.0,
            step=0.05,
        )
        ce_opacity = LabeledFloatSlider(
            "Element Opacity",
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.01,
        )
        ce_text = LabeledLineEdit("Custom Text")
        ce_logo = FilePicker(
            "Custom Logo",
            file_filter="Images (*.png *.jpg *.jpeg *.webp)",
        )
        for widget in (ce_x, ce_y, ce_scale, ce_opacity, ce_text, ce_logo):
            composer.addWidget(widget)

        composer_button = QPushButton("Generate Video")
        composer_button.setProperty("definersPrimary", True)
        composer.addWidget(composer_button)
        composer_output = VideoOutput("Output Video")
        composer.addWidget(composer_output)

        bind_progress_click(
            composer_button,
            generate_video_handler,
            progress_output=progress_status,
            inputs=[
                audio_in,
                image_in,
                style_dd,
                resolution,
                fps,
                sensitivity,
                reactivity,
                palette,
                overlays,
                effects,
                custom_elem,
                ce_x,
                ce_y,
                ce_scale,
                ce_opacity,
                ce_text,
                ce_logo,
            ],
            outputs=[composer_output],
            action_label="Generate Video",
            steps=(
                "Validate media",
                "Render composition",
                "Publish result",
            ),
            running_detail="Rendering the video composition.",
            success_detail="Video render is ready.",
        )
        tabs.addTab(composer, "Composer")

    if "lyrics" in selected_tabs:
        lyrics = Section()
        lyric_header = DividerLabel("### Lyric Video Creator")
        lyrics.addWidget(lyric_header)
        lyric_audio = FilePicker(
            "Audio File",
            file_filter="Audio (*.wav *.mp3 *.flac *.ogg *.m4a)",
        )
        lyric_background = ImageInput("Background Image", height=220)
        lyric_text = LabeledTextbox("Lyrics", lines=6)
        lyric_position = LabeledCombo(
            "Text Position",
            choices=("top", "center", "bottom"),
            value="bottom",
        )
        lyric_max_dimension = LabeledSpin(
            "Max Dimension",
            minimum=64,
            maximum=4096,
            value=640,
        )
        lyric_font_size = LabeledSpin(
            "Font Size",
            minimum=8,
            maximum=512,
            value=70,
        )
        lyric_color = LabeledLineEdit("Text Color", value="white")
        lyric_stroke_color = LabeledLineEdit("Stroke Color", value="black")
        lyric_stroke_width = LabeledSlider(
            "Stroke Width",
            minimum=0,
            maximum=10,
            value=2,
            step=1,
        )
        lyric_fade_duration = LabeledFloatSlider(
            "Fade Duration",
            minimum=0.0,
            maximum=5.0,
            value=0.5,
            step=0.1,
        )
        for widget in (
            lyric_audio,
            lyric_background,
            lyric_text,
            lyric_position,
            lyric_max_dimension,
            lyric_font_size,
            lyric_color,
            lyric_stroke_color,
            lyric_stroke_width,
            lyric_fade_duration,
        ):
            lyrics.addWidget(widget)
        lyric_button = QPushButton("Make Lyric Video")
        lyric_button.setProperty("definersPrimary", True)
        lyrics.addWidget(lyric_button)
        lyric_output = VideoOutput("Lyric Output")
        lyrics.addWidget(lyric_output)

        bind_progress_click(
            lyric_button,
            lyric_video,
            progress_output=progress_status,
            inputs=[
                lyric_audio,
                lyric_background,
                lyric_text,
                lyric_position,
                lyric_max_dimension,
                lyric_font_size,
                lyric_color,
                lyric_stroke_color,
                lyric_stroke_width,
                lyric_fade_duration,
            ],
            outputs=[lyric_output],
            action_label="Make Lyric Video",
            steps=(
                "Validate media",
                "Render lyric video",
                "Publish result",
            ),
            running_detail="Rendering the lyric video.",
            success_detail="Lyric video is ready.",
        )
        tabs.addTab(lyrics, "Lyric Video")

    if "visualizer" in selected_tabs:
        visualizer = Section()
        viz_header = DividerLabel("### Music Visualizer")
        visualizer.addWidget(viz_header)
        visualizer_audio = FilePicker(
            "Audio File",
            file_filter="Audio (*.wav *.mp3 *.flac *.ogg *.m4a)",
        )
        visualizer_width = LabeledSpin(
            "Width",
            minimum=64,
            maximum=8192,
            value=1920,
        )
        visualizer_height = LabeledSpin(
            "Height",
            minimum=64,
            maximum=8192,
            value=1080,
        )
        visualizer_fps = LabeledSlider(
            "FPS",
            minimum=1,
            maximum=60,
            value=30,
            step=1,
        )
        for widget in (
            visualizer_audio,
            visualizer_width,
            visualizer_height,
            visualizer_fps,
        ):
            visualizer.addWidget(widget)
        visualizer_button = QPushButton("Generate Visualizer")
        visualizer_button.setProperty("definersPrimary", True)
        visualizer.addWidget(visualizer_button)
        visualizer_output = VideoOutput("Visualizer Output")
        visualizer.addWidget(visualizer_output)

        bind_progress_click(
            visualizer_button,
            music_video,
            progress_output=progress_status,
            inputs=[
                visualizer_audio,
                visualizer_width,
                visualizer_height,
                visualizer_fps,
            ],
            outputs=[visualizer_output],
            action_label="Generate Visualizer",
            steps=(
                "Validate audio",
                "Render visualizer",
                "Publish result",
            ),
            running_detail="Rendering the music visualizer.",
            success_detail="Visualizer video is ready.",
        )
        tabs.addTab(visualizer, "Visualizer")

    window = build_main_window(app_title, body)
    launch_blocks(window)
